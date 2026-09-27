"""DecisionEngine: load a causal LM, verify answer slots, and answer a System One request with
one shared-prefix prefill and one batched suffix forward (no token is ever sampled).

Batched mode forks the prefix cache along the batch dimension, including the convolutional and
recurrent states of linear-attention layers (Qwen3.5), which the generic cache API does not fork.
The fork is exact: in float32, batched and sequential branch probabilities agree to 3e-4 on
Qwen3.5-4B. In bfloat16 the two paths differ by batch-dependent kernel rounding (up to about 0.03
in probability on near-tied options), so the load-time check compares probabilities with a
tolerance of 0.05 and only guards against structural failures (an unknown cache layer type).
"""
from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .answers import make_choice, make_noul, make_score
from .calibrate import Calibration, softmax
from .prompt import (DEFAULT_STYLE, LETTERS, MAX_SLOTS, PROMPT_VERSION, SLOT_LABELS, PromptStyle, RenderedQuestion,
                     build_messages, render_question)
from .schema import ChoiceQuestion
from .schema import Answer, SystemOneRequest, SystemOneResponse, Usage

log = logging.getLogger("qev.engine")

NONE_OPTION = "none of the options listed here"
NONE_DESCRIPTION = "The correct option is not among the options listed in this question."


@dataclass
class BranchResult:
    qid: str
    kind: str
    labels: list[str]
    descriptions: list[Any]
    logits: list[float]
    allowed_mass: float
    suffix_tokens: int


@dataclass
class EngineInfo:
    model_path: str
    model_type: str
    dtype: str
    device: str
    fork_mode: str
    fork_max_abs_diff: float | None
    prompt_version: str = PROMPT_VERSION
    transformers_version: str = ""
    torch_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        calibration: Calibration | None = None,
        max_branch_tokens: int = 32768,
        max_request_tokens: int = 65536,
        fork_token_budget: int = 200_000,
        fork_mode: str = "auto",
        fork_tolerance: float = 5e-2,
        style: PromptStyle = DEFAULT_STYLE,
        head_dtype: str = "float32",
    ) -> None:
        import torch
        import transformers

        self.model_path = model_path
        self.device = torch.device(device)
        self.torch_dtype = getattr(torch, dtype)
        self.calibration = calibration or Calibration()
        self.max_branch_tokens = max_branch_tokens
        self.max_request_tokens = max_request_tokens
        self.fork_token_budget = fork_token_budget
        self.fork_tolerance = fork_tolerance
        self.style = style

        t0 = time.perf_counter()
        local = Path(model_path).exists()
        common = {"local_files_only": local, "trust_remote_code": False}
        config = transformers.AutoConfig.from_pretrained(model_path, **common)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, **common)
        cls = transformers.AutoModelForCausalLM
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            cls = transformers.Qwen3_5ForCausalLM
            config = config.get_text_config()
        self.model = cls.from_pretrained(model_path, config=config, dtype=self.torch_dtype,
                                         device_map={"": str(self.device)}, low_cpu_mem_usage=True, **common)
        self.model.eval()
        self.head_dtype = head_dtype
        if head_dtype != dtype:
            self._upcast_head(getattr(torch, head_dtype))
        self.model_type = config.model_type
        self.template_kwargs = {"enable_thinking": False} if "qwen" in self.model_type else {}
        self.slot_ids = self._verify_slots()
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.info = EngineInfo(model_path=model_path, model_type=self.model_type, dtype=dtype, device=str(self.device),
                               fork_mode="sequential", fork_max_abs_diff=None,
                               transformers_version=transformers.__version__, torch_version=torch.__version__,
                               extra={"head_dtype": head_dtype})
        if fork_mode == "auto":
            self._select_fork_mode()
        else:
            self.info.fork_mode = fork_mode
        log.info("loaded %s (%s) in %.1fs, fork_mode=%s", model_path, self.model_type, time.perf_counter() - t0, self.info.fork_mode)

    def _upcast_head(self, head_dtype) -> None:
        """Compute the LM head in a wider dtype. bf16 logits at magnitude 25 to 50 are quantized to
        0.125, which makes two-option probabilities move in steps of about 3 percent and lets options
        tie exactly. The head weight is untied from the embedding (a clone) so the rest of the forward
        stays in the model dtype; a pre-hook casts the readout hidden states to the head dtype."""
        import torch

        head = self.model.get_output_embeddings()
        if head is None or not hasattr(head, "weight"):
            log.warning("model has no separable output head; logits stay in %s", self.torch_dtype)
            return
        head.weight = torch.nn.Parameter(head.weight.detach().clone().to(head_dtype), requires_grad=False)
        if getattr(head, "bias", None) is not None:
            head.bias = torch.nn.Parameter(head.bias.detach().clone().to(head_dtype), requires_grad=False)
        head.register_forward_pre_hook(lambda module, args: (args[0].to(head_dtype),) + tuple(args[1:]))

    def _verify_slots(self) -> list[int]:
        """Slot token ids for every label the tokenizer holds as one round-trip token. A..Z are required. The
        two-letter labels are all-or-nothing: a tokenizer that splits any of them keeps 26 slots, and choices
        with more options go through the chunked path."""
        def one(label):
            enc = self.tokenizer.encode(label, add_special_tokens=False)
            return enc[0] if len(enc) == 1 and self.tokenizer.decode(enc) == label else None

        ids = [one(l) for l in LETTERS]
        if any(i is None for i in ids):
            bad = [l for l, i in zip(LETTERS, ids) if i is None]
            raise ValueError(f"answer slots {bad} are not single round-trip tokens in {self.model_path}")
        extra = [one(l) for l in SLOT_LABELS[len(LETTERS):]]
        if all(i is not None for i in extra):
            ids += extra
        else:
            log.warning("tokenizer splits some two-letter labels; %d slots, larger choices are chunked", len(ids))
        if len(set(ids)) != len(ids):
            raise ValueError("answer slot tokens collide")
        self.max_slots = len(ids)
        return ids

    def _render_prompt(self, state: Any, suffix: str) -> str:
        return self.tokenizer.apply_chat_template(build_messages(state, suffix, self.style), tokenize=False,
                                                  add_generation_prompt=True, **self.template_kwargs)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _select_fork_mode(self) -> None:
        """Compare batched-fork logits with sequential logits on a small synthetic request."""
        import torch

        state = {"ticket": "My card was charged twice for the same order and support has not replied in 3 days.",
                 "customer": {"tenure_months": 14, "vip": False}}
        from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
        qs = {
            "a": NoulQuestion(type="noul", instructions="Is the customer asking for money back?"),
            "b": ChoiceQuestion(type="choice", instructions="Which team should handle this?",
                                criteria={"billing": "charges and refunds", "technical": "bugs", "sales": "pricing"}),
            "c": ScoreQuestion(type="score", instructions="How frustrated is the customer?",
                               criteria=["calm", "irritated", "angry"]),
        }
        rendered = [render_question(k, q, style=self.style) for k, q in qs.items()]
        try:
            seq = self._run_sequential(state, rendered)
            bat = self._run_batched(state, rendered)
        except Exception as exc:
            log.warning("batched fork unavailable (%s); using sequential branches", exc)
            self.info.fork_mode, self.info.fork_max_abs_diff = "sequential", None
            return
        diff = max(abs(a - b) for s, bb in zip(seq, bat) for a, b in zip(softmax(s.logits), softmax(bb.logits)))
        self.info.fork_max_abs_diff = float(diff)
        self.info.fork_mode = "batched" if diff <= self.fork_tolerance else "sequential"
        if self.info.fork_mode == "sequential":
            log.warning("batched fork differs from sequential by %.4f in probability (> %.4f); using sequential branches", diff, self.fork_tolerance)
        else:
            log.info("batched fork verified: max probability difference vs sequential %.5f", diff)
        with torch.no_grad():
            torch.cuda.empty_cache() if self.device.type == "cuda" else None

    def _prefix_and_suffix_ids(self, state: Any, rendered: list[RenderedQuestion]) -> tuple[list[int], list[list[int]]]:
        """Tokenize every branch fully, then find the longest common token prefix that ends at or
        before the end of the state block, so the shared prefix is exact by construction."""
        full = [self._encode(self._render_prompt(state, r.suffix)) for r in rendered]
        prefix_text = self._render_prompt(state, "QEV_PREFIX_MARK")
        cut = prefix_text.index("QEV_PREFIX_MARK")
        prefix_ids = self._encode(prefix_text[:cut])
        n = len(prefix_ids)
        while n > 0 and not all(ids[:n] == prefix_ids[:n] for ids in full):
            n -= 1
        prefix_ids = prefix_ids[:n]
        suffixes = [ids[n:] for ids in full]
        if any(len(s) == 0 for s in suffixes):
            raise ValueError("a branch has an empty suffix; state block boundary mismatch")
        return prefix_ids, suffixes

    def _readout(self, vocab_logits, n_slots: int) -> tuple[list[float], float]:
        import torch

        v = vocab_logits.float()
        slots = self.slot_ids[:n_slots]
        sel = v[slots]
        mass = float((torch.logsumexp(sel, -1) - torch.logsumexp(v, -1)).exp())
        return sel.tolist(), mass

    def _run_sequential(self, state: Any, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        import torch

        prefix_ids, suffixes = self._prefix_and_suffix_ids(state, rendered)
        dev = self.device
        with torch.inference_mode():
            out = self.model(input_ids=torch.tensor([prefix_ids], device=dev),
                             attention_mask=torch.ones((1, len(prefix_ids)), dtype=torch.long, device=dev),
                             use_cache=True, return_dict=True, logits_to_keep=1)
            base = out.past_key_values
            del out
            results = []
            for r, suf in zip(rendered, suffixes):
                cache = copy.deepcopy(base)
                o = self.model(input_ids=torch.tensor([suf], device=dev),
                               attention_mask=torch.ones((1, len(prefix_ids) + len(suf)), dtype=torch.long, device=dev),
                               past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
                logits, mass = self._readout(o.logits[0, -1], r.n_slots)
                results.append(BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, mass, len(suf)))
                del o, cache
            del base
        return results

    def _fork(self, cache, n: int):
        """Duplicate a batch-1 cache into n rows. Handles attention layers (keys/values) and
        linear-attention layers (conv and recurrent states), which the generic
        `DynamicCache.batch_repeat_interleave` does not fork in transformers 5.17."""
        import torch

        forked = copy.deepcopy(cache)
        layers = getattr(forked, "layers", None)
        if layers is None:
            raise RuntimeError(f"unsupported cache object {type(cache).__name__}")
        for layer in layers:
            if hasattr(layer, "batch_repeat_interleave"):
                layer.batch_repeat_interleave(n)
            elif hasattr(layer, "conv_states") and hasattr(layer, "recurrent_states"):
                for states in (layer.conv_states, layer.recurrent_states):
                    for k, v in list(states.items()):
                        if torch.is_tensor(v):
                            states[k] = v.repeat_interleave(n, dim=0)
            else:
                raise RuntimeError(f"cannot fork cache layer {type(layer).__name__}")
        return forked

    def _run_batched(self, state: Any, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        """Prefill once, then run branches in chunks off the forked cache. The chunk size starts
        from the token budget and halves on CUDA out-of-memory; at chunk size 1 it gives up and the
        caller falls back to sequential branches."""
        import torch

        prefix_ids, suffixes = self._prefix_and_suffix_ids(state, rendered)
        dev = self.device
        P = len(prefix_ids)
        chunk = max(1, min(len(rendered), self.fork_token_budget // max(P, 1)))
        results: list[BranchResult | None] = [None] * len(rendered)
        with torch.inference_mode():
            out = self.model(input_ids=torch.tensor([prefix_ids], device=dev),
                             attention_mask=torch.ones((1, P), dtype=torch.long, device=dev),
                             use_cache=True, return_dict=True, logits_to_keep=1)
            base = out.past_key_values
            del out
            start = 0
            while start < len(rendered):
                idx = list(range(start, min(start + chunk, len(rendered))))
                try:
                    self._run_chunk(base, P, idx, rendered, suffixes, results)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if chunk == 1:
                        del base
                        raise
                    chunk = max(1, chunk // 2)
                    log.warning("batched branches out of memory at prefix %d tokens; retrying with chunk %d", P, chunk)
                    continue
                start += len(idx)
            del base
        return results

    def _run_chunk(self, base, P: int, idx: list[int], rendered, suffixes, results) -> None:
        import torch

        dev = self.device
        sufs = [suffixes[i] for i in idx]
        width = max(len(s) for s in sufs)
        ids = torch.full((len(idx), width), self.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((len(idx), P + width), dtype=torch.long, device=dev)
        pos = torch.zeros((len(idx), width), dtype=torch.long, device=dev)
        ends = []
        for row, s in enumerate(sufs):
            ids[row, : len(s)] = torch.tensor(s, device=dev)
            mask[row, : P + len(s)] = 1
            pos[row, : len(s)] = torch.arange(P, P + len(s), device=dev)
            ends.append(len(s) - 1)
        keep = sorted(set(ends))
        cache = self._fork(base, len(idx))
        try:
            o = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache,
                           use_cache=True, return_dict=True,
                           logits_to_keep=torch.tensor(keep, dtype=torch.long, device=dev))
            for row, i in enumerate(idx):
                r = rendered[i]
                logits, mass = self._readout(o.logits[row, keep.index(ends[row])], r.n_slots)
                results[i] = BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, mass, len(sufs[row]))
            del o
        finally:
            del cache

    def _plan_chunks(self, q: ChoiceQuestion) -> list[list[str]]:
        """Split a many-option choice into chunks of at most max_slots - 1 options (plus one shared
        "none of the options listed here" slot per chunk), balanced in size."""
        names = list(q.criteria.keys())
        n_chunks = -(-len(names) // (self.max_slots - 1))
        size = -(-len(names) // n_chunks)
        return [names[i:i + size] for i in range(0, len(names), size)]

    def decide(self, request: SystemOneRequest, debias: int = 1, debug: bool = False) -> tuple[SystemOneResponse, dict[str, Any]]:
        """Answer a request. `debias > 1` averages probabilities over that many cyclic shifts of
        the option order (all sharing the same prefix)."""
        t0 = time.perf_counter()
        rendered: list[RenderedQuestion] = []
        owners: list[tuple[str, list[int], int | None]] = []
        chunk_plans: dict[str, list[list[str]]] = {}
        for qid, q in request.questions.items():
            if isinstance(q, ChoiceQuestion) and len(q.criteria) > self.max_slots:
                chunk_plans[qid] = self._plan_chunks(q)
                for ci, names in enumerate(chunk_plans[qid]):
                    sub = ChoiceQuestion(type="choice", instructions=q.instructions,
                                         criteria={**{n: q.criteria[n] for n in names}, NONE_OPTION: NONE_DESCRIPTION})
                    rendered.append(render_question(qid, sub, style=self.style))
                    owners.append((qid, list(range(len(names) + 1)), ci))
                continue
            base = render_question(qid, q, style=self.style)
            k = base.n_slots
            shifts = range(min(max(debias, 1), k)) if k > 1 else range(1)
            for s in shifts:
                order = [(i + s) % k for i in range(k)]
                rendered.append(render_question(qid, q, order=order if s else None, style=self.style))
                owners.append((qid, order, None))

        prefix_ids, suffixes = self._prefix_and_suffix_ids(request.state, rendered)
        longest = len(prefix_ids) + max(len(s) for s in suffixes)
        total = len(prefix_ids) + sum(len(s) for s in suffixes)
        if longest > self.max_branch_tokens:
            raise ValueError(f"state plus longest question is {longest} tokens, limit is {self.max_branch_tokens}")
        if total > self.max_request_tokens:
            raise ValueError(f"request is {total} tokens, limit is {self.max_request_tokens}")

        t1 = time.perf_counter()
        if self.info.fork_mode == "batched":
            try:
                branches = self._run_batched(request.state, rendered)
            except Exception as exc:
                if type(exc).__name__ != "OutOfMemoryError":
                    raise
                log.warning("batched path out of memory for %d branches over %d prefix tokens; running sequentially", len(rendered), len(prefix_ids))
                branches = self._run_sequential(request.state, rendered)
        else:
            branches = self._run_sequential(request.state, rendered)
        t2 = time.perf_counter()

        acc: dict[str, dict[str, Any]] = {}
        chunked: dict[str, dict[str, Any]] = {}
        for br, (qid, order, ci) in zip(branches, owners):
            temp = self.calibration.temperature(br.kind, len(br.labels))
            probs = softmax(br.logits, temp)
            if ci is not None:
                c = chunked.setdefault(qid, {"scores": {}, "weights": [], "mass": 0.0, "n": 0, "logits": []})
                none_p = probs[br.labels.index(NONE_OPTION)]
                keep = max(1.0 - none_p, 1e-9)
                for lab, p in zip(br.labels, probs):
                    if lab != NONE_OPTION:
                        c["scores"][lab] = (p / keep, keep)
                c["weights"].append(keep)
                c["mass"] += br.allowed_mass
                c["n"] += 1
                c["logits"].append(br.logits)
                continue
            slot = acc.setdefault(qid, {"kind": br.kind, "probs": None, "labels": None, "descriptions": None,
                                        "n": 0, "mass": 0.0, "logits": []})
            if slot["probs"] is None:
                k = len(br.labels)
                slot["probs"] = [0.0] * k
                slot["labels"] = [None] * k
                slot["descriptions"] = [None] * k
            for pos_i, orig_i in enumerate(order):
                slot["probs"][orig_i] += probs[pos_i]
                slot["labels"][orig_i] = br.labels[pos_i]
                slot["descriptions"][orig_i] = br.descriptions[pos_i]
            slot["n"] += 1
            slot["mass"] += br.allowed_mass
            slot["logits"].append(br.logits)

        answers: dict[str, Answer] = {}
        diag: dict[str, Any] = {}
        for qid, c in chunked.items():
            total_w = sum(c["weights"]) or 1.0
            names = list(request.questions[qid].criteria.keys())
            probs = [c["scores"][n][0] * (c["scores"][n][1] / total_w) for n in names]
            answers[qid] = make_choice(names, probs)
            diag[qid] = {"allowed_token_mass": c["mass"] / c["n"], "slot_logits": c["logits"], "chunks": c["n"],
                         "temperature": self.calibration.temperature("choice", self.max_slots)}
        for qid, slot in acc.items():
            probs = [p / slot["n"] for p in slot["probs"]]
            if slot["kind"] == "noul":
                answers[qid] = make_noul(probs[0])
            elif slot["kind"] == "choice":
                answers[qid] = make_choice(slot["labels"], probs)
            else:
                answers[qid] = make_score(slot["descriptions"], probs)
            diag[qid] = {"allowed_token_mass": slot["mass"] / slot["n"], "slot_logits": slot["logits"],
                         "temperature": self.calibration.temperature(slot["kind"], len(probs))}

        usage = Usage(input_tokens=total, output_tokens=len(request.questions))
        response = SystemOneResponse(model="", answers=answers, usage=usage)
        meta = {
            "prompt_version": self.style.name,
            "prompt_sha256": hashlib.sha256(self._render_prompt(request.state, rendered[0].suffix).encode()).hexdigest(),
            "fork_mode": self.info.fork_mode,
            "prefix_tokens": len(prefix_ids),
            "branches": len(rendered),
            "timing": {"render_s": t1 - t0, "forward_s": t2 - t1, "total_s": time.perf_counter() - t0},
        }
        if debug:
            meta["diagnostics"] = diag
        return response, meta
