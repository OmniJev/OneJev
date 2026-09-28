"""Shared pieces for training, calibration and heldout evaluation.

Every string the model sees is produced by qev.prompt, exactly as qev.engine does it,
so a checkpoint trained here renders identically when served.
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from qev.prompt import SLOT_LABELS, build_messages, render_question
from qev.schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

log = logging.getLogger("train.common")

KINDS = ("noul", "choice", "score")


def load_rows(path: str | Path, limit: int | None = None, split: str | None = None) -> list[dict]:
    """Read the unified JSONL example format of DESIGN.md 4.1."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split is not None and row.get("split") != split:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_question(qdict: dict) -> Question:
    """Dict to the pydantic question model that qev.prompt.render_question expects."""
    kind = qdict.get("type")
    if kind == "noul":
        return NoulQuestion(**qdict)
    if kind == "choice":
        return ChoiceQuestion(**qdict)
    if kind == "score":
        return ScoreQuestion(**qdict)
    raise ValueError(f"unknown question type {kind!r}")


def target_vector(target: dict[str, float], labels: Sequence[str]) -> list[float]:
    """Target probabilities in slot order. Labels come from the rendered question, so a permuted
    rendering permutes the target for free."""
    vec = [float(target.get(lbl, 0.0)) for lbl in labels]
    total = sum(vec)
    if total <= 0:
        raise ValueError(f"target {target} has no mass on the rendered labels {list(labels)}")
    return [v / total for v in vec]


def serialize_state(state: Any, compact: bool) -> Any:
    """Light template augmentation on the state serialization only.

    qev.prompt.render_state pretty-prints non-string states with indent 2 and passes strings
    through verbatim, so pre-serializing here with indent 2 is byte-identical to production and
    the compact form is the only thing that varies. The question text is never touched.
    """
    if isinstance(state, str) or state is None:
        return state
    if compact:
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(state, ensure_ascii=False, indent=2)


@dataclass
class Example:
    """One rendered training or evaluation item."""

    idx: int
    prompt: str
    kind: str
    labels: list[str]
    target: list[float]
    source: str
    task: str


PAD_FAMILIES = ("next_action",)
PAD_MAX_OPTIONS = 255


def pad_key(row: dict) -> str | None:
    """Rows that may be padded, and the pool they draw from. Tool-choice next_action rows (tau-style agents) only:
    a tool of another domain is a real tool that is simply not the right next step here. Mind2Web elements, any row
    with media (GUI screens, video) and edited_file files are left alone, because an element or file from another row
    could exist in this state or would break the question's premise ("elements on the current page", "files the
    agent has looked at")."""
    q = row.get("question") or {}
    src = str(row.get("source", ""))
    if row.get("media"):
        return None
    if row.get("task") not in PAD_FAMILIES or q.get("type") != "choice" or src.startswith("retention") or "mind2web" in src:
        return None
    return f"{row['task']}:tools"


def build_option_pool(rows: Sequence[dict]) -> dict[str, list[tuple[str, Any]]]:
    """Every distinct option (name, description) of the paddable rows, per pad_key, for option padding."""
    pool: dict[str, list[tuple[str, Any]]] = {}
    seen: dict[str, set] = {}
    for r in rows:
        key = pad_key(r)
        if key is None:
            continue
        names = seen.setdefault(key, set())
        for name, desc in r["question"]["criteria"].items():
            if name not in names:
                names.add(name)
                pool.setdefault(key, []).append((name, desc))
    return pool


def pad_options(row: dict, pool: dict, prob: float, seed: int, char_budget: int = 24000,
                state_char_cap: int = 30000) -> dict:
    """Option-count augmentation. With probability `prob`, a tool-choice row (pad_key) gets extra wrong options:
    tools of other agent domains, up to a random total between 27 and 255 (in practice the pool holds 95 tools). The gold option and the target are unchanged. Rows with long states are left alone so the
    padded prompt stays under max_len. Uses its own random stream, so unpadded rows render exactly as before."""
    q = row["question"]
    key = pad_key(row)
    cands = pool.get(key) if key else None
    if not cands:
        return q
    rng = random.Random(f"pad-{seed}")
    if rng.random() >= prob:
        return q
    state = row.get("state")
    state_chars = len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))
    if state_chars > state_char_cap:
        return q
    crit = dict(q["criteria"])
    k0 = len(crit)
    target_k = rng.randint(max(k0 + 1, 27), PAD_MAX_OPTIONS)
    budget = char_budget
    for _ in range(4 * target_k):
        if len(crit) >= target_k:
            break
        name, desc = cands[rng.randrange(len(cands))]
        if name in crit:
            continue
        cost = len(name) + len(render_entry_text(desc)) + 8
        if cost > budget:
            break
        crit[name] = desc
        budget -= cost
    if len(crit) <= k0:
        return q
    return {**q, "criteria": crit}


def render_entry_text(desc: Any) -> str:
    return "" if desc is None else (desc if isinstance(desc, str) else json.dumps(desc, ensure_ascii=False))


def render_example(row: dict, tmpl, idx: int, seed: int | None = None, augment: bool = False,
                   option_pool: dict | None = None, pad_prob: float = 0.0) -> Example:
    """Render one row. `seed` drives the option permutation, the state serialization choice and (with an
    option pool and pad_prob > 0) option padding; with seed None the frozen production rendering is used
    (no permutation, no padding, indent 2)."""
    qdict = row["question"]
    if seed is not None and augment and option_pool and pad_prob > 0:
        qdict = pad_options(row, option_pool, pad_prob, seed)
    question = build_question(qdict)
    base = render_question("q", question)
    k = len(base.labels)
    if seed is None:
        rendered = base
        compact = False
    else:
        rng = random.Random(seed)
        order = list(range(k))
        rng.shuffle(order)
        rendered = render_question("q", question, order=order)
        compact = augment and rng.random() < 0.5
    state = serialize_state(row.get("state"), compact)
    prompt = tmpl(state, rendered.suffix)
    return Example(
        idx=idx,
        prompt=prompt,
        kind=rendered.kind,
        labels=list(rendered.labels),
        target=target_vector(row["target"], rendered.labels),
        source=str(row.get("source", "unknown")),
        task=str(row.get("task", "unknown")),
    )


class _BlockImport:
    """Meta path finder that makes a package look uninstalled."""

    def __init__(self, root: str) -> None:
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.root or fullname.startswith(self.root + "."):
            raise ImportError(f"{self.root} is disabled by train.common.block_fla_kernels")
        return None


def block_fla_kernels() -> None:
    """Force transformers onto its reference PyTorch gated delta rule instead of the fla kernel.

    transformers resolves `chunk_gated_delta_rule` by importing `fla` at model-import time and
    falls back to `torch_chunk_gated_delta_rule` when that import fails. fla 0.5.2 refuses to run
    its gated backward kernel on H100/H200 with Triton in [3.4.0, 3.7.1) because the kernel produces
    incorrect gradients there, so on such a box the fla path cannot be used for training at all.
    The torch path is the correct reference implementation and is more than an order of magnitude
    slower, so this is opt-in and never silent. Must be called before any model is loaded.
    """
    import sys

    for name in [m for m in sys.modules if m == "fla" or m.startswith("fla.")]:
        del sys.modules[name]
    try:
        import transformers.utils.import_utils as import_utils

        import_utils.is_flash_linear_attention_available = lambda *a, **k: False
    except ImportError:
        pass
    if not any(isinstance(f, _BlockImport) and f.root == "fla" for f in sys.meta_path):
        sys.meta_path.insert(0, _BlockImport("fla"))
    log.warning("fla kernels disabled: gated delta rule layers will use the reference PyTorch "
                "implementation, which is correct but more than an order of magnitude slower")


@dataclass
class Bundle:
    model: Any
    tokenizer: Any
    config: Any
    model_type: str
    slot_ids: list[int]
    template_kwargs: dict
    processor: Any = None
    counter: Any = None

    @property
    def multimodal(self) -> bool:
        return self.processor is not None

    def render_prompt(self, state: Any, suffix: str) -> str:
        return self.tokenizer.apply_chat_template(
            build_messages(state, suffix), tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )

    def template_for(self, row: dict):
        """The prompt renderer for one row: the text template, or for a multimodal bundle the template that turns
        the row's media placeholders into vision blocks (identical text for rows without media)."""
        if not self.multimodal:
            return self.render_prompt
        from train import mm

        media = row.get("media") or []
        return lambda state, suffix: mm.render_prompt_mm(self.processor, self.template_kwargs, build_messages,
                                                         state, suffix, media)

    def prompt_length(self, prompt: str, row: dict) -> int:
        if not self.multimodal:
            return len(self.encode(prompt))
        return self.counter.length(prompt, row.get("media") or [])

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


def verify_slots(tokenizer, model_path: str) -> list[int]:
    """Answer slots are the standalone label tokens (A..Z, then two-letter labels up to 256 slots), as qev.engine
    verifies them. Training needs all of them."""
    ids = []
    for letter in SLOT_LABELS:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1 or tokenizer.decode(enc) != letter:
            raise ValueError(f"answer slot {letter!r} is not a single round-trip token in {model_path}")
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise ValueError("answer slot tokens collide")
    return ids


def load_bundle(model_path: str, device: str = "cuda:0", dtype: str = "bfloat16", train: bool = False,
                multimodal: bool = False, chat_template: str | None = None, load_model: bool = True) -> Bundle:
    """Load a causal LM the same way qev.engine does, including the Qwen3.5 special case. With multimodal=True the
    vision model (Qwen3_5ForConditionalGeneration) and its processor are loaded instead of the text-only LM.
    `chat_template` (a .jinja path) is only for base checkpoints that ship without one."""
    import torch
    import transformers

    local = Path(model_path).exists()
    common = {"local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(model_path, **common)
    processor = counter = None
    if multimodal:
        from train import mm

        processor = transformers.AutoProcessor.from_pretrained(model_path, **common)
        if chat_template:
            processor.chat_template = Path(chat_template).read_text()
        tokenizer = processor.tokenizer
        if tokenizer.chat_template is None:
            tokenizer.chat_template = processor.chat_template
        cls = transformers.Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else \
            transformers.AutoModelForImageTextToText
        counter = mm.TokenCounter(processor)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, **common)
        if chat_template:
            tokenizer.chat_template = Path(chat_template).read_text()
        cls = transformers.AutoModelForCausalLM
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            cls = transformers.Qwen3_5ForCausalLM
            config = config.get_text_config()
    model = None
    if load_model:
        model = cls.from_pretrained(model_path, config=config, dtype=getattr(torch, dtype),
                                    device_map={"": device}, low_cpu_mem_usage=True, **common)
        model.train() if train else model.eval()
    model_type = config.model_type
    template_kwargs = {"enable_thinking": False} if "qwen" in model_type else {}
    slot_ids = verify_slots(tokenizer, model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return Bundle(model=model, tokenizer=tokenizer, config=config, model_type=model_type,
                  slot_ids=slot_ids, template_kwargs=template_kwargs, processor=processor, counter=counter)


def linear_projection_modules(model) -> list[str]:
    """Every linear projection leaf, by full module path. Routing gates and the LM head stay frozen.

    Full paths rather than short names on purpose: peft matches a short name against every module
    whose path ends with it, and some models (gemma-4-E4B-it) give the same suffix to a wrapper
    that is not an `nn.Linear` at all, which peft then refuses to adapt. Matching the exact paths
    of the modules we checked avoids that.
    """
    import torch

    names: list[str] = []
    for name, module in model.named_modules():
        if type(module) is not torch.nn.Linear:
            continue
        if ".visual." in f".{name}." or name.startswith("visual"):
            continue
        short = name.split(".")[-1]
        if short in {"lm_head", "router", "score", "classifier"}:
            continue
        if "proj" not in short:
            continue
        names.append(name)
    return names


def short_names(paths: Sequence[str]) -> list[str]:
    return sorted({p.split(".")[-1] for p in paths})


_TENSOR_KEEP = True


def readout_logits(model, input_ids, attention_mask, ends):
    """Full-vocabulary logits at each row's last real token, under RIGHT padding.

    Right padding is what the readout wants: under causal attention a padded tail cannot reach
    the readout position, so row i reads out at position ends[i] = len_i - 1 and the default
    position_ids (arange) are already correct for every real token. Left padding would shift the
    positions of the real tokens, which is why it is not used here.

    We never materialise [B, T, V]. The primary path passes `logits_to_keep` as a tensor of the
    distinct readout positions, exactly like qev.engine._run_batched. If a model rejects the
    tensor form, we fall back to an integer tail width, which every implementation supports and
    which is just as cheap because batches are length-sorted.
    """
    import torch

    global _TENSOR_KEEP
    bsz, seqlen = input_ids.shape
    ends = [int(e) for e in ends]
    rows = torch.arange(bsz, device=input_ids.device)
    if _TENSOR_KEEP:
        keep = sorted(set(ends))
        try:
            ltk = torch.tensor(keep, dtype=torch.long, device=input_ids.device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                        return_dict=True, logits_to_keep=ltk)
            pos = torch.tensor([keep.index(e) for e in ends], dtype=torch.long, device=input_ids.device)
            return out.logits[rows, pos]
        except Exception as exc:
            _TENSOR_KEEP = False
            log.warning("tensor logits_to_keep unavailable (%s); using integer tail width", exc)
    width = seqlen - min(ends)
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                return_dict=True, logits_to_keep=width)
    pos = torch.tensor([e - (seqlen - width) for e in ends], dtype=torch.long, device=input_ids.device)
    return out.logits[rows, pos]


def slot_view(vocab_logits, slot_ids, ks):
    """Gather the first K slot logits per row into a padded [B, Kmax] tensor plus a mask."""
    import torch

    bsz = vocab_logits.shape[0]
    kmax = max(ks)
    idx = torch.tensor(slot_ids[:kmax], dtype=torch.long, device=vocab_logits.device)
    sel = vocab_logits.index_select(1, idx).float()
    mask = torch.zeros((bsz, kmax), dtype=torch.bool, device=vocab_logits.device)
    for i, k in enumerate(ks):
        mask[i, :k] = True
    sel = sel.masked_fill(~mask, float("-inf"))
    return sel, mask


def pad_batch(id_lists: Sequence[Sequence[int]], pad_id: int, device):
    """Right padding. Returns input_ids, attention_mask and the readout index per row."""
    import torch

    width = max(len(x) for x in id_lists)
    bsz = len(id_lists)
    input_ids = torch.full((bsz, width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, width), dtype=torch.long, device=device)
    ends = []
    for i, ids in enumerate(id_lists):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, : len(ids)] = 1
        ends.append(len(ids) - 1)
    return input_ids, attention_mask, ends


def length_bucket_batches(lengths: Sequence[int], batch_tokens: int, max_rows: int = 256,
                          shuffle_seed: int | None = None) -> list[list[int]]:
    """Sort by length, then cut batches so that rows * padded width stays under batch_tokens.

    Sorting keeps padding waste small; the batch order is shuffled so the optimizer does not see
    all the short examples first.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        width = max(cur_max, lengths[i])
        if cur and ((len(cur) + 1) * width > batch_tokens or len(cur) + 1 > max_rows):
            batches.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur.append(i)
            cur_max = width
    if cur:
        batches.append(cur)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(batches)
    return batches


def chunked(items: Iterable, size: int):
    buf = []
    for it in items:
        buf.append(it)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


CONFIG_ALIASES = {"train": "train_file"}


def load_yaml_config(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    out = {}
    for k, v in data.items():
        key = str(k).replace("-", "_")
        out[CONFIG_ALIASES.get(key, key)] = v
    return out


def apply_config(parser, path: str | Path) -> None:
    """Load a YAML config as argparse defaults. Unknown keys are an error, not a silent no-op."""
    cfg = load_yaml_config(path)
    known = {a.dest for a in parser._actions}
    unknown = sorted(set(cfg) - known)
    if unknown:
        parser.error(f"unknown keys in {path}: {', '.join(unknown)}")
    parser.set_defaults(**cfg)


def progress(done: int, total: int, prefix: str = "", width: int = 30) -> str:
    frac = 0.0 if total <= 0 else done / total
    filled = int(round(frac * width))
    return f"{prefix}[{'#' * filled}{'.' * (width - filled)}] {done}/{total} ({100 * frac:.1f}%)"


def fmt(x: float) -> str:
    return "nan" if x is None or math.isnan(x) else f"{x:.4f}"


@dataclass
class Scored:
    kind: str
    k: int
    logits: list[float]
    target: list[float]
    source: str
    task: str
    labels: list[str]
    idx: int = -1


def collect_slot_logits(bundle: Bundle, rows: Sequence[dict], max_len: int = 4096,
                        batch_tokens: int = 16384, max_rows: int = 64, device: str = "cuda:0",
                        report_every: int = 10, quiet: bool = False, sync_batches=None) -> list[Scored]:
    """Run the frozen production rendering (no permutation, indent 2) over every row and keep the
    slot logits at the readout position. No gradients, no sampling.

    `sync_batches` is for a sharded (FSDP) model, where every forward is collective: it maps this rank's
    batch count to the count all ranks must run, and the rank pads with one-token forwards. It also swaps
    inference_mode for no_grad, since FSDP keeps the gathered weights' storage for later training steps."""
    import torch

    texts, examples = [], []
    for i, row in enumerate(rows):
        ex = render_example(row, bundle.render_prompt, i, seed=None, augment=False)
        examples.append(ex)
        texts.append(ex.prompt)
    lengths = []
    for chunk in chunked(texts, 512):
        enc = bundle.tokenizer(chunk, add_special_tokens=False)["input_ids"]
        lengths.extend(len(x) for x in enc)
    usable = [i for i in range(len(rows)) if lengths[i] <= max_len]
    if len(usable) < len(rows) and not quiet:
        log.warning("%d of %d rows exceed max-len %d and are skipped", len(rows) - len(usable), len(rows), max_len)
    batches = length_bucket_batches([lengths[i] for i in usable], batch_tokens, max_rows, shuffle_seed=None)
    batches = [[usable[j] for j in b] for b in batches]

    out: list[Scored] = []
    dev = torch.device(device)
    pad_id = bundle.tokenizer.pad_token_id
    done = 0
    last_report = -1
    n_total = sync_batches(len(batches)) if sync_batches is not None else len(batches)
    grad_off = torch.no_grad() if sync_batches is not None else torch.inference_mode()
    with grad_off:
        for _ in range(n_total - len(batches)):
            ids = torch.tensor([[pad_id]], dtype=torch.long, device=dev)
            readout_logits(bundle.model, ids, torch.ones_like(ids), [0])
        for bi, batch in enumerate(batches):
            exs = [examples[i] for i in batch]
            id_lists = [bundle.encode(ex.prompt) for ex in exs]
            input_ids, attention_mask, ends = pad_batch(id_lists, pad_id, dev)
            ks = [len(ex.labels) for ex in exs]
            vocab_logits = readout_logits(bundle.model, input_ids, attention_mask, ends)
            sel, _ = slot_view(vocab_logits, bundle.slot_ids, ks)
            sel = sel.float().cpu().tolist()
            for r, ex in enumerate(exs):
                out.append(Scored(kind=ex.kind, k=len(ex.labels), logits=sel[r][: len(ex.labels)],
                                  target=list(ex.target), source=ex.source, task=ex.task, labels=list(ex.labels),
                                  idx=ex.idx))
            done += len(exs)
            pct = int(100 * done / max(len(usable), 1)) // 10
            if pct != last_report and not quiet:
                last_report = pct
                print(progress(done, len(usable), prefix="score "), flush=True)
    return out


def metrics(scored: Sequence["Scored"], temp_for) -> dict:
    """Accuracy, NLL, Brier and ECE for a group of scored rows under a temperature function."""
    from qev.calibrate import brier as brier_fn
    from qev.calibrate import ece as ece_fn
    from qev.calibrate import nll as nll_fn
    from qev.calibrate import softmax as softmax_fn

    if not scored:
        return {"n": 0}
    accs, nlls, briers, confs = [], [], [], []
    for s in scored:
        p = softmax_fn(s.logits, temp_for(s.kind, s.k))
        gold = max(range(len(s.target)), key=s.target.__getitem__)
        pred = max(range(len(p)), key=p.__getitem__)
        accs.append(1.0 if pred == gold else 0.0)
        nlls.append(nll_fn(p, s.target))
        briers.append(brier_fn(p, s.target))
        confs.append(max(p))
    return {
        "n": len(scored),
        "acc": sum(accs) / len(accs),
        "nll": sum(nlls) / len(nlls),
        "brier": sum(briers) / len(briers),
        "ece": ece_fn(confs, accs),
    }


def row_to_question(row: dict) -> dict:
    prim = row.get("primitive") or ("choice" if len(row["options"]) != 2 or {o["id"] for o in row["options"]} != {"true", "false"} else "noul")
    if prim == "noul":
        desc = {o["id"]: o["description"] for o in row["options"]}
        return {"type": "noul", "instructions": row["question"], "criteria": {"true": desc.get("true"), "false": desc.get("false")}}
    if prim == "score":
        levels = [o["description"] for o in sorted(row["options"], key=lambda o: int(o["id"]))]
        levels = [l.split(": ", 1)[1] if l.lower().startswith("level ") and ": " in l else l for l in levels]
        return {"type": "score", "instructions": row["question"], "criteria": levels}
    return {"type": "choice", "instructions": row["question"], "criteria": {o["id"]: o["description"] for o in row["options"]}}
