"""MMDecisionEngine: the DecisionEngine for vision checkpoints (OneJev). A request may carry images and videos
(qev/media.py); they sit in the state, so they are part of the shared prefix: the vision encoder runs once per
request, and every question is a text-only branch off the forked cache, exactly as in the text engine.

Qwen3.5 vision models place tokens with multimodal RoPE: a vision block takes (t, h, w) positions and the text after
it continues from the block's largest position, so from the last vision block on, the token at sequence index i has
position i + delta on all three axes. The prefix positions come from the model's own get_rope_index; every suffix
token gets i + delta. Branch probabilities therefore equal those of the full prompt run in one piece (the training
and evaluation path).

The prompt is the frozen text prompt with each `<image:N>` / `<video:N>` replaced by the model's vision block; a
request without media renders exactly like a text request.
"""
from __future__ import annotations

import collections
import copy
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import media as mmedia
from .calibrate import Calibration, softmax
from .engine import BranchResult, DecisionEngine, EngineInfo
from .prompt import DEFAULT_STYLE, PromptStyle, RenderedQuestion, build_messages, render_question
from .schema import SystemOneRequest, SystemOneResponse

log = logging.getLogger("qev.mm_engine")

MARK = "QEV_PREFIX_MARK"


def local_model_dir(name: str) -> str:
    """A local directory, a Hub repository id, or a folder inside a Hub repository (org/repo/folder), as a local
    directory; a folder is fetched on its own."""
    if Path(name).exists():
        return name
    from huggingface_hub import snapshot_download

    parts = name.split("/")
    if len(parts) > 2:
        repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
        return str(Path(snapshot_download(repo, allow_patterns=[f"{sub}/*"])) / sub)
    return snapshot_download(name)


class GraphMemoryLow(RuntimeError):
    """Not enough free GPU memory to capture another graph, even after dropping the cached ones."""


def _host_available() -> int:
    """MemAvailable from /proc/meminfo: on a GPU that shares system memory, reclaimable page cache is free for CUDA
    too, while torch.cuda.mem_get_info counts it as used."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _make_room(evict, reserve_fraction: float = 0.1, reserve_bytes: int = 8 << 30) -> None:
    """Before a capture: drop the least recently used graphs (evict() drops one, False when none are left) until the
    device has the reserve free; raise GraphMemoryLow when it still does not. Each graph keeps its own memory pool, so
    a model with many prompt lengths would otherwise fill the device with graphs."""
    import torch

    def low() -> bool:
        free, total = torch.cuda.mem_get_info()
        if torch.cuda.get_device_properties(torch.cuda.current_device()).is_integrated:
            free = max(free, _host_available())
        return free < max(reserve_bytes, reserve_fraction * total)

    if low():
        torch.cuda.empty_cache()
    while low() and evict():
        torch.cuda.empty_cache()
    if low():
        raise GraphMemoryLow("less than the reserve free on the device")


class DecoderGraphs:
    """CUDA graphs of the text decoder over a whole prompt without a cache, one per padded length (a multiple of
    `step` tokens), captured on first use. The prompt is padded on the right, which leaves every real position as it
    was (attention is causal and the linear-attention recurrence only runs forward), and the last real position is
    read from the padded run. A replay launches the whole decoder at once instead of a few thousand kernels one by one
    from Python, which is most of the time of a single request. Every graph has its own memory pool: in a shared pool
    one graph's capture can take memory another graph still reads on replay, which silently corrupts it. The least
    recently used graphs are dropped beyond `max_graphs`."""

    def __init__(self, decoder, step: int = 128, max_len: int = 16384, max_graphs: int = 8) -> None:
        self.decoder, self.step, self.max_len, self.max_graphs = decoder, step, max_len, max_graphs
        self.graphs: collections.OrderedDict[int, tuple] = collections.OrderedDict()

    def __call__(self, embeds, positions):
        """embeds (1, L, hidden), positions (3, 1, L) -> final-norm hidden state at position L - 1, or None when L is
        over the largest graph."""
        import torch

        n = embeds.shape[1]
        size = -(-n // self.step) * self.step
        if size > self.max_len:
            return None
        if size in self.graphs:
            self.graphs.move_to_end(size)
        emb, pos, out, graph = self.graphs.get(size) or self._capture(size, embeds, positions)
        emb[:, :n].copy_(embeds)
        emb[:, n:].zero_()
        pos[..., :n].copy_(positions)
        pos[..., n:].copy_(positions[..., -1:] + torch.arange(1, size - n + 1, device=pos.device, dtype=pos.dtype))
        graph.replay()
        return out[:, n - 1]

    def _evict(self) -> bool:
        if not self.graphs:
            return False
        self.graphs.popitem(last=False)
        return True

    def clear(self) -> None:
        self.graphs.clear()

    def _capture(self, size: int, embeds, positions):
        import torch

        emb = torch.zeros((1, size, embeds.shape[-1]), dtype=embeds.dtype, device=embeds.device)
        pos = torch.arange(size, device=embeds.device, dtype=positions.dtype).view(1, 1, -1).repeat(3, 1, 1)

        def run():
            return self.decoder(inputs_embeds=emb, position_ids=pos, use_cache=False, return_dict=True).last_hidden_state

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)
        while len(self.graphs) >= self.max_graphs:
            self.graphs.popitem(last=False)
        _make_room(self._evict)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        self.graphs[size] = (emb, pos, out, graph)
        log.info("captured a decoder graph for %d tokens", size)
        return self.graphs[size]


def _bucket(n: int, step: int) -> int:
    return -(-n // step) * step


class ForkGraphs:
    """CUDA graphs for a request with several branches: one graph runs the shared prefix and keeps its cache, one
    runs the batched branches off a copy of that cache per row. Graphs are keyed by padded sizes and captured on first
    use. The prefix is padded on the left with zero embeddings: in every layer a zero input stays zero (no biases) and a
    zero linear-attention state stays zero, so the recurrent states after the real tokens are exactly those of the
    unpadded prefix, and a mask keeps the real tokens from attending to the padding. Branch rows are padded on the
    right, as in the eager path, and there is one graph per number of branches. A captured function may only read
    tensors that live as long as its graph (the static inputs stored with it, the cache, the weights) or that it makes
    itself: a temporary made outside the capture is freed afterwards and the replay would read whatever takes its
    memory next. Each graph has its own memory pool (see DecoderGraphs); the least recently used prefix sizes are
    dropped with their branch graphs beyond `max_graphs`."""

    def __init__(self, decoder, fork, pad_id: int, step: int = 128, max_len: int = 16384, width_step: int = 16,
                 max_width: int = 512, max_rows: int = 32, max_graphs: int = 8) -> None:
        self.decoder, self.fork, self.pad_id = decoder, fork, pad_id
        self.step, self.max_len, self.width_step, self.max_width, self.max_rows = step, max_len, width_step, max_width, max_rows
        self.max_graphs = max_graphs
        self.prefills: collections.OrderedDict[int, tuple] = collections.OrderedDict()
        self.branches: collections.OrderedDict[tuple[int, int, int], tuple] = collections.OrderedDict()

    def __call__(self, embeds, positions, suffixes: list[list[int]], delta: int):
        """embeds (1, P, hidden) and positions (3, 1, P) of the prefix; token ids of each branch. Returns the
        final-norm hidden state at each branch's last token (rows x hidden), or None when a size is over the limits."""
        import torch

        n, P = len(suffixes), embeds.shape[1]
        width = max(len(s) for s in suffixes)
        size, rows, wide = _bucket(P, self.step), n, _bucket(width, self.width_step)
        if size > self.max_len or rows > self.max_rows or wide > self.max_width:
            return None
        if size in self.prefills:
            self.prefills.move_to_end(size)
        emb, pos, pad, cache, graph = self.prefills.get(size) or self._capture_prefill(size, embeds, positions)
        a = size - P
        emb[:, :a].zero_()
        emb[:, a:].copy_(embeds)
        pos[..., :a].zero_()
        pos[..., a:].copy_(positions)
        pad.fill_(a)
        graph.replay()
        key = (size, rows, wide)
        if key in self.branches:
            self.branches.move_to_end(key)
        ids, base, pad_b, out, bgraph = self.branches.get(key) or self._capture_branch(key, cache, positions.dtype)
        host = torch.full((rows, wide), self.pad_id, dtype=torch.long)
        for r in range(rows):
            s = suffixes[r % n]
            host[r, :len(s)] = torch.tensor(s, dtype=torch.long)
        ids.copy_(host, non_blocking=True)
        base.fill_(P + delta)
        pad_b.fill_(a)
        bgraph.replay()
        ends = torch.tensor([len(s) - 1 for s in suffixes], device=out.device)
        return out[torch.arange(n, device=out.device), ends]

    def _drop_prefix(self, size: int) -> None:
        self.prefills.pop(size, None)
        for key in [k for k in self.branches if k[0] == size]:
            del self.branches[key]

    def _evict(self, keep: int | None = None) -> bool:
        """Drop the oldest branch graph, else the oldest prefix graph with its branches; never the prefix `keep`,
        whose cache the graph being captured reads."""
        for key in self.branches:
            self.branches.pop(key)
            return True
        for size in self.prefills:
            if size != keep:
                self._drop_prefix(size)
                return True
        return False

    def clear(self) -> None:
        self.branches.clear()
        self.prefills.clear()

    def _side_warmup(self, run) -> None:
        import torch

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)

    def _capture_prefill(self, size: int, embeds, positions):
        import torch

        dev = embeds.device
        emb = torch.zeros((1, size, embeds.shape[-1]), dtype=embeds.dtype, device=dev)
        pos = torch.zeros((3, 1, size), dtype=positions.dtype, device=dev)
        pad = torch.zeros((), dtype=torch.long, device=dev)

        def run():
            idx = torch.arange(size, device=dev)
            i, j = idx.view(-1, 1), idx.view(1, -1)
            mask = (j <= i) & ((i < pad) == (j < pad))
            return self.decoder(inputs_embeds=emb, position_ids=pos, use_cache=True, return_dict=True,
                                attention_mask={"full_attention": mask.view(1, 1, size, size),
                                                "linear_attention": None}).past_key_values

        self._side_warmup(run)
        while len(self.prefills) >= self.max_graphs:
            self._drop_prefix(next(iter(self.prefills)))
        _make_room(self._evict)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cache = run()
        self.prefills[size] = (emb, pos, pad, cache, graph)
        log.info("captured a prefix graph for %d tokens", size)
        return self.prefills[size]

    def _capture_branch(self, key, cache, pos_dtype):
        import torch

        size, rows, wide = key
        dev = self.prefills[size][0].device
        ids = torch.full((rows, wide), self.pad_id, dtype=torch.long, device=dev)
        base = torch.zeros((), dtype=pos_dtype, device=dev)
        pad = torch.zeros((), dtype=torch.long, device=dev)

        def run():
            q = torch.arange(wide, device=dev)
            k = torch.arange(size + wide, device=dev)
            pos = (base + q).view(1, 1, -1).expand(3, rows, -1)
            kk, qq = k.view(1, -1), q.view(-1, 1)
            mask = torch.where(kk < size, kk >= pad, kk - size <= qq)
            forked = self.fork(cache, rows)
            return self.decoder(input_ids=ids, position_ids=pos, past_key_values=forked, use_cache=True,
                                return_dict=True,
                                attention_mask={"full_attention": mask.view(1, 1, wide, size + wide).expand(rows, 1, -1, -1),
                                                "linear_attention": None}).last_hidden_state

        self._side_warmup(run)
        while len(self.branches) >= 8 * self.max_graphs:
            self.branches.popitem(last=False)
        _make_room(lambda: self._evict(keep=size))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        self.branches[key] = (ids, base, pad, out, graph)
        log.info("captured a branch graph for %d rows x %d tokens over a %d-token prefix", rows, wide, size)
        return self.branches[key]


@dataclass
class Prepared:
    """One request's shared prefix (processor tensors, M-RoPE positions) and its text-only branch suffixes."""

    key: tuple
    prefix_ids: list[int]
    suffixes: list[list[int]]
    inputs: dict
    positions: Any
    delta: int


class MMDecisionEngine(DecisionEngine):
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
        media_root: str | None = None,
        allow_local_paths: bool = True,
        gpu_preprocess: bool = False,
        cuda_graphs: bool = False,
        quantize: str | None = None,
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
        self.media_root = media_root
        self.allow_local_paths = allow_local_paths
        self.gpu_preprocess = gpu_preprocess
        self._want_mass = False
        self._lock = threading.Lock()
        self._loaded: tuple = ([], [], [])
        self._media: list[dict] = []
        self._prep: Prepared | None = None

        t0 = time.perf_counter()
        model_path = local_model_dir(model_path)
        common = {"local_files_only": True, "trust_remote_code": False}
        config = transformers.AutoConfig.from_pretrained(model_path, **common)
        from .fp8 import is_fp8_checkpoint

        stored_fp8 = is_fp8_checkpoint(config)
        if stored_fp8:
            config.quantization_config = None
            del config.quantization_config
            quantize = "fp8"
        if getattr(config, "vision_config", None) is None:
            raise ValueError(f"{model_path} has no vision tower; use DecisionEngine for text-only checkpoints")
        self.processor = transformers.AutoProcessor.from_pretrained(model_path, **common)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = self.processor.chat_template
        cls = transformers.Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else \
            transformers.AutoModelForImageTextToText
        self.model = cls.from_pretrained(model_path, config=config, dtype=self.torch_dtype,
                                         device_map={"": str(self.device)}, low_cpu_mem_usage=True, **common)
        self.model.eval()
        if quantize:
            self._quantize(quantize, stored_dir=model_path if stored_fp8 else None)
        self.head_dtype = head_dtype
        if head_dtype != dtype:
            self._upcast_head(getattr(torch, head_dtype))
        self.model_type = config.model_type
        self.template_kwargs = {"enable_thinking": False} if "qwen" in self.model_type else {}
        self.slot_ids = self._verify_slots()
        self._init_slot_head()
        if gpu_preprocess and self.device.type == "cuda":
            self._normalize_on_gpu()
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self._graphs = self._fork_graphs = None
        if cuda_graphs and self.device.type == "cuda":
            decoder = self.model.model.language_model
            self._graphs = DecoderGraphs(decoder)
            self._fork_graphs = ForkGraphs(decoder, self._fork, self.pad_id)
        self.vision_ids = {i for i in (getattr(config, "image_token_id", None), getattr(config, "video_token_id", None),
                                       getattr(config, "vision_start_token_id", None),
                                       getattr(config, "vision_end_token_id", None)) if i is not None}
        self.info = EngineInfo(model_path=model_path, model_type=self.model_type, dtype=dtype, device=str(self.device),
                               fork_mode="sequential", fork_max_abs_diff=None,
                               transformers_version=transformers.__version__, torch_version=torch.__version__,
                               extra={"head_dtype": head_dtype, "multimodal": True})
        if fork_mode == "auto":
            self._select_fork_mode()
        else:
            self.info.fork_mode = fork_mode
        log.info("loaded %s (%s, vision) in %.1fs, fork_mode=%s", model_path, self.model_type,
                 time.perf_counter() - t0, self.info.fork_mode)

    def _quantize(self, kind: str, stored_dir: str | None = None) -> None:
        """8-bit floating point for the decoder's linear layers (qev/fp8.py: one scale per weight row, one per token on
        the fly). The vision tower, the embeddings and the output head keep their dtype; so do layers with a side
        under 128 (the small gate projections). From an FP8 checkpoint the stored weights and scales are used."""
        from .fp8 import quantize_linears, stored_reader

        if kind != "fp8":
            raise ValueError(f"unknown quantization {kind!r}")
        n = quantize_linears(self.model, stored_reader(stored_dir) if stored_dir else None)
        log.info("fp8: %d decoder linear layers%s", n, " from the checkpoint" if stored_dir else " quantized on load")

    def _init_slot_head(self) -> None:
        """The output-head rows of the option-letter slots, in the head dtype. An answer needs only these logits; the
        full vocabulary is computed only for the debug diagnostics (allowed-token mass)."""
        import torch

        head = self.model.get_output_embeddings()
        ids = torch.tensor(self.slot_ids, device=head.weight.device)
        dt = getattr(torch, self.head_dtype)
        self._slot_w = head.weight.detach().index_select(0, ids).to(dt)
        bias = getattr(head, "bias", None)
        self._slot_b = None if bias is None else bias.detach().index_select(0, ids).to(dt)

    def _readout_rows(self, hidden, n_slots: list[int]) -> list[tuple[list[float], float]]:
        """Slot logits for each row of `hidden` (rows x hidden size, final-norm output), with one host sync."""
        if self._want_mass:
            vocab = self.model.get_output_embeddings()(hidden)
            return [self._readout(vocab[i], n) for i, n in enumerate(n_slots)]
        sel = hidden.to(self._slot_w.dtype) @ self._slot_w.T
        if self._slot_b is not None:
            sel = sel + self._slot_b
        return [(row[:n], float("nan")) for row, n in zip(sel.tolist(), n_slots)]

    def _embed(self, inputs: dict):
        """Token embeddings with the vision features written into the image and video placeholders, as the model's
        own forward does before the decoder."""
        import torch

        inner = self.model.model
        ids = inputs["input_ids"]
        emb = inner.get_input_embeddings()(ids)
        for pix, grid, get, key, which in (("pixel_values", "image_grid_thw", inner.get_image_features, "image_features", 0),
                                           ("pixel_values_videos", "video_grid_thw", inner.get_video_features,
                                            "video_features", 1)):
            if pix in inputs:
                feats = torch.cat(get(inputs[pix], inputs[grid], return_dict=True).pooler_output, 0).to(emb.device, emb.dtype)
                mask = inner.get_placeholder_mask(ids, inputs_embeds=emb, **{key: feats})[which]
                emb = emb.masked_scatter(mask, feats)
        return emb

    def _graph_call(self, name: str, call):
        """Run a CUDA-graph path. Short of memory: drop every cached graph and run this request eagerly. Any other
        failure (a kernel that cannot be captured, ...): turn that path off for good and run eagerly."""
        import torch

        graphs = getattr(self, name)
        if graphs is None:
            return None
        try:
            return call(graphs)
        except (GraphMemoryLow, torch.OutOfMemoryError) as exc:
            log.warning("CUDA graphs dropped for %s after %s; this request runs eagerly", name, type(exc).__name__)
            for other in (self._graphs, self._fork_graphs):
                if other is not None:
                    other.clear()
            torch.cuda.empty_cache()
            return None
        except Exception as exc:
            log.warning("CUDA graphs off for %s after %s: %s", name, type(exc).__name__, exc)
            setattr(self, name, None)
            return None

    def _processor_kwargs(self) -> dict:
        return mmedia.processor_kwargs(*self._loaded)

    def _normalize_on_gpu(self) -> None:
        """Resize images and video frames on the CPU exactly as in training, then move the resized 8-bit pixels to the
        GPU for rescaling, normalisation and patching, which is where the CPU spent its time (float32 work and a
        float32 host-to-device copy). The GPU resize kernel differs from the CPU one, so the resize stays put."""
        for proc in (getattr(self.processor, "image_processor", None), getattr(self.processor, "video_processor", None)):
            if proc is None or not hasattr(proc, "rescale_and_normalize"):
                continue
            original = proc.rescale_and_normalize

            def on_gpu(images, *args, _original=original, **kwargs):
                return _original(images.to(self.device, non_blocking=True), *args, **kwargs)
            proc.rescale_and_normalize = on_gpu

    def _render_prompt(self, state: Any, suffix: str) -> str:
        return mmedia.render_prompt(self.processor, self.template_kwargs, build_messages(state, suffix, self.style),
                                    self._media)

    def _prepare(self, state: Any, rendered: list[RenderedQuestion]) -> Prepared:
        """Tokenize every branch with its vision blocks unexpanded, take the longest common prefix that ends at or
        before the end of the state block (all media sit inside it), then run the processor once on the first branch
        to expand the vision blocks and build the pixel tensors. The first branch's tokens after the expanded prefix
        must equal its unexpanded suffix, so the split is exact by construction."""
        import torch

        key = (id(state), tuple(r.suffix for r in rendered), id(self._media))
        if self._prep is not None and self._prep.key == key:
            return self._prep
        prompts = [self._render_prompt(state, r.suffix) for r in rendered]
        full = [self._encode(p) for p in prompts]
        prefix_text = self._render_prompt(state, MARK)
        prefix_u = self._encode(prefix_text[:prefix_text.index(MARK)])
        n = len(prefix_u)
        while n > 0 and not all(ids[:n] == prefix_u[:n] for ids in full):
            n -= 1
        suffixes = [ids[n:] for ids in full]
        if any(len(s) == 0 for s in suffixes):
            raise ValueError("a branch has an empty suffix; state block boundary mismatch")
        if any(t in self.vision_ids for s in suffixes for t in s):
            raise ValueError("media placeholders belong in the state, not in a question")

        self.tokenizer.padding_side = "right"
        out = self.processor(text=[prompts[0]], return_tensors="pt", **self._processor_kwargs())
        expanded = out["input_ids"][0].tolist()
        P = n + len(expanded) - len(full[0])
        if expanded[P:] != suffixes[0]:
            raise RuntimeError("expanded prompt does not end with the branch suffix; vision block outside the prefix")
        inputs = {"input_ids": out["input_ids"][:, :P], "attention_mask": out["attention_mask"][:, :P]}
        if "mm_token_type_ids" in out:
            inputs["mm_token_type_ids"] = out["mm_token_type_ids"][:, :P]
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if k in out:
                inputs[k] = out[k]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        grids = {k: inputs.get(k) for k in ("image_grid_thw", "video_grid_thw")}
        if any(v is not None for v in grids.values()):
            positions, deltas = self.model.model.get_rope_index(
                inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"], attention_mask=inputs["attention_mask"],
                **grids)
            delta = int(deltas.view(-1)[0])
        else:
            positions = torch.arange(P, device=self.device).view(1, 1, -1).expand(3, 1, -1)
            delta = 0
        self._prep = Prepared(key=key, prefix_ids=expanded[:P], suffixes=suffixes, inputs=inputs,
                              positions=positions, delta=delta)
        return self._prep

    def _prefix_and_suffix_ids(self, state: Any, rendered: list[RenderedQuestion]) -> tuple[list[int], list[list[int]]]:
        prep = self._prepare(state, rendered)
        return prep.prefix_ids, prep.suffixes

    def _prefill(self, prep: Prepared):
        """The shared prefix through the decoder, without the output head (its logits are never read)."""
        out = self.model.model(**prep.inputs, position_ids=prep.positions, use_cache=True, return_dict=True)
        cache = out.past_key_values
        del out
        return cache

    def _run_single(self, prep: Prepared, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        """One branch: the whole prompt in one forward without a cache (the training evaluation path), one pass where
        a prefill plus a branch would take two."""
        import torch

        r, suf, P, dev = rendered[0], prep.suffixes[0], len(prep.prefix_ids), self.device
        s = torch.tensor([suf], device=dev)
        inputs = dict(prep.inputs)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], s], 1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        if "mm_token_type_ids" in inputs:
            t = inputs["mm_token_type_ids"]
            inputs["mm_token_type_ids"] = torch.cat([t, torch.zeros((1, len(suf)), dtype=t.dtype, device=dev)], 1)
        pos = torch.cat([prep.positions, self._suffix_positions(P, len(suf), 1, prep.delta).to(prep.positions.dtype)], -1)
        last = self._graph_call("_graphs", lambda g: g(self._embed(inputs), pos))
        if last is None:
            last = self.model.model(**inputs, position_ids=pos, use_cache=False, return_dict=True).last_hidden_state[:, -1]
        (logits, mass), = self._readout_rows(last, [r.n_slots])
        return [BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, mass, len(suf))]

    def _suffix_positions(self, P: int, width: int, rows: int, delta: int):
        import torch

        pos = torch.arange(P, P + width, device=self.device) + delta
        return pos.view(1, 1, -1).expand(3, rows, -1)

    def _run_sequential(self, state: Any, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        import torch

        prep = self._prepare(state, rendered)
        P = len(prep.prefix_ids)
        dev = self.device
        with torch.inference_mode():
            if len(rendered) == 1:
                return self._run_single(prep, rendered)
            base = self._prefill(prep)
            results = []
            for r, suf in zip(rendered, prep.suffixes):
                cache = copy.deepcopy(base)
                o = self.model.model(input_ids=torch.tensor([suf], device=dev),
                                     attention_mask=torch.ones((1, P + len(suf)), dtype=torch.long, device=dev),
                                     position_ids=self._suffix_positions(P, len(suf), 1, prep.delta),
                                     past_key_values=cache, use_cache=True, return_dict=True)
                (logits, mass), = self._readout_rows(o.last_hidden_state[:, -1], [r.n_slots])
                results.append(BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, mass, len(suf)))
                del o, cache
            del base
        return results

    def _run_batched(self, state: Any, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        import torch

        prep = self._prepare(state, rendered)
        P = len(prep.prefix_ids)
        chunk = max(1, min(len(rendered), self.fork_token_budget // max(P, 1)))
        results: list[BranchResult | None] = [None] * len(rendered)
        with torch.inference_mode():
            if len(rendered) == 1:
                return self._run_single(prep, rendered)
            last = self._graph_call("_fork_graphs",
                                    lambda g: g(self._embed(prep.inputs), prep.positions, prep.suffixes, prep.delta))
            if last is not None:
                outs = self._readout_rows(last, [r.n_slots for r in rendered])
                return [BranchResult(r.qid, r.kind, r.labels, r.descriptions, lg, m, len(s))
                        for r, s, (lg, m) in zip(rendered, prep.suffixes, outs)]
            base = self._prefill(prep)
            start = 0
            while start < len(rendered):
                idx = list(range(start, min(start + chunk, len(rendered))))
                try:
                    self._run_chunk_mm(base, P, prep.delta, idx, rendered, prep.suffixes, results)
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

    def _run_chunk_mm(self, base, P: int, delta: int, idx: list[int], rendered, suffixes, results) -> None:
        import torch

        dev = self.device
        sufs = [suffixes[i] for i in idx]
        width = max(len(s) for s in sufs)
        ids = torch.full((len(idx), width), self.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((len(idx), P + width), dtype=torch.long, device=dev)
        ends = []
        for row, s in enumerate(sufs):
            ids[row, : len(s)] = torch.tensor(s, device=dev)
            mask[row, : P + len(s)] = 1
            ends.append(len(s) - 1)
        cache = self._fork(base, len(idx))
        try:
            o = self.model.model(input_ids=ids, attention_mask=mask,
                                 position_ids=self._suffix_positions(P, width, len(idx), delta),
                                 past_key_values=cache, use_cache=True, return_dict=True)
            last = o.last_hidden_state[torch.arange(len(idx), device=dev), torch.tensor(ends, device=dev)]
            outs = self._readout_rows(last, [rendered[i].n_slots for i in idx])
            for row, i in enumerate(idx):
                r = rendered[i]
                results[i] = BranchResult(r.qid, r.kind, r.labels, r.descriptions, outs[row][0], outs[row][1],
                                          len(sufs[row]))
            del o
        finally:
            del cache

    def _full_logits(self, state: Any, rendered: list[RenderedQuestion]) -> list[list[float]]:
        """Slot logits of each branch's whole prompt in one forward without a cache (the training evaluation path)."""
        import torch

        out_logits = []
        with torch.inference_mode():
            for r in rendered:
                out = self.processor(text=[self._render_prompt(state, r.suffix)], return_tensors="pt",
                                     **self._processor_kwargs())
                out = {k: v.to(self.device) for k, v in out.items()}
                o = self.model(**out, use_cache=False, return_dict=True, logits_to_keep=1)
                out_logits.append(self._readout(o.logits[0, -1], r.n_slots)[0])
                del o
        return out_logits

    def _set_media(self, items: list[dict]) -> None:
        self._prep = None
        self._media = items
        try:
            self._loaded = mmedia.load(items, self.media_root, self.allow_local_paths) if items else ([], [], [])
        except mmedia.MediaError:
            raise
        except Exception as exc:
            self._media, self._loaded = [], ([], [], [])
            raise mmedia.MediaError(f"could not load media: {type(exc).__name__}: {exc}") from exc

    def decide(self, request: SystemOneRequest, debias: int = 1, debug: bool = False,
               media: list | None = None) -> tuple[SystemOneResponse, dict[str, Any]]:
        """Answer a request whose state may reference `media` (the request's own `media` field, or the argument)."""
        items = mmedia.check_items(media if media is not None else getattr(request, "media", None))
        with self._lock:
            t0 = time.perf_counter()
            self._set_media(items)
            t_media = time.perf_counter() - t0
            self._want_mass = debug
            try:
                response, meta = super().decide(request, debias=debias, debug=debug)
            finally:
                self._want_mass = False
                self._set_media([])
        meta["media"] = {"images": sum(m["type"] == "image" for m in items),
                         "videos": sum(m["type"] == "video" for m in items), "load_s": t_media}
        meta["timing"]["total_s"] += t_media
        return response, meta

    def _select_fork_mode(self) -> None:
        """Compare batched-fork and sequential logits on a small synthetic request with one image."""
        import torch
        from PIL import Image, ImageDraw

        from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

        img = Image.new("RGB", (320, 240), (250, 250, 250))
        d = ImageDraw.Draw(img)
        d.rectangle([20, 20, 140, 100], fill=(200, 40, 40))
        d.ellipse([180, 120, 300, 220], fill=(40, 80, 200))
        state = {"screen": "<image:1>", "task": "Pay the open invoice."}
        qs = {
            "a": NoulQuestion(type="noul", instructions="Is there a red rectangle on the screen?"),
            "b": ChoiceQuestion(type="choice", instructions="Which shape is blue?",
                                criteria={"circle": "a round shape", "rectangle": "a box", "none": "no blue shape"}),
            "c": ScoreQuestion(type="score", instructions="How cluttered is the screen?",
                               criteria=["empty", "a few items", "crowded"]),
        }
        rendered = [render_question(k, q, style=self.style) for k, q in qs.items()]
        self._set_media([{"type": "image", "image": img}])
        try:
            seq = self._run_sequential(state, rendered)
            full = self._full_logits(state, rendered)
            one = self._run_sequential(state, rendered[:1])[0]
            split = max(abs(a - b) for s, f in zip(seq + [one], full + full[:1])
                        for a, b in zip(softmax(s.logits), softmax(f)))
            self.info.extra["split_vs_full_max_abs_diff"] = float(split)
            if split > self.fork_tolerance:
                raise RuntimeError(f"shared-prefix branches differ from the full prompt by {split:.4f} in probability; "
                                   f"the multimodal position handling does not match this transformers version")
            try:
                bat = self._run_batched(state, rendered)
            except Exception as exc:
                log.warning("batched fork unavailable (%s); using sequential branches", exc)
                self.info.fork_mode, self.info.fork_max_abs_diff = "sequential", None
                return
        finally:
            self._set_media([])
        diff = max(abs(a - b) for s, bb in zip(seq, bat) for a, b in zip(softmax(s.logits), softmax(bb.logits)))
        self.info.fork_max_abs_diff = float(diff)
        self.info.fork_mode = "batched" if diff <= self.fork_tolerance else "sequential"
        if self.info.fork_mode == "sequential":
            log.warning("batched fork differs from sequential by %.4f in probability (> %.4f); using sequential branches",
                        diff, self.fork_tolerance)
        else:
            log.info("batched fork verified with an image: max probability difference vs sequential %.5f", diff)
        with torch.no_grad():
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
