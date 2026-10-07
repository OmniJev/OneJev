"""OneJev on llama.cpp: the System One engine over a GGUF model served by llama-server.

Prompts, answer slots, calibration and answer assembly are those of `qev.engine`. llama-server computes the next-token
distribution at the end of each branch; the answer is read off the slot tokens of that distribution, as in the
PyTorch engine. Images are resized here the way the model's processor resizes them, so llama.cpp sees the same pixels.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import media as mmedia
from .calibrate import Calibration
from .engine import BranchResult, DecisionEngine, EngineInfo
from .prompt import DEFAULT_STYLE, PROMPT_VERSION, PromptStyle, RenderedQuestion, build_messages
from .schema import SystemOneRequest, SystemOneResponse

log = logging.getLogger("qev.llamacpp")

VISION_BLOCK = re.compile(r"<\|vision_start\|>(<\|image_pad\|>|<\|video_pad\|>)<\|vision_end\|>")
SIZES = ("0.8B", "4B", "9B", "27B")


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def tokenizer_repo(gguf: str) -> str | None:
    """The OneJev repository a GGUF file or repository was made from, guessed from its name."""
    m = re.search(r"OneJev-(0\.8B|4B|9B|27B)", gguf, re.IGNORECASE)
    if not m:
        return None
    size = next(s for s in SIZES if s.lower() == m.group(1).lower())
    return f"OmniJev/OneJev-{size}"


def resolve_gguf(spec: str, default_quant: str = "Q8_0") -> tuple[str, str | None]:
    """A local .gguf path, or `repo[:quant]` on the Hub, as local paths of the model and its vision projector."""
    if Path(spec).exists():
        found = sorted(Path(spec).parent.glob("*mmproj*.gguf"))
        mmproj = next((p for p in found if "f16" in p.name.lower()), found[0] if found else None)
        return spec, str(mmproj) if mmproj else None
    from huggingface_hub import HfApi, hf_hub_download

    repo, _, quant = spec.partition(":")
    quant = (quant or default_quant).lower()
    files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf")]
    models = [f for f in files if "mmproj" not in f.lower() and re.search(rf"[.\-_]{re.escape(quant)}\.gguf$", f.lower())]
    if not models:
        quants = sorted({f.rsplit(".", 2)[-2] for f in files if "mmproj" not in f.lower()})
        raise FileNotFoundError(f"no {quant} file in {repo}; available: {', '.join(quants)}")
    projs = sorted(f for f in files if "mmproj" in f.lower())
    proj = next((f for f in projs if "f16" in f.lower()), projs[0] if projs else None)
    return hf_hub_download(repo, models[0]), hf_hub_download(repo, proj) if proj else None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    """A llama-server child process for one GGUF model (a local file or a Hub repo:quant)."""

    def __init__(self, gguf: str, binary: str = "llama-server", mmproj: str | None = None, ctx: int = 32768,
                 parallel: int = 1, extra: list[str] | None = None) -> None:
        exe = shutil.which(binary) or (binary if Path(binary).exists() else None)
        if exe is None:
            raise FileNotFoundError(f"{binary} not found; install llama.cpp (macOS: brew install llama.cpp) or pass "
                                    f"--llama-server /path/to/llama-server")
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [exe, "--host", "127.0.0.1", "--port", str(self.port), "-c", str(ctx * parallel), "-np", str(parallel),
               "--image-min-tokens", "64", "--image-max-tokens", "16384"]
        model, found = resolve_gguf(gguf)
        cmd += ["-m", model]
        if mmproj or found:
            cmd += ["--mmproj", mmproj or found]
        cmd += list(extra or [])
        log.info("starting %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self._tail: list[str] = []
        threading.Thread(target=self._drain, daemon=True).start()
        self._wait()

    def _drain(self) -> None:
        for line in self.proc.stderr:
            self._tail = (self._tail + [line.rstrip()])[-40:]

    def _wait(self, timeout: float = 3600) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited:\n" + "\n".join(self._tail))
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError("llama-server did not become ready")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class LlamaCppEngine(DecisionEngine):
    def __init__(self, url: str, tokenizer: str, calibration: Calibration | None = None, name: str = "gguf",
                 max_branch_tokens: int = 32768, max_request_tokens: int = 65536, style: PromptStyle = DEFAULT_STYLE,
                 timeout: float = 900, image_config: dict | None = None) -> None:
        import transformers

        self.url = url.rstrip("/")
        self.timeout = timeout
        self.calibration = calibration or Calibration()
        self.max_branch_tokens = max_branch_tokens
        self.max_request_tokens = max_request_tokens
        self.style = style
        self.model_path = name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.model_type = "qwen3_5"
        self.template_kwargs = {"enable_thinking": False}
        self.slot_ids = self._verify_slots()
        cfg = image_config or {}
        self.factor = int(cfg.get("patch_size", 16)) * int(cfg.get("merge_size", 2))
        size = cfg.get("size", {})
        self.min_pixels = int(size.get("shortest_edge", 65536))
        self.max_pixels = int(size.get("longest_edge", 16777216))
        self._media: list[dict] = []
        self._images: list[str] = []
        self._lock = threading.Lock()
        props = self._get("/props")
        self.vision = bool((props.get("modalities") or {}).get("vision"))
        self.media_marker = props.get("media_marker") or "<__media__>"
        self._check_tokenizer()
        self.info = EngineInfo(model_path=name, model_type="gguf", dtype="gguf", device=self.url,
                               fork_mode="llama.cpp", fork_max_abs_diff=None, prompt_version=PROMPT_VERSION,
                               transformers_version=transformers.__version__,
                               extra={"vision": self.vision, "llama_build": props.get("build_info", "")})
        log.info("llama.cpp engine on %s (%s), vision=%s, tokenizer %s", self.url, name, self.vision, tokenizer)

    def _post(self, path: str, body: dict) -> Any:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"llama-server {path} returned {exc.code}: {exc.read()[:500]!r}") from None

    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(self.url + path, timeout=30) as r:
            return json.loads(r.read())

    def _check_tokenizer(self) -> None:
        """The GGUF vocabulary must tokenize a rendered prompt exactly as the model's own tokenizer does."""
        text = self._render_prompt({"screen": "Pay the invoice.", "total": "$1,284.50"},
                                   "Question: Is it paid?\n\nOptions:\nA. yes\nB. no\n\nAnswer with one letter: A, B.")
        ours = self._encode(text)
        theirs = self._post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
        if ours != theirs:
            raise ValueError("the GGUF vocabulary does not match the OneJev tokenizer; pass the matching --tokenizer")

    def _render_prompt(self, state: Any, suffix: str) -> str:
        return mmedia.render_prompt(self.tokenizer, self.template_kwargs, build_messages(state, suffix, self.style),
                                    self._media)

    def _png(self, image) -> str:
        h, w = smart_resize(image.height, image.width, self.factor, self.min_pixels, self.max_pixels)
        if (w, h) != image.size:
            from PIL import Image

            image = image.resize((w, h), Image.BICUBIC)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _set_media(self, items: list[dict], root: str | None = None, allow_paths: bool = False) -> None:
        if any(m["type"] == "video" for m in items):
            raise mmedia.MediaError("video is not supported by the llama.cpp backend; use the PyTorch server")
        if items and not self.vision:
            raise mmedia.MediaError("this llama-server has no vision projector (start it with --mmproj)")
        try:
            self._images = [self._png(mmedia.open_image(m, root, allow_paths)) for m in items]
        except mmedia.MediaError:
            raise
        except Exception as exc:
            raise mmedia.MediaError(f"could not load media: {type(exc).__name__}: {exc}") from exc
        self._media = items

    def _prompts(self, state: Any, rendered: list[RenderedQuestion]) -> tuple[Any, list[Any]]:
        """The shared state prefix and one full prompt per branch."""
        if not self._media:
            prefix_ids, suffixes = self._prefix_and_suffix_ids(state, rendered)
            return prefix_ids, [prefix_ids + s for s in suffixes]
        text = VISION_BLOCK.sub(self.media_marker, self._render_prompt(state, "QEV_PREFIX_MARK"))
        head, tail = text.split("QEV_PREFIX_MARK")
        return ({"prompt_string": head, "multimodal_data": self._images},
                [{"prompt_string": head + r.suffix + tail, "multimodal_data": self._images} for r in rendered])

    def _complete(self, prompts: list[Any], extra: dict) -> list[list[dict]]:
        body = {"prompt": prompts, "n_predict": 1, "cache_prompt": True, "temperature": 0.0, **extra}
        out = self._post("/completion", body)
        out = out if isinstance(out, list) else [out]
        out = sorted(out, key=lambda r: r.get("index", 0))
        return [r["completion_probabilities"][0] for r in out]

    def _prefill(self, prefix: Any) -> dict:
        """Compute the shared state alone, so its slot keeps a checkpoint at the end of the state; every branch then
        runs in that slot and computes only its own suffix."""
        out = self._post("/completion", {"prompt": [prefix], "n_predict": 0, "cache_prompt": True})
        slot = (out[0] if isinstance(out, list) else out).get("id_slot")
        return {} if slot is None else {"id_slot": slot}

    def _run_sequential(self, state: Any, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        prefix, prompts = self._prompts(state, rendered)
        pin = self._prefill(prefix)
        n_top = max(64, 2 * max(r.n_slots for r in rendered))
        tops = self._complete(prompts, {"n_probs": n_top, **pin})
        results: list[BranchResult | None] = [None] * len(rendered)
        missing = []
        for i, (r, top) in enumerate(zip(rendered, tops)):
            lp = {t["id"]: t["logprob"] for t in top["top_logprobs"]}
            slots = self.slot_ids[: r.n_slots]
            if all(s in lp for s in slots):
                logits = [lp[s] for s in slots]
                mass = sum(math.exp(x) for x in logits)
                results[i] = BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, mass, 0)
            else:
                missing.append(i)
        for i in missing:
            r = rendered[i]
            slots = self.slot_ids[: r.n_slots]
            top = self._complete([prompts[i]], {"n_probs": r.n_slots, "post_sampling_probs": True,
                                                "samplers": ["temperature"], "temperature": 1.0,
                                                "logit_bias": [[s, 100.0] for s in slots], **pin})[0]
            p = {t["id"]: t["prob"] for t in top["top_probs"]}
            logits = [math.log(max(p.get(s, 0.0), 1e-30)) for s in slots]
            lp = {t["id"]: t["logprob"] for t in tops[i]["top_logprobs"]}
            mass = sum(math.exp(lp[s]) for s in slots if s in lp)
            results[i] = BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, mass, 0)
        return results

    def _run_batched(self, state: Any, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        return self._run_sequential(state, rendered)

    def decide(self, request: SystemOneRequest, debias: int = 1, debug: bool = False,
               media: list | None = None) -> tuple[SystemOneResponse, dict[str, Any]]:
        items = mmedia.check_items(media if media is not None else getattr(request, "media", None))
        with self._lock:
            t0 = time.perf_counter()
            self._set_media(items)
            t_media = time.perf_counter() - t0
            try:
                response, meta = super().decide(request, debias=debias, debug=debug)
            finally:
                self._media, self._images = [], []
        meta["media"] = {"images": len(items), "videos": 0, "load_s": t_media}
        meta["timing"]["total_s"] += t_media
        meta["backend"] = "llama.cpp"
        return response, meta
