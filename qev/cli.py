"""Command line: qev serve | ask | models."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "OmniJev/OneJev-4B"

MODEL_ALIASES = {f"onejev-{s.lower()}": f"OmniJev/OneJev-{s}" for s in ("0.8B", "4B", "9B", "27B", "27B-FP8")}


def _resolve_model(name: str) -> str:
    """Accept a release alias, a local path, an HF id, or a cached HF id (offline)."""
    name = MODEL_ALIASES.get(name, name)
    if Path(name).exists():
        return name
    import glob

    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + name.replace("/", "--"), "snapshots", "*")))
    if snaps and os.environ.get("HF_HUB_OFFLINE") == "1":
        return snaps[-1]
    return name


def _find_calibration(model_path: str) -> str | None:
    """`calibration.json` next to the weights: in a local model directory, or in the Hub repository."""
    local = Path(model_path) / "calibration.json"
    if local.exists():
        return str(local)
    if Path(model_path).exists():
        return None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(model_path, "calibration.json")
    except Exception:
        return None


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .calibrate import Calibration
    from .engine import DecisionEngine
    from .server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from .mm_engine import local_model_dir

    model_path = local_model_dir(_resolve_model(args.model))
    calib_path = args.calibration or _find_calibration(model_path)
    calib = Calibration.load(calib_path) if calib_path else None
    if calib_path:
        logging.getLogger("qev.cli").info("calibration: %s", calib_path)
    config = json.loads((Path(model_path) / "config.json").read_text())
    if args.multimodal or config.get("vision_config") is not None:
        from .mm_engine import MMDecisionEngine

        engine = MMDecisionEngine(model_path, device=args.device, dtype=args.dtype, calibration=calib,
                                  fork_mode=args.fork_mode, head_dtype=args.head_dtype, media_root=args.media_root,
                                  allow_local_paths=args.media_root is not None,
                                  gpu_preprocess=not args.no_gpu_preprocess, cuda_graphs=not args.no_cuda_graphs,
                                  quantize=args.quantize, max_branch_tokens=args.max_branch_tokens,
                                  max_request_tokens=args.max_request_tokens)
    else:
        engine = DecisionEngine(model_path, device=args.device, dtype=args.dtype, calibration=calib,
                                fork_mode=args.fork_mode, head_dtype=args.head_dtype,
                                max_branch_tokens=args.max_branch_tokens, max_request_tokens=args.max_request_tokens)
    parts = args.model.split("/")
    default_name = "-".join(parts[1:]) if len(parts) > 2 else Path(args.model).name
    app = create_app(engine, model_name=args.name or default_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_ask(args: argparse.Namespace) -> None:
    from .client import Client

    body = json.load(open(args.request)) if args.request != "-" else json.load(sys.stdin)
    c = Client(args.url)
    r = c.system_one(body["state"], body["questions"], model=body.get("model"), media=body.get("media"),
                     **({"debug": 1} if args.debug else {}))
    print(json.dumps({"model": r.model, "answers": {k: v.raw for k, v in r.answers.items()}, "usage": r.usage, "qev": r.meta},
                     ensure_ascii=False, indent=2))


def cmd_models(args: argparse.Namespace) -> None:
    from .client import Client

    print(json.dumps(Client(args.url).models(), indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="qev", description="Open System One decision model and TypeSafe-compatible server")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the API server")
    s.add_argument("--model", default=DEFAULT_MODEL,
                   help="release alias (onejev-4b), HF id (OmniJev/OneJev-4B), or local path")
    s.add_argument("--name", default=None, help="model name reported by the API (default: last path component)")
    s.add_argument("--calibration", default=None,
                   help="calibration.json with temperatures (default: the one shipped next to the weights, if any)")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--head-dtype", default="float32", help="dtype of the LM head for the readout (float32 recommended)")
    s.add_argument("--fork-mode", default="auto", choices=["auto", "batched", "sequential"])
    s.add_argument("--multimodal", action="store_true",
                   help="vision checkpoint (OneJev): requests may carry media referenced as <image:N> / <video:N>")
    s.add_argument("--media-root", default=None,
                   help="with --multimodal, also accept media paths inside this directory (URLs and base64 always work)")
    s.add_argument("--no-cuda-graphs", action="store_true",
                   help="with --multimodal, run the decoder eagerly (CUDA graphs are captured per padded length on first use)")
    s.add_argument("--quantize", default=None, choices=["fp8"],
                   help="with --multimodal, run the decoder's linear layers in FP8 (an -FP8 checkpoint does this by itself)")
    s.add_argument("--no-gpu-preprocess", action="store_true",
                   help="with --multimodal, normalise images and frames on the CPU (they are resized on the CPU either way)")
    s.add_argument("--max-branch-tokens", type=int, default=32768,
                   help="longest state plus one question the server accepts (the bases take 262,144 positions)")
    s.add_argument("--max-request-tokens", type=int, default=65536, help="longest whole request the server accepts")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("ask", help="send a request JSON file (or - for stdin) to a running server")
    a.add_argument("request")
    a.add_argument("--url", default=os.environ.get("QEV_BASE_URL", "http://localhost:8000"))
    a.add_argument("--debug", action="store_true")
    a.set_defaults(func=cmd_ask)

    m = sub.add_parser("models", help="list models of a running server")
    m.add_argument("--url", default=os.environ.get("QEV_BASE_URL", "http://localhost:8000"))
    m.set_defaults(func=cmd_models)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
