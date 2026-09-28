"""Score a trained model on eval files (unified or benchmark rows, with or without media), one GPU.

    python -m train.eval_mm --model train/runs/onejev_4b/final --multimodal --media-root data/onejev \
        --eval test=data/onejev/test.jsonl --out results.json

Frozen production rendering (no permutation, indent 2), the same readout as training. Reports per eval file:
accuracy, state-macro accuracy, NLL, ECE, and accuracy by question type, by source and by (group, family); the group
of a row is its meta group or, failing that, its source.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
from pathlib import Path

from train import common
from train.sft import collect_mm, load_eval_rows


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--multimodal", action="store_true")
    ap.add_argument("--media-root", default=None)
    ap.add_argument("--chat-template", default=None)
    ap.add_argument("--eval", dest="evals", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-preds", default=None, help="directory: one <name>.preds.jsonl per eval file (id, labels, "
                                                       "probabilities, gold) for per-row comparisons between models")
    args = ap.parse_args()

    import torch

    from qev.calibrate import nll as nll_fn
    from qev.calibrate import softmax as softmax_fn

    if args.media_root:
        from train import mm

        mm.set_root(args.media_root)
    bundle = common.load_bundle(args.model, device=args.device, dtype="bfloat16", multimodal=args.multimodal,
                                chat_template=args.chat_template)
    results = {}
    for spec in args.evals:
        name, _, path = spec.partition("=")
        rows = load_eval_rows(path)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if bundle.multimodal:
                scored = collect_mm(bundle, rows, args.max_len, args.batch_tokens, args.max_rows, args.device)
            else:
                scored = common.collect_slot_logits(bundle, rows, max_len=args.max_len, batch_tokens=args.batch_tokens,
                                                    max_rows=args.max_rows, device=args.device, quiet=True)
        per = collections.defaultdict(list)
        states = collections.defaultdict(list)
        dump = open(Path(args.dump_preds) / f"{name}.preds.jsonl", "w") if args.dump_preds else None
        bins = [[] for _ in range(10)]
        nlls = []
        for sc in scored:
            p = softmax_fn(sc.logits, 1.0)
            gold = max(range(len(sc.target)), key=sc.target.__getitem__)
            ok = int(max(range(len(p)), key=p.__getitem__) == gold)
            r = rows[sc.idx]
            meta = r.get("meta") or {}
            per["all"].append(ok)
            per[f"type={sc.kind}"].append(ok)
            per[f"source={r.get('source')}"].append(ok)
            per[f"group_family={meta.get('group', r.get('source'))}/{r.get('task')}"].append(ok)
            states[meta.get("state_id") or meta.get("item_id") or r["id"]].append(ok)
            bins[min(9, int(max(p) * 10))].append((max(p), ok))
            nlls.append(nll_fn(p, sc.target))
            if dump:
                dump.write(json.dumps({"id": r["id"], "task": r.get("task"), "kind": sc.kind, "labels": sc.labels,
                                       "probs": [round(x, 5) for x in p], "gold": gold, "ok": ok}) + "\n")
        if dump:
            dump.close()
        n = len(per["all"])
        ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins if b) if n else math.nan
        res = {"n": n, "skipped": len(rows) - n, "acc": sum(per["all"]) / max(n, 1),
               "state_macro": sum(sum(v) / len(v) for v in states.values()) / max(len(states), 1),
               "nll": sum(nlls) / max(n, 1), "ece": ece,
               "by": {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for k, v in sorted(per.items()) if k != "all"}}
        results[name] = res
        print(f"{name}: n {n} (skipped {res['skipped']}) acc {res['acc']:.4f} state_macro {res['state_macro']:.4f} "
              f"nll {res['nll']:.4f} ece {res['ece']:.4f}", flush=True)
        for k, v in res["by"].items():
            if k.startswith("type=") or k.startswith("group_family="):
                print(f"   {k}: {v['acc']:.3f} (n {v['n']})", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
