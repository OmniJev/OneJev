"""Latency of one request on one image: 1 question, 10 questions in one request, and the 10 as separate requests.
Median of warm runs, image decoding and preprocessing included (the setup of the README speed chart).

    python benchmarks/latency.py --model OmniJev/OneJev-4B --image screenshot.png
"""
from __future__ import annotations

import argparse
import statistics
import time

QUESTIONS = {
    "done": {"type": "noul", "instructions": "Has the agent already completed the task?"},
    "stuck": {"type": "noul", "instructions": "Is the agent stuck repeating the same action?"},
    "error": {"type": "noul", "instructions": "Does the screen show an error message?"},
    "login": {"type": "noul", "instructions": "Is a login or sign-in form visible?"},
    "next": {"type": "choice", "instructions": "What should the agent do next?",
             "criteria": {"click": "click an element", "type": "type text", "scroll": "scroll the page",
                          "wait": "wait for the page", "stop": "stop, the task is finished"}},
    "page": {"type": "choice", "instructions": "What kind of page is shown?",
             "criteria": {"form": "a form", "list": "a list or table", "article": "an article or document",
                          "dashboard": "a dashboard", "other": "something else"}},
    "progress": {"type": "score", "instructions": "How far along the task is the agent?",
                 "criteria": ["not started", "early", "halfway", "almost done", "done"]},
    "risk": {"type": "score", "instructions": "How risky is the agent's next action?",
             "criteria": ["harmless", "minor", "moderate", "serious"]},
    "clutter": {"type": "score", "instructions": "How cluttered is the screen?",
                "criteria": ["empty", "sparse", "moderate", "crowded"]},
    "success": {"type": "noul", "instructions": "Will the agent finish the task successfully?"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OmniJev/OneJev-4B")
    ap.add_argument("--image", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--quantize", default=None, choices=["fp8"])
    args = ap.parse_args()

    import torch

    from qev.media import data_uri
    from qev.mm_engine import MMDecisionEngine
    from qev.schema import SystemOneRequest

    engine = MMDecisionEngine(args.model, gpu_preprocess=True, cuda_graphs=True, quantize=args.quantize)
    state = {"task": "Finish the checkout on this shopping site", "screen": "<image:1>"}
    media = [{"type": "image", "data": data_uri(args.image)}]

    def timed(questions: dict) -> float:
        request = SystemOneRequest.model_validate({"state": state, "questions": questions})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.decide(request, media=media)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    first = {"done": QUESTIONS["done"]}
    for _ in range(2):
        timed(first), timed(QUESTIONS), [timed({k: v}) for k, v in QUESTIONS.items()]
    one, ten, apart = [], [], []
    for _ in range(args.runs):
        one.append(timed(first))
        ten.append(timed(QUESTIONS))
        apart.append(sum(timed({k: v}) for k, v in QUESTIONS.items()))
    ms = lambda xs: statistics.median(xs) * 1000
    print(f"{args.model} on {torch.cuda.get_device_name(0)}")
    print(f"  1 question                  {ms(one):7.0f} ms")
    print(f"  10 questions, one request   {ms(ten):7.0f} ms   ({ms(ten) / 10:.1f} ms per question)")
    print(f"  10 questions, 10 requests   {ms(apart):7.0f} ms")


if __name__ == "__main__":
    main()
