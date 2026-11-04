"""Measure request planning with a real tokenizer, without loading weights or running a forward pass.

    python -m benchmarks.request_planning --tokenizer /path/to/local/OneJev --baseline-ref HEAD
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from qev.calibrate import Calibration
from qev.engine import BranchResult, DecisionEngine
from qev.prompt import DEFAULT_STYLE, MAX_SLOTS
from qev.schema import SystemOneRequest


def planning_engine(base, tokenizer):
    class PlanningEngine(base):
        def __init__(self):
            self.tokenizer = tokenizer
            self.template_kwargs = {"enable_thinking": False}
            self.style = DEFAULT_STYLE
            self.max_slots = MAX_SLOTS
            self.max_branch_tokens = self.max_request_tokens = 1_000_000
            self.calibration = Calibration()
            self.info = SimpleNamespace(fork_mode="sequential")
            self._lock = threading.RLock()
            self.encode_calls = 0

        def _encode(self, text):
            self.encode_calls += 1
            return self.tokenizer.encode(text, add_special_tokens=False)

        def _run_sequential(self, state, rendered):
            _, suffixes = self._prefix_and_suffix_ids(state, rendered)
            return [BranchResult(r.qid, r.kind, r.labels, r.descriptions, [0.0] * r.n_slots, 1.0, len(s))
                    for r, s in zip(rendered, suffixes)]

    return PlanningEngine()


def measure(engine, request, repeats):
    response, meta = engine.decide(request)
    times = []
    engine.encode_calls = 0
    for _ in range(repeats):
        start = time.perf_counter()
        response, meta = engine.decide(request)
        times.append(time.perf_counter() - start)
    return {
        "median_ms": statistics.median(times) * 1000,
        "encode_calls_per_request": engine.encode_calls / repeats,
        "input_tokens": response.usage.input_tokens,
        "prompt_sha256": meta["prompt_sha256"],
    }, response.model_dump()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path, help="local checkpoint directory; weights are not loaded")
    parser.add_argument("--baseline-ref", help="optional Git revision to compare, e.g. HEAD")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--questions", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--state-chars", type=int, nargs="+", default=[4000, 40000])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.tokenizer.is_dir():
        parser.error("--tokenizer must be a local directory")
    if min([args.repeats, *args.questions, *args.state_chars]) < 1:
        parser.error("repeat, question and state sizes must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    engines = {"current": planning_engine(DecisionEngine, tokenizer)}
    with tempfile.TemporaryDirectory() as temporary:
        if args.baseline_ref:
            path = Path(temporary) / "baseline_engine.py"
            path.write_bytes(subprocess.check_output(["git", "show", f"{args.baseline_ref}:qev/engine.py"]))
            spec = importlib.util.spec_from_file_location("qev._planning_baseline", path)
            baseline = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = baseline
            spec.loader.exec_module(baseline)
            engines["baseline"] = planning_engine(baseline.DecisionEngine, tokenizer)
        results = []
        for chars in args.state_chars:
            state = ("The ticket is about a duplicate payment. " * (chars // 40 + 1))[:chars]
            for count in args.questions:
                request = SystemOneRequest(state=state, questions={f"q{i}": {"type": "noul", "instructions": f"Is question {i} true?"}
                                                                  for i in range(count)})
                result = {"state_chars": chars, "questions": count}
                responses = []
                for label, engine in engines.items():
                    result[label], response = measure(engine, request, args.repeats)
                    responses.append(response)
                assert all(response == responses[0] for response in responses), "planning changed the assembled answers"
                assert len({result[label]["prompt_sha256"] for label in engines}) == 1, "rendered prompt changed"
                if "baseline" in result:
                    result["speedup"] = result["baseline"]["median_ms"] / result["current"]["median_ms"]
                results.append(result)
    output = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
