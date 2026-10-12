import time

from decision_index.engines import Engine, Unsupported


def _budgeted_engine_class():
    from qev.engine import DecisionEngine

    class BudgetedDecisionEngine(DecisionEngine):
        def _run_batched(self, state, rendered):
            import torch

            from qev.engine import log

            prefix_ids, suffixes = self._prefix_and_suffix_ids(state, rendered)
            dev = self.device
            P = len(prefix_ids)
            width = max(len(s) for s in suffixes)
            chunk = max(1, min(len(rendered), self.fork_token_budget // max(P + width, 1)))
            results = [None] * len(rendered)
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

    return BudgetedDecisionEngine


class QevEngine(Engine):
    name = "qev"
    latency = "In-process DecisionEngine.decide wall time, CUDA-synchronized, one request at a time; excludes model loading."

    def __init__(self, model=None, device="cuda:0", dtype="bfloat16", max_branch_tokens=131072, debias=1,
                 fork_token_budget=32_000, fork_mode="auto", **options):
        super().__init__(model=model, **options)
        import torch

        from qev.prompt import PROMPT_VERSION

        DecisionEngine = _budgeted_engine_class()

        self.torch = torch
        self.debias = int(debias)
        t0 = time.perf_counter()
        self.engine = DecisionEngine(model, device=device, dtype=dtype, max_branch_tokens=int(max_branch_tokens),
                                     max_request_tokens=max(2 * int(max_branch_tokens), 262144),
                                     fork_token_budget=int(fork_token_budget), fork_mode=fork_mode)
        self.load_seconds = time.perf_counter() - t0
        self.provenance = {"engine": "qev.engine.DecisionEngine (branch chunks sized by prefix + longest suffix)", "model": model, "prompt_version": PROMPT_VERSION,
                           "debias": self.debias, "dtype": dtype, "head_dtype": "float32",
                           "max_branch_tokens": int(max_branch_tokens), "fork_token_budget": int(fork_token_budget),
                           "fork_mode": fork_mode}

    def __call__(self, state, questions):
        from qev.schema import SystemOneRequest

        req = SystemOneRequest.model_validate({"model": "qev", "state": state, "questions": questions})
        try:
            resp, _ = self.engine.decide(req, debias=self.debias)
        except ValueError as exc:
            if "limit is" in str(exc) or "slots" in str(exc):
                raise Unsupported(str(exc)) from exc
            raise
        out = resp.model_dump()
        return {"model": "qev", "answers": out["answers"]}, None

    def synchronize(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def runtime(self):
        return {"torch": self.torch.__version__, "cuda": self.torch.version.cuda, "load_seconds": round(self.load_seconds, 1),
                "device": self.torch.cuda.get_device_name(0) if self.torch.cuda.is_available() else "cpu"}
