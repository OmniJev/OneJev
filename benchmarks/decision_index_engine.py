import time

from decision_index.engines import Engine, Unsupported


class QevEngine(Engine):
    name = "qev"
    latency = "In-process DecisionEngine.decide wall time, CUDA-synchronized, one request at a time; excludes model loading."

    def __init__(self, model=None, device="cuda:0", dtype="bfloat16", max_branch_tokens=131072, debias=1,
                 fork_token_budget=32_000, fork_mode="auto", cuda_graphs=1, **options):
        super().__init__(model=model, **options)
        import torch

        from qev.engine import DecisionEngine
        from qev.prompt import PROMPT_VERSION

        self.torch = torch
        self.debias = int(debias)
        t0 = time.perf_counter()
        self.engine = DecisionEngine(model, device=device, dtype=dtype, max_branch_tokens=int(max_branch_tokens),
                                     max_request_tokens=max(2 * int(max_branch_tokens), 262144),
                                     fork_token_budget=int(fork_token_budget), fork_mode=fork_mode,
                                     cuda_graphs=bool(int(cuda_graphs)))
        self.load_seconds = time.perf_counter() - t0
        self.provenance = {"engine": "qev.engine.DecisionEngine", "model": model, "prompt_version": PROMPT_VERSION,
                           "debias": self.debias, "dtype": dtype, "head_dtype": "float32",
                           "max_branch_tokens": int(max_branch_tokens), "fork_token_budget": int(fork_token_budget),
                           "fork_mode": fork_mode, "cuda_graphs": self.engine._graphs is not None}

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
