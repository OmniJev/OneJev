import time

from decision_index.engines.base import Engine, Unsupported
from decision_index.vision import images as img
from decision_index.vision.engines import resize_policy, single_choice

MAX_PIXELS = img.MAX_PIXELS


def to_request(questions, n_images, prompt=None):
    key, q, letters = single_choice(questions)
    texts = [str(q["criteria"][k]).strip() for k in letters]
    letter_keys = bool((prompt or {}).get("options_in_image")) or len(set(texts)) < len(texts) or any(not t for t in texts) \
        or texts == letters
    if letter_keys:
        criteria = {k: (None if t == k else t) for k, t in zip(letters, texts)}
        names = list(letters)
    else:
        criteria = {t: None for t in texts}
        names = texts
    if n_images == 1:
        state = {"image": "<image:1>"}
    else:
        state = {f"image {j + 1}": f"<image:{j + 1}>" for j in range(n_images)}
    question = {"type": "choice", "instructions": q["instructions"], "criteria": criteria}
    return key, letters, names, state, question


class QevVisionEngine(Engine):
    name = "qev-vision"
    latency = "In-process MMDecisionEngine.decide wall time including image decode and resize, one request at a time; excludes model loading."

    def __init__(self, model=None, device="cuda:0", dtype="bfloat16", max_branch_tokens=131072, debias=1,
                 fork_mode="auto", max_pixels=MAX_PIXELS, image_root=None, **options):
        super().__init__(model=model, **options)
        import torch

        from qev.mm_engine import MMDecisionEngine
        from qev.prompt import PROMPT_VERSION

        self.torch = torch
        self.debias = int(debias)
        self.max_pixels = None if max_pixels in (None, "null", 0) else int(max_pixels)
        self.image_root = image_root
        t0 = time.perf_counter()
        self.engine = MMDecisionEngine(model, device=device, dtype=dtype, max_branch_tokens=int(max_branch_tokens),
                                       max_request_tokens=max(2 * int(max_branch_tokens), 262144), fork_mode=fork_mode)
        self.load_seconds = time.perf_counter() - t0
        size = getattr(getattr(self.engine.processor, "image_processor", None), "size", None)
        try:
            size = {k: v for k, v in dict(size).items() if v is not None}
        except Exception:
            size = {k: v for k, v in vars(size).items() if v is not None and not k.startswith("_")} if size is not None else None
        self.provenance = {"engine": "qev.mm_engine.MMDecisionEngine", "model": model, "prompt_version": PROMPT_VERSION,
                           "debias": self.debias, "dtype": dtype, "head_dtype": "float32",
                           "max_branch_tokens": int(max_branch_tokens), "fork_mode": fork_mode,
                           "request_format": "images as <image:N> in state; option texts as option keys (letters when the options are in the image or not distinct)",
                           "max_pixels": self.max_pixels, "image_resize": resize_policy(self.max_pixels),
                           "processor_image_budget": size}

    def __call__(self, state, questions, images=(), prompt=None):
        from qev.schema import SystemOneRequest

        if not images:
            raise Unsupported("vision rows carry at least one image")
        key, letters, names, st, question = to_request(questions, len(images), prompt)
        pics = [img.pil(ref, self.image_root, max_pixels=self.max_pixels) for ref in images]
        media = [{"type": "image", "image": p} for p in pics]
        req = SystemOneRequest.model_validate({"model": "qev", "state": st, "questions": {key: question}})
        try:
            resp, meta = self.engine.decide(req, debias=self.debias, media=media)
        except ValueError as exc:
            if "limit is" in str(exc) or "slots" in str(exc):
                raise Unsupported(str(exc)) from exc
            raise
        ans = resp.model_dump()["answers"][key]
        to_letter = dict(zip(names, letters))
        probs = {to_letter[n]: float(p) for n, p in (ans.get("probabilities") or {}).items()}
        choice = to_letter[ans["choice"]]
        raw = {"image_pixels": [p.width * p.height for p in pics], "request_criteria": list(question["criteria"])}
        return {"model": "qev", "answers": {key: {"type": "choice", "choice": choice, "probabilities": probs}}}, raw

    def synchronize(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def runtime(self):
        return {"torch": self.torch.__version__, "cuda": self.torch.version.cuda, "load_seconds": round(self.load_seconds, 1),
                "device": self.torch.cuda.get_device_name(0) if self.torch.cuda.is_available() else "cpu"}
