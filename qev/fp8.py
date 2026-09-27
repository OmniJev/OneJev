"""8-bit floating point linear layers for the decoder: the weight in float8 (e4m3) with one scale per output row, the
input scaled per token on the fly, the product from torch._scaled_mm (cuBLAS FP8) in bfloat16. Plain torch, no extra
package. The per-token scale is clamped away from zero, so an all-zero token (padding) stays zero instead of 0/0.
"""
from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) weight -> (N, K) float8 and (N, 1) float32 scale, w ~ q * scale. The same bits on the CPU and the GPU:
    the scale is a multiplication by 1/448 (CUDA turns a division by a Python number into exactly that, the CPU
    divides), and the tensor-by-tensor division is IEEE on both."""
    scale = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) * (1.0 / FP8_MAX)
    return (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


def _quantize_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sx = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
    return (x.float() / sx).clamp(-FP8_MAX, FP8_MAX).to(FP8), sx


_compiled = None


def quantize_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, K) activations -> float8 and a (M, 1) float32 scale per token; one fused kernel through torch.compile
    (four separate float32 passes cost more than the FP8 matmul saves), eager if compilation is unavailable."""
    global _compiled
    if _compiled is None:
        try:
            _compiled = torch.compile(_quantize_tokens, dynamic=True)
        except Exception:
            _compiled = _quantize_tokens
    return _compiled(x)


class Fp8Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, out_dtype: torch.dtype):
        super().__init__()
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", bias)
        self.out_dtype = out_dtype

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> "Fp8Linear":
        q, s = quantize_rows(linear.weight.data)
        bias = None if linear.bias is None else linear.bias.data
        return cls(q, s, bias, linear.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xq, sx = quantize_tokens(x.reshape(-1, shape[-1]))
        y = torch._scaled_mm(xq, self.weight.t(), scale_a=sx, scale_b=self.weight_scale.t(), out_dtype=self.out_dtype)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)


def quantizable(name: str, shape) -> bool:
    """The decoder linear layers that go to FP8: 2-D weights under the language model, the embedding excluded, both
    sides multiples of 16 and at least 128 (the small gate projections stay). `name` is the module's qualified name."""
    return (name.startswith("model.language_model.") and "embed_tokens" not in name and len(shape) == 2
            and min(shape) >= 128 and shape[0] % 16 == 0 and shape[1] % 16 == 0)


def quantize_linears(model: torch.nn.Module, stored=None) -> int:
    """Replace every quantizable nn.Linear by an Fp8Linear, in place. `stored(name)` may return the (float8 weight,
    scale) pair saved in an FP8 checkpoint; otherwise the weight is quantized here, which gives the same tensors."""
    targets = [(name, m) for name, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and quantizable(name, tuple(m.weight.shape))]
    for name, m in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        pair = stored(name) if stored is not None else None
        if pair is None:
            new = Fp8Linear.from_linear(m)
        else:
            q, s = pair
            bias = None if m.bias is None else m.bias.data
            new = Fp8Linear(q.to(m.weight.device), s.to(m.weight.device), bias, m.weight.dtype)
        setattr(parent, child, new)
        del m
    torch.cuda.empty_cache()
    return len(targets)


QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "quantization_status": "compressed",
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True, "dynamic": False},
        "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "symmetric": True,
                              "dynamic": True},
    }},
    "ignore": ["lm_head", "re:.*visual.*", "re:.*in_proj_a$", "re:.*in_proj_b$"],
}


def is_fp8_checkpoint(config) -> bool:
    qc = getattr(config, "quantization_config", None)
    qc = qc if isinstance(qc, dict) else (qc.to_dict() if hasattr(qc, "to_dict") else {})
    return qc.get("quant_method") == "compressed-tensors" and qc.get("format") == "float-quantized"


def stored_reader(model_dir: str):
    """name -> (float8 weight, scale) from the safetensors files of an FP8 checkpoint."""
    import json
    from pathlib import Path

    from safetensors import safe_open

    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    where = json.loads(index.read_text())["weight_map"] if index.exists() else None
    handles = {}

    def read(name: str):
        key = f"{name}.weight"
        file = where.get(key) if where is not None else "model.safetensors"
        if file is None:
            return None
        f = handles.get(file) or handles.setdefault(file, safe_open(str(d / file), framework="pt"))
        if f"{name}.weight_scale" not in f.keys():
            return None
        return f.get_tensor(key), f.get_tensor(f"{name}.weight_scale")
    return read
