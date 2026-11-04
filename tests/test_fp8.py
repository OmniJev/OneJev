import inspect
import os
from unittest.mock import Mock

import pytest


@pytest.fixture
def fp8():
    pytest.importorskip("torch")
    from qev import fp8

    return fp8


def test_zero_rows_quantize_to_finite_zeros(fp8):
    import torch

    values = torch.zeros(3, 32)
    for quantize in (fp8.quantize_rows, fp8._quantize_tokens):
        quantized, scale = quantize(values)
        assert torch.isfinite(scale).all()
        assert (scale > 0).all()
        assert torch.equal(quantized.float(), values)


def test_row_quantization_reconstructs_weights_with_bounded_error(fp8):
    import torch

    values = torch.randn(16, 128, generator=torch.Generator().manual_seed(42))
    quantized, scale = fp8.quantize_rows(values)
    error = (quantized.float() * scale - values).abs().amax(1, keepdim=True)
    assert (error <= values.abs().amax(1, keepdim=True) / 16 + 1e-6).all()


def test_deferred_compile_failure_uses_eager_quantization(fp8, monkeypatch):
    import torch
    from torch._dynamo.exc import BackendCompilerFailed

    kwargs = {"backend_fn": lambda: None, "inner_exception": RuntimeError("compiler unavailable")}
    if "first_useful_frame" in inspect.signature(BackendCompilerFailed).parameters:
        kwargs["first_useful_frame"] = None
    compiled = Mock(side_effect=BackendCompilerFailed(**kwargs))
    monkeypatch.setattr(fp8, "_compiled", compiled)
    values = torch.ones(2, 32)
    for _ in range(2):
        quantized, scale = fp8.quantize_tokens(values)
        torch.testing.assert_close(quantized.float() * scale, values)
    assert compiled.call_count == 1


def test_runtime_errors_are_not_hidden_by_compiler_fallback(fp8, monkeypatch):
    import torch

    monkeypatch.setattr(fp8, "_compiled", Mock(side_effect=RuntimeError("invalid tensor")))
    with pytest.raises(RuntimeError, match="invalid tensor"):
        fp8.quantize_tokens(torch.ones(2, 32))


@pytest.mark.gpu
def test_fp8_linear_handles_unaligned_token_count(fp8, monkeypatch):
    import torch

    if not (os.environ.get("QEV_TEST_MODEL") or os.environ.get("QEV_TEST_GPU")):
        pytest.skip("enable the optional GPU checks")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 rowwise scaling needs a supported CUDA device")
    monkeypatch.setattr(fp8, "_compiled", fp8._quantize_tokens)
    torch.manual_seed(42)
    linear = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    quantized = fp8.Fp8Linear.from_linear(linear)
    expected, actual = linear(inputs).float(), quantized(inputs).float()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert ((actual - expected).square().mean() / expected.square().mean()) < 0.003
