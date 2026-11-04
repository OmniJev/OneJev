from types import SimpleNamespace

import pytest

from train import common


def test_legacy_readout_fallback_does_not_disable_other_models(monkeypatch):
    torch = pytest.importorskip("torch")
    if hasattr(common, "_TENSOR_KEEP"):
        monkeypatch.setattr(common, "_TENSOR_KEEP", True)

    class Model(torch.nn.Module):
        def __init__(self, accepts_tensor):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24)
            self.accepts_tensor = accepts_tensor
            self.calls = []

        def forward(self, input_ids, logits_to_keep, **kwargs):
            tensor = torch.is_tensor(logits_to_keep)
            self.calls.append(tensor)
            if tensor and not self.accepts_tensor:
                raise TypeError("integer readout positions required")
            logits = torch.nn.functional.embedding(input_ids, self.weight)
            return SimpleNamespace(logits=logits[:, logits_to_keep] if tensor else logits[:, -logits_to_keep:])

    ids = torch.tensor([[1, 2, 0, 0], [2, 3, 4, 5]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    legacy, modern = Model(False), Model(True)
    first = common.readout_logits(legacy, ids, mask, [1, 3])
    common.readout_logits(legacy, ids, mask, [1, 3])
    second = common.readout_logits(modern, ids, mask, [1, 3])
    assert legacy.calls == [True, False, False]
    assert modern.calls == [True]
    torch.testing.assert_close(first, second)
    first.square().sum().backward()
    second.square().sum().backward()
    torch.testing.assert_close(legacy.weight.grad, modern.weight.grad)


def test_out_of_memory_does_not_retry_readout_or_change_model_capability(monkeypatch):
    torch = pytest.importorskip("torch")
    if hasattr(common, "_TENSOR_KEEP"):
        monkeypatch.setattr(common, "_TENSOR_KEEP", True)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fail = True
            self.calls = []

        def forward(self, input_ids, logits_to_keep, **kwargs):
            self.calls.append(torch.is_tensor(logits_to_keep))
            if self.fail:
                self.fail = False
                raise torch.OutOfMemoryError("simulated allocation failure")
            return SimpleNamespace(logits=torch.ones(1, 1, 3))

    model = Model()
    ids = torch.tensor([[1, 2]])
    with pytest.raises(torch.OutOfMemoryError, match="allocation failure"):
        common.readout_logits(model, ids, torch.ones_like(ids), [1])
    assert model.calls == [True]
    common.readout_logits(model, ids, torch.ones_like(ids), [1])
    assert model.calls == [True, True]
