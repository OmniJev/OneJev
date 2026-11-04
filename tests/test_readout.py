from types import SimpleNamespace

import pytest

from qev.engine import DecisionEngine


@pytest.mark.parametrize("head_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("vocab", [64, 129])
def test_debug_keeps_slot_precision_and_matches_full_vocabulary_mass(head_dtype, vocab, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("qev.engine.HEAD_CHUNK_BYTES", 4096)
    torch.manual_seed(42)
    head = torch.nn.Linear(16, vocab, bias=True, dtype=torch.bfloat16)
    head.requires_grad_(False)
    engine = object.__new__(DecisionEngine)
    engine.model = SimpleNamespace(get_output_embeddings=lambda: head)
    engine.slot_ids = [0, vocab // 2, vocab - 1]
    engine.head_dtype = head_dtype
    engine._init_slot_head()
    hidden = torch.randn(2, 16, dtype=torch.bfloat16)
    engine._want_mass = False
    ordinary = engine._readout_rows(hidden, [2, 3])
    engine._want_mass = True
    diagnostic = engine._readout_rows(hidden, [2, 3])
    assert [row[0] for row in ordinary] == [row[0] for row in diagnostic]
    dtype = getattr(torch, head_dtype)
    reference = torch.nn.functional.linear(hidden.to(dtype), head.weight.to(dtype), head.bias.to(dtype)).float()
    probabilities = reference.softmax(-1)
    for row, count in enumerate([2, 3]):
        expected = probabilities[row, engine.slot_ids[:count]].sum().item()
        assert diagnostic[row][1] == pytest.approx(expected, rel=2e-5, abs=1e-8)
    assert head.weight.dtype == torch.bfloat16
