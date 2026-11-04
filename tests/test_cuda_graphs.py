"""Small optional CUDA checks for graph cache lifetime, without model downloads."""
import gc
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("forked", [False, True], ids=["decoder", "fork"])
def test_graph_eviction_reuses_warmup_workspace(forked):
    if not (os.environ.get("QEV_TEST_GPU") or os.environ.get("QEV_TEST_MODEL")):
        pytest.skip("enable the optional GPU checks")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 9 * 2**30:
        pytest.skip("graph captures keep an 8 GiB free memory reserve")
    from qev.mm_engine import DecoderGraphs, ForkGraphs

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 64, device="cuda")
            self.proj = torch.nn.Linear(64, 64, bias=False, device="cuda")
            self.requires_grad_(False)

        def forward(self, inputs_embeds=None, input_ids=None, past_key_values=None, **kwargs):
            hidden = self.proj(inputs_embeds if inputs_embeds is not None else self.embed(input_ids))
            if past_key_values is not None:
                hidden = hidden + past_key_values[:, None]
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=hidden[:, -1].clone())

    with torch.inference_mode():
        decoder = Decoder()
        graphs = (ForkGraphs(decoder, lambda cache, n: cache.expand(n, -1).clone(), pad_id=0, max_graphs=2)
                  if forked else DecoderGraphs(decoder, max_graphs=2))

        def run(length):
            embeds = torch.randn(1, length, 64, device="cuda")
            positions = torch.arange(length, device="cuda").view(1, 1, -1).expand(3, 1, -1)
            if forked:
                result = graphs(embeds, positions, [[1, 2], [3]], delta=0)
                expected = decoder.proj(decoder.embed(torch.tensor([2, 3], device="cuda"))) + decoder.proj(embeds[:, -1])
                assert len(graphs.prefills) <= 2
                assert len(graphs.branches) <= 16
            else:
                result = graphs(embeds, positions)
                expected = decoder.proj(embeds[:, -1])
                assert len(graphs.graphs) <= 2
            torch.testing.assert_close(result, expected)

        def clear_and_measure():
            graphs.clear()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            return torch.cuda.memory_allocated()

        run(128)
        baseline = clear_and_measure()
        for index in range(12):
            run(128 * (1 + index % 4))
        retained = clear_and_measure() - baseline
        assert retained < 8 * 2**20, f"evicted graphs retained {retained / 2**20:.1f} MiB after warmup"


def test_graph_replay_survives_destruction_of_peer_graphs():
    if not (os.environ.get("QEV_TEST_GPU") or os.environ.get("QEV_TEST_MODEL")):
        pytest.skip("enable the optional GPU checks")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("_graph_workspace_survival.py"))],
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
