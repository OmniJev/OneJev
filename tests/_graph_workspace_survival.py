"""Run in a child process so a regressed CUDA fault cannot poison other tests."""
import gc

import torch

from qev.mm_engine import _capture_graph

torch.manual_seed(42)
stream = torch.cuda.Stream()
left = torch.randn(32768, 640, device="cuda", dtype=torch.bfloat16, requires_grad=True)
right = torch.randn(640, 640, device="cuda", dtype=torch.bfloat16, requires_grad=True)
seed = torch.ones_like(left)


def compute():
    product = left @ right
    return product, torch.autograd.grad(product, (left, right), seed)


stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    reference, reference_gradients = compute()
torch.cuda.synchronize()
kept = torch.cuda.CUDAGraph()
with _capture_graph(kept, stream=stream):
    output, gradients = compute()

for _ in range(4):
    discarded = torch.cuda.CUDAGraph()
    with _capture_graph(discarded, stream=stream):
        temporary, temporary_gradients = compute()
    discarded.replay()
    torch.cuda.synchronize()
    discarded.reset()
    del discarded, temporary, temporary_gradients
    gc.collect()
    torch.cuda.empty_cache()
    kept.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference)
    for actual, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, expected)
