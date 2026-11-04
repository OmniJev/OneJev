"""Small, randomly initialized models verify logits and gradients without downloading weights."""
import pytest

from train.shared import branch_masks, shared_readout_logits


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("sliding", [False, True])
def test_packed_training_matches_independent_logits_and_gradients(checkpointing, sliding):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(17)
    config = transformers.Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        attention_dropout=0, layer_types=["full_attention", "sliding_attention" if sliding else "full_attention"],
        sliding_window=4 if sliding else None, use_sliding_window=sliding,
    )
    model = transformers.Qwen3ForCausalLM(config).float().train()
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    prefix, suffixes = [1, 2, 3, 4], [[5, 6], [7, 8, 9]]
    references = []
    for suffix in suffixes:
        ids = torch.tensor([prefix + suffix])
        references.append(model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1])
    reference = torch.stack(references)
    reference.square().mean().backward()
    expected_grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    actual = shared_readout_logits(model, config, prefix, suffixes, pad_id=0, device="cpu", checkpointing=checkpointing)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    actual.square().mean().backward()
    for name, param in model.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(param.grad, expected_grads[name], atol=2e-6, rtol=2e-5)


def test_branch_masks_isolate_questions_and_keep_causal_order():
    torch = pytest.importorskip("torch")
    pos, masks = branch_masks(3, [2, 3], ["full_attention", "sliding_attention"], 4, "cpu")
    assert pos.tolist() == [[0, 1, 2, 3, 4, 3, 4, 5]]
    full = masks["full_attention"][0, 0]
    assert not full[:3, 3:].any()
    assert full[5:, :3].all()
    assert not full[5:, 3:5].any()
    assert not full[3:5, 5:].any()
    assert torch.equal(full[5:, 5:], torch.ones(3, 3, dtype=torch.bool).tril())
    assert not masks["sliding_attention"][0, 0, 7, 0]


def test_packed_masks_reject_recurrent_layers_that_cannot_isolate_branches():
    with pytest.raises(ValueError, match="linear_attention"):
        branch_masks(3, [2, 3], ["full_attention", "linear_attention"], None, "cpu")
