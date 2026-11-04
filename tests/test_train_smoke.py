"""Exercise the complete trainer with a tiny local checkpoint and no network access."""
import json

import pytest

from qev.prompt import SLOT_LABELS
from train import sft


@pytest.mark.parametrize("full", [True, False], ids=["full", "lora"])
def test_one_cpu_training_step_updates_weights_without_warmup(tmp_path, full):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not full:
        pytest.importorskip("peft")
    from tokenizers import Tokenizer, models, pre_tokenizers

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(19)
        vocab = {token: i for i, token in enumerate(("<unk>", "<pad>", "<eos>", *SLOT_LABELS))}
        raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>",
            chat_template="{% for m in messages %}{{m['role']}} {{m['content']}}\n{% endfor %}"
                          "{% if add_generation_prompt %}assistant {% endif %}",
        )
        config = transformers.Qwen3Config(
            vocab_size=len(vocab), hidden_size=32, intermediate_size=48, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=512,
            pad_token_id=1, eos_token_id=2, tie_word_embeddings=True,
        )
        model = transformers.Qwen3ForCausalLM(config)
        original = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        checkpoint = tmp_path / "base"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        data = tmp_path / "train.jsonl"
        data.write_text(json.dumps({"state": "The ball is red", "question": {"type": "noul"},
                                    "target": {"yes": 1}}) + "\n")

        mode = ["--full-finetune", "--no-gradient-checkpointing"] if full else [
            "--target-modules", "q_proj,v_proj", "--lora-r", "2", "--lora-alpha", "4", "--lora-dropout", "0",
        ]
        assert sft.main([
            "--model", str(checkpoint), "--train", str(data), "--runs-dir", str(tmp_path), "--name", "smoke",
            "--device", "cpu", "--dtype", "float32", "--no-autocast", "--no-augment",
            "--warmup-frac", "0", "--lr", "0.01", "--max-steps", "1", "--save-every", "0",
        ] + mode) == 0

        if full:
            trained = transformers.Qwen3ForCausalLM.from_pretrained(tmp_path / "smoke/final", dtype=torch.float32)
            actual = trained.model.layers[0].self_attn.q_proj.weight.detach()
            assert (actual - original.to(torch.bfloat16).float()).abs().max() > 0.001
        else:
            from safetensors.torch import load_file

            adapter = load_file(tmp_path / "smoke/adapter/adapter_model.safetensors")
            assert any(tensor.abs().max() > 0.001 for name, tensor in adapter.items() if "lora_B" in name)
        records = [json.loads(line) for line in (tmp_path / "smoke/log.jsonl").read_text().splitlines()]
        assert records[0]["peak_mem_gb"] == 0
        assert records[0]["lr"] == pytest.approx(0.01)
        assert records[-1]["steps"] == 1
        assert records[-1]["examples"] == 1
    finally:
        torch.set_num_threads(previous)
