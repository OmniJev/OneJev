import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qev.prompt import build_messages
from train import common, sft, shared


def test_load_rows_limit_and_split(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('\n'.join(json.dumps({"id": i, "split": "train" if i % 2 else "test"}) for i in range(5)))
    assert len(common.load_rows(path, limit=0)) == 0
    assert [r["id"] for r in common.load_rows(path, limit=1, split="train")] == [1]
    with pytest.raises(ValueError, match="limit"):
        common.load_rows(path, limit=-1)


@pytest.mark.parametrize("target", [{"yes": -1, "no": 2}, {"yes": float("nan")}, {"yes": float("inf")}])
def test_invalid_training_targets_are_rejected(target):
    with pytest.raises(ValueError, match="target"):
        common.target_vector(target, ["yes", "no"])


def test_training_targets_follow_permuted_labels():
    assert common.target_vector({"yes": 0.75, "no": 0.25}, ["no", "yes"]) == [0.25, 0.75]
    with pytest.raises(ValueError, match="no mass"):
        common.target_vector({"yes": 1}, ["other"])


def test_padding_already_full_choice_keeps_question():
    row = {"state": "x", "task": "next_action", "question": {"type": "choice", "criteria": {str(i): str(i) for i in range(255)}}}
    question = common.pad_options(row, {"next_action:tools": [("extra", "another tool")]}, prob=1, seed=0)
    assert question is row["question"]


@pytest.mark.parametrize("config", ["[]", "123", "true", '"hello"'])
def test_yaml_configuration_must_be_a_mapping(tmp_path, config):
    pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(config)
    with pytest.raises(ValueError, match="mapping"):
        common.load_yaml_config(path)


def test_yaml_configuration_rejects_normalized_key_collisions(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text("max-rows: 2\nmax_rows: 3\n")
    with pytest.raises(ValueError, match="duplicate"):
        common.load_yaml_config(path)


@pytest.mark.parametrize("flag,value", [
    ("--epochs", "0"), ("--max-rows", "0"), ("--max-len", "-1"),
    ("--batch-tokens", "0"), ("--log-every", "0"), ("--lr", "nan"),
    ("--lr", "-1"), ("--warmup-frac", "2"), ("--pad-options", "1.1"),
])
def test_training_flags_fail_before_model_loading(flag, value):
    with pytest.raises(SystemExit) as exc:
        sft.parse_args(["--model", "unused", "--train", "unused", flag, value])
    assert exc.value.code == 2


def test_shared_groups_keep_multimodal_rows_in_regular_batches():
    rows = [{"state": "same"}, {"state": "same"},
            {"state": "<image:1>", "media": [{"type": "image", "path": "a.png"}]},
            {"state": "<image:1>", "media": [{"type": "image", "path": "b.png"}]}]
    groups, singles = shared.group_rows(rows)
    assert groups == [[0, 1]]
    assert sorted(singles) == [2, 3]


class FakeBundle:
    multimodal = False

    def __init__(self):
        self.tokenizer = lambda texts, **kw: {"input_ids": [list(s.encode()) for s in texts]}

    def encode(self, text):
        return list(text.encode())

    def render_prompt(self, state, suffix):
        return "\n".join(m["content"] for m in build_messages(state, suffix))

    def template_for(self, row):
        return self.render_prompt


def test_shared_planner_uses_original_row_indices_for_single_row_seeds():
    rows = [{"state": state, "question": {"type": "noul"}, "target": {"yes": 1}} for state in ("shared", "shared", "unique")]
    args = sft.parse_args(["--model", "unused", "--train", "unused"])
    args.seed = 7
    seeds = []
    original = common.render_example

    def record(*args, **kwargs):
        seeds.append(kwargs["seed"])
        return original(*args, **kwargs)

    with patch.object(common, "render_example", side_effect=record):
        sft.plan_epoch_shared(FakeBundle(), rows, args, epoch=3)
    assert seeds == [7 * 1000003 + 3 * 1000033 + 2]


def test_cpu_gloo_setup_does_not_select_cuda_device(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    args = SimpleNamespace(device="cpu", fsdp=False, dist_backend="gloo")
    with patch.object(torch.cuda, "set_device") as select, patch("torch.distributed.init_process_group"):
        assert sft.dist_setup(args) == (0, 2)
    select.assert_not_called()


def test_shared_planner_caps_packed_tokens_and_rows_without_losing_examples():
    rows = [{"state": "shared", "question": {"type": "noul", "instructions": f"Question {i}"},
             "target": {"yes": 1}} for i in range(20)]
    args = sft.parse_args(["--model", "unused", "--train", "unused", "--max-rows", "3", "--batch-tokens", "700"])
    bundle = FakeBundle()
    steps, dropped, _ = sft.plan_epoch_shared(bundle, rows, args, epoch=0)
    observed = []
    for step in steps:
        for kind, indices in step:
            observed.extend(indices)
            assert len(indices) <= args.max_rows
            if kind == "group":
                seed = args.seed * 1000003 + indices[0]
                state, suffixes, _, _, _ = shared.render_group(rows, indices, seed, args.augment)
                prefix, suffix_ids = shared.split_prefix(bundle, state, suffixes)
                assert len(prefix) + sum(map(len, suffix_ids)) <= args.batch_tokens
    assert dropped == 0
    assert sorted(observed) == list(range(len(rows)))


def test_plan_only_does_not_require_peft(tmp_path):
    pytest.importorskip("torch")
    with patch.dict("sys.modules", {"peft": None}), \
            patch.object(common, "load_rows", return_value=[{"id": "test"}]), \
            patch.object(common, "load_bundle", return_value=FakeBundle()), \
            patch.object(sft, "plan_only", return_value=0) as plan:
        result = sft.main(["--model", "unused", "--train", "unused", "--plan-only", "--runs-dir", str(tmp_path)])
    assert result == 0
    plan.assert_called_once()


def test_plan_only_uses_the_requested_rank_count(monkeypatch, capsys):
    monkeypatch.delenv("PLAN_WORLD", raising=False)
    rows = [{"state": str(i), "question": {"type": "noul"}, "target": {"yes": 1}} for i in range(4)]
    args = sft.parse_args(["--model", "unused", "--train", "unused", "--max-rows", "1"])
    assert sft.plan_only(FakeBundle(), rows, args, world=2) == 0
    assert "2 steps for 2 ranks" in capsys.readouterr().out
    monkeypatch.setenv("PLAN_WORLD", "4")
    assert sft.plan_only(FakeBundle(), rows, args, world=2) == 0
    assert "1 steps for 4 ranks" in capsys.readouterr().out
    monkeypatch.setenv("PLAN_WORLD", "0")
    with pytest.raises(ValueError, match="positive"):
        sft.plan_only(FakeBundle(), rows, args, world=2)


def test_no_usable_training_rows_does_not_save_an_untrained_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    bundle = FakeBundle()
    bundle.model = torch.nn.Linear(2, 2)
    bundle.model.config = SimpleNamespace(use_cache=True)
    bundle.model_type, bundle.template_kwargs = "stub", {}
    bundle.tokenizer.pad_token_id = 0
    row = {"state": "example", "question": {"type": "noul"}, "target": {"yes": 1}}
    with patch.object(common, "load_rows", return_value=[row]), \
            patch.object(common, "load_bundle", return_value=bundle), \
            patch.object(sft, "save_weights") as save:
        with pytest.raises(SystemExit, match="no.*max-len"):
            sft.main(["--model", "unused", "--train", "unused", "--runs-dir", str(tmp_path),
                      "--device", "cpu", "--full-finetune", "--no-gradient-checkpointing", "--max-len", "1"])
    save.assert_not_called()
