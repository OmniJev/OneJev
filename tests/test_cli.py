import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qev import cli


def test_offline_model_uses_main_ref_instead_of_lexicographic_snapshot(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    repo = cache / "models--org--model"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs/main").write_text("aaa")
    for name in ("aaa", "zzz"):
        (repo / "snapshots" / name).mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert cli._resolve_model("org/model") == str(repo / "snapshots/aaa")


def test_model_alias_and_local_path(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert cli._resolve_model("onejev1.6-0.8b") == "OmniJev/OneJev1.6-0.8B"
    assert cli._resolve_model(str(tmp_path)) == str(tmp_path)


def test_ask_accepts_optional_state_and_reads_utf8(tmp_path, capsys):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"questions": {"完成": {"type": "noul"}}}, ensure_ascii=False), encoding="utf-8")
    response = SimpleNamespace(model="test", answers={}, usage={}, meta={})
    with patch("qev.client.Client") as client:
        client.return_value.system_one.return_value = response
        cli.cmd_ask(SimpleNamespace(request=str(path), url="http://test", debug=False))
    assert client.return_value.system_one.call_args.args == (None, {"完成": {"type": "noul"}})
    assert json.loads(capsys.readouterr().out)["model"] == "test"


@pytest.mark.parametrize("flag", ["--parallel", "--max-branch-tokens", "--max-request-tokens", "--max-image-bytes"])
def test_invalid_server_limits_are_rejected_before_loading(flag, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qev", "serve", flag, "0"])
    with patch.object(cli, "cmd_serve"), pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
