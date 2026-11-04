import json
import sys

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from train import unpack


def write_shard(root, paths, images):
    rows = [{
        **dict.fromkeys(unpack.FIELDS),
        **dict.fromkeys(unpack.JSON_FIELDS),
        "id": "demo", "state": json.dumps({"screen": "<image:1>"}),
        "media": json.dumps([{"type": "image", "path": p} for p in paths]),
        "images": [{"bytes": data, "path": None} for data in images],
    }]
    shard = root / "data" / "train-00000.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), shard)


def run_unpack(monkeypatch, root):
    monkeypatch.setattr(sys, "argv", ["train.unpack", str(root)])
    unpack.main()


def test_unpack_round_trip(tmp_path, monkeypatch):
    write_shard(tmp_path, ["media/猫.png"], [b"image bytes"])
    run_unpack(monkeypatch, tmp_path)
    record = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8"))
    assert record["id"] == "demo"
    assert record["state"] == {"screen": "<image:1>"}
    assert (tmp_path / "media/猫.png").read_bytes() == b"image bytes"


@pytest.mark.parametrize("path", ["../escaped.png", "media/../../escaped.png"])
def test_unpack_rejects_path_traversal_and_preserves_manifest(tmp_path, monkeypatch, path):
    root = tmp_path / "dataset"
    root.mkdir()
    write_shard(root, [path], [b"must stay inside dataset"])
    manifest = root / "train.jsonl"
    manifest.write_text("existing manifest\n")
    with pytest.raises(ValueError, match="outside|relative"):
        run_unpack(monkeypatch, root)
    assert manifest.read_text() == "existing manifest\n"
    assert not (tmp_path / "escaped.png").exists()


def test_unpack_rejects_absolute_path(tmp_path, monkeypatch):
    outside = tmp_path / "outside.png"
    root = tmp_path / "dataset"
    root.mkdir()
    write_shard(root, [str(outside)], [b"outside"])
    with pytest.raises(ValueError, match="relative"):
        run_unpack(monkeypatch, root)
    assert not outside.exists()


def test_unpack_does_not_silently_truncate_media(tmp_path, monkeypatch):
    write_shard(tmp_path, ["media/a.png", "media/b.png"], [b"only one image"])
    with pytest.raises(ValueError, match="media.*images|images.*media"):
        run_unpack(monkeypatch, tmp_path)
    assert not (tmp_path / "train.jsonl").exists()


def test_unpack_does_not_follow_media_symlinks_outside_root(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "media").symlink_to(outside, target_is_directory=True)
    write_shard(root, ["media/a.png"], [b"outside"])
    with pytest.raises(ValueError, match="outside"):
        run_unpack(monkeypatch, root)
    assert not (outside / "a.png").exists()


def test_unpack_deduplicates_identical_media_references(tmp_path, monkeypatch):
    write_shard(tmp_path, ["media/a.png", "media/a.png"], [b"same bytes", b"same bytes"])
    run_unpack(monkeypatch, tmp_path)
    assert (tmp_path / "media/a.png").read_bytes() == b"same bytes"
    assert len(list((tmp_path / "media").iterdir())) == 1


def test_unpack_rejects_conflicting_media_references(tmp_path, monkeypatch):
    write_shard(tmp_path, ["media/a.png", "media/a.png"], [b"first", b"second"])
    with pytest.raises(ValueError, match="conflicting"):
        run_unpack(monkeypatch, tmp_path)
    assert (tmp_path / "media/a.png").read_bytes() == b"first"
    assert not (tmp_path / "train.jsonl").exists()
    assert not list(tmp_path.glob(".train-*"))


@pytest.mark.parametrize("path", ["train.jsonl", "data/train-00000.parquet"])
def test_unpack_media_cannot_overwrite_manifest_or_source(tmp_path, monkeypatch, path):
    write_shard(tmp_path, [path], [b"not a dataset"])
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("existing\n")
    source = tmp_path / "data/train-00000.parquet"
    original = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        run_unpack(monkeypatch, tmp_path)
    assert manifest.read_text() == "existing\n"
    assert source.read_bytes() == original
