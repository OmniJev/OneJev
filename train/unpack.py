"""Turn the OneJev-Data Parquet shards into train.jsonl plus media files, the layout train.sft reads.

    python -m train.unpack data/onejev
    python -m train.unpack data/onejev --sample
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

FIELDS = ("id", "source", "task", "split", "license", "teacher")
JSON_FIELDS = ("state", "question", "target", "media", "meta")


def media_files(media: list | None) -> list[str]:
    out = []
    for m in media or []:
        out += [m["path"]] if m["type"] == "image" else list(m["frames"])
    return out


def _media_target(root: Path, path: str) -> Path:
    """Resolve a dataset-relative file without following it out of the extraction root."""
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ValueError(f"media path must be a non-empty relative path: {path!r}")
    target = (root / path).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError(f"media path {path!r} is outside the dataset root")
    if target == root / "train.jsonl":
        raise ValueError("media cannot overwrite the training manifest")
    return target


def _write_image(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=".image-", delete=False) as out:
            pending = Path(out.name)
            out.write(data)
        pending.replace(target)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/onejev"))
    parser.add_argument("--sample", action="store_true", help="unpack only the 100-example demo")
    args = parser.parse_args()
    root = args.root.resolve()
    pattern = "sample/sample-100.parquet" if args.sample else "data/train-*.parquet"
    shards = sorted(root.glob(pattern))
    if not shards:
        parser.error(f"no files match {root / pattern}; download the selected dataset first")
    n = 0
    sources = {p.resolve() for p in shards}
    written: dict[Path, bytes] = {}
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, prefix=".train-", suffix=".jsonl.tmp",
                                         delete=False) as out:
            pending = Path(out.name)
            for shard in shards:
                with pq.ParquetFile(shard) as parquet:
                    for batch in parquet.iter_batches(batch_size=64):
                        for row in batch.to_pylist():
                            record = {k: row[k] for k in FIELDS if row[k] is not None}
                            record.update({k: json.loads(row[k]) for k in JSON_FIELDS if row[k] is not None})
                            paths, images = media_files(record.get("media")), row["images"] or []
                            if len(paths) != len(images):
                                raise ValueError(f"row {record.get('id')!r}: {len(paths)} media references but {len(images)} images")
                            targets = [_media_target(root, path) for path in paths]
                            if any(target in sources for target in targets):
                                raise ValueError("media cannot overwrite a source Parquet shard")
                            if any(not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes) for image in images):
                                raise ValueError(f"row {record.get('id')!r}: images must contain embedded bytes")
                            for target, image in zip(targets, images):
                                digest = hashlib.sha256(image["bytes"]).digest()
                                if target in written:
                                    if written[target] != digest:
                                        raise ValueError(f"conflicting images for media path {str(target.relative_to(root))!r}")
                                    continue
                                _write_image(target, image["bytes"])
                                written[target] = digest
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            n += 1
        pending.replace(root / "train.jsonl")
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    print(f"{n} rows from {len(shards)} shards in {root}")


if __name__ == "__main__":
    main()
