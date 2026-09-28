"""Turn the OneJev-Data Parquet shards into train.jsonl plus media files, the layout train.sft reads.

    python -m train.unpack data/onejev
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

FIELDS = ("id", "source", "task", "split", "license", "teacher")
JSON_FIELDS = ("state", "question", "target", "media", "meta")


def media_files(media: list | None) -> list[str]:
    out = []
    for m in media or []:
        out += [m["path"]] if m["type"] == "image" else list(m["frames"])
    return out


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/onejev")
    shards = sorted((root / "data").glob("train-*.parquet"))
    n = 0
    with open(root / "train.jsonl", "w") as out:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    record = {k: row[k] for k in FIELDS if row[k] is not None}
                    record.update({k: json.loads(row[k]) for k in JSON_FIELDS if row[k] is not None})
                    for path, image in zip(media_files(record.get("media")), row["images"] or []):
                        target = root / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(image["bytes"])
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n += 1
    print(f"{n} rows from {len(shards)} shards in {root}")


if __name__ == "__main__":
    main()
