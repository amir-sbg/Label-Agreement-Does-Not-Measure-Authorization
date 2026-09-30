from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _case_ids(rows: list[dict[str, Any]], source: str) -> list[str]:
    ids: list[str] = []
    for index, row in enumerate(rows):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"{source} row {index} has no valid case_id")
        ids.append(case_id)
    duplicates = sorted(
        case_id for case_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise RuntimeError(
            f"{source} contains {len(duplicates)} duplicate case IDs: {preview}"
        )
    return ids


def load_and_validate_shards(
    shards_dir: str | Path,
    oracle_path: str | Path,
    expected: int | None = None,
) -> list[dict[str, Any]]:
    paths = sorted(Path(shards_dir).glob("shard_*.jsonl"))
    if not paths:
        raise RuntimeError(f"No shard_*.jsonl files found in {shards_dir}")

    rows = [row for path in paths for row in _read_jsonl(path)]
    prediction_ids = _case_ids(rows, "predictions")
    oracle_rows = _read_jsonl(Path(oracle_path))
    oracle_ids = _case_ids(oracle_rows, "oracle")

    if expected is not None:
        if len(oracle_ids) != expected:
            raise RuntimeError(
                f"Expected {expected} oracle cases, found {len(oracle_ids)}"
            )
        if len(prediction_ids) != expected:
            raise RuntimeError(
                f"Expected {expected} prediction rows, found {len(prediction_ids)}"
            )

    prediction_set = set(prediction_ids)
    oracle_set = set(oracle_ids)
    missing = sorted(oracle_set - prediction_set)
    unexpected = sorted(prediction_set - oracle_set)
    if missing or unexpected:
        raise RuntimeError(
            "Prediction/oracle case-set mismatch: "
            f"{len(missing)} missing ({', '.join(missing[:5])}); "
            f"{len(unexpected)} unexpected ({', '.join(unexpected[:5])})"
        )

    by_id = {row["case_id"]: row for row in rows}
    return [by_id[case_id] for case_id in oracle_ids]
