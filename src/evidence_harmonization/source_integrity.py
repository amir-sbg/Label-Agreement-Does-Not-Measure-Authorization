from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import data_path
from .source_io import sha256_file


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_source_integrity(cards_path: str | Path) -> dict[str, Any]:
    rows = []
    for card in read_jsonl(cards_path):
        locator = card["source_path"]
        source = Path(data_path(locator.removeprefix("neuromark:")) if locator.startswith("neuromark:") else locator)
        expected = str(card["source_sha256"])
        if source.exists():
            observed = sha256_file(source)
            status = "match" if observed == expected else "mismatch"
        else:
            observed = None
            status = "missing"
        rows.append({
            "artifact_id": card["artifact_id"],
            "study": card["study"],
            "source_path": locator,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "status": status,
        })
    failures = [row for row in rows if row["status"] != "match"]
    return {
        "status": "pass" if not failures else "fail",
        "artifact_count": len(rows),
        "match_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "artifacts": rows,
    }


def write_source_integrity(summary: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
