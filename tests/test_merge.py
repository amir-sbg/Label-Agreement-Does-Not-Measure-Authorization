import json

import pytest

from evidence_harmonization.merge import load_and_validate_shards


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_merge_requires_exact_oracle_cover_and_uses_oracle_order(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    write_jsonl(shards / "shard_000.jsonl", [{"case_id": "b"}])
    write_jsonl(shards / "shard_001.jsonl", [{"case_id": "a"}])
    oracle = tmp_path / "oracle.jsonl"
    write_jsonl(oracle, [{"case_id": "a"}, {"case_id": "b"}])

    rows = load_and_validate_shards(shards, oracle, expected=2)

    assert [row["case_id"] for row in rows] == ["a", "b"]


def test_merge_rejects_duplicate_predictions(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    write_jsonl(shards / "shard_000.jsonl", [{"case_id": "a"}])
    write_jsonl(shards / "shard_001.jsonl", [{"case_id": "a"}])
    oracle = tmp_path / "oracle.jsonl"
    write_jsonl(oracle, [{"case_id": "a"}])

    with pytest.raises(RuntimeError, match="duplicate case IDs"):
        load_and_validate_shards(shards, oracle)


def test_merge_rejects_missing_and_unexpected_ids_at_equal_count(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    write_jsonl(
        shards / "shard_000.jsonl",
        [{"case_id": "a"}, {"case_id": "c"}],
    )
    oracle = tmp_path / "oracle.jsonl"
    write_jsonl(oracle, [{"case_id": "a"}, {"case_id": "b"}])

    with pytest.raises(RuntimeError, match="case-set mismatch"):
        load_and_validate_shards(shards, oracle, expected=2)


def test_merge_rejects_duplicate_oracle_ids(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    write_jsonl(shards / "shard_000.jsonl", [{"case_id": "a"}])
    oracle = tmp_path / "oracle.jsonl"
    write_jsonl(oracle, [{"case_id": "a"}, {"case_id": "a"}])

    with pytest.raises(RuntimeError, match="oracle contains"):
        load_and_validate_shards(shards, oracle)
