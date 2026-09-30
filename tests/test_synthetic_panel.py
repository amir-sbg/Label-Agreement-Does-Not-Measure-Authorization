from conftest import read_jsonl

from evidence_harmonization.evaluate import validate_benchmark_oracle_panel


def test_panel_size_and_tasks(synthetic_panel):
    cases = read_jsonl(synthetic_panel / "cases.jsonl")
    assert len(cases) == 90
    counts = {}
    for case in cases:
        counts[case["task"]] = counts.get(case["task"], 0) + 1
    assert counts == {"T-FILE": 24, "T-COL": 48, "T-VAL": 6, "T-ID": 6, "T-GOV": 6}


def test_cases_and_oracle_align(synthetic_panel):
    cases, oracle = validate_benchmark_oracle_panel(
        read_jsonl(synthetic_panel / "cases.jsonl"),
        read_jsonl(synthetic_panel / "oracle.jsonl"),
    )
    assert set(cases) == set(oracle)


def test_every_semantic_case_is_paired(synthetic_panel):
    oracle = read_jsonl(synthetic_panel / "oracle.jsonl")
    semantic = [row for row in oracle if row["task"] in {"T-FILE", "T-COL"}]
    by_artifact = {}
    for row in semantic:
        by_artifact.setdefault(row["base_artifact"], set()).add(row["oracle"]["corruption_present"])
    assert all(flags == {False, True} for flags in by_artifact.values())
