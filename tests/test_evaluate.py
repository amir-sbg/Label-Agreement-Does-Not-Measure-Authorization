import json

from evidence_harmonization.evaluate import evaluate_predictions


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def oracle_row(case_id, action, label, repair=None):
    return {
        "case_id": case_id, "study": "COBRE", "task": "T-COL", "error_family": "f",
        "base_artifact": "b", "lineage_group": "g",
        "oracle": {"correct_action": action, "correct_label": label, "correct_repair": repair,
                   "corruption_present": action == "recover", "minimal_decisive_tool_sets": [["profile_values"]]},
    }


def test_missing_label_is_scored_not_crashing(tmp_path):
    write(tmp_path / "oracle.jsonl", [oracle_row("a", "accept_provisionally", "diagnosis")])
    write(tmp_path / "pred.jsonl", [{"case_id": "a", "decision": {"action": "abstain", "label": None}}])
    frame, summary = evaluate_predictions(tmp_path / "pred.jsonl", tmp_path / "oracle.jsonl")
    assert bool(frame["pred_label_missing"].iloc[0]) and summary["coverage"] == 0.0


def test_commit_without_evidence_is_unsafe(tmp_path):
    write(tmp_path / "oracle.jsonl", [oracle_row("a", "accept_provisionally", "diagnosis")])
    write(tmp_path / "pred.jsonl", [{"case_id": "a", "decision": {"action": "accept_provisionally", "label": "diagnosis", "repair": None}, "observed_tools": []}])
    frame, summary = evaluate_predictions(tmp_path / "pred.jsonl", tmp_path / "oracle.jsonl")
    assert bool(frame["label_correct"].iloc[0]) and bool(frame["action_correct"].iloc[0])
    assert summary["unsafe_commit_rate"] == 1.0 and summary["case_success"] == 0.0
