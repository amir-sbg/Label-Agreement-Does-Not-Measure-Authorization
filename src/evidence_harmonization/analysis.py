from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .evaluate import evaluate_predictions
from .llm import read_jsonl
from .stats import cluster_robust_ci, mcnemar_exact

SEMANTIC_TASKS = {"T-FILE", "T-COL"}


def by_case(path: str | Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in read_jsonl(path)}


def proposal(case: dict[str, Any]) -> str | None:
    observation = case.get("initial_observation", {})
    if case["task"] == "T-FILE":
        return observation.get("proposed_primary_role")
    if case["task"] == "T-COL":
        return observation.get("proposed_concept")
    return None


def arm_metrics(predictions: str | Path, oracle: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, summary = evaluate_predictions(predictions, oracle)
    truth = by_case(oracle)
    frame["has_repair_target"] = frame["case_id"].map(
        lambda case_id: truth[case_id]["oracle"].get("correct_repair") is not None
    )
    committed = frame[frame["committed"]]
    targets = frame[frame["has_repair_target"]]
    metrics = {
        "n": int(len(frame)),
        "label": float(frame["label_correct"].mean()),
        "action": float(summary["action_accuracy"]),
        "case": float(summary["case_success"]),
        "unsafe": float(summary["unsafe_commit_rate"]),
        "coverage": float(summary["coverage"]),
        "parse_success": float(summary["parse_success_rate"]),
        "label_missing": int(frame["pred_label_missing"].sum()),
        "committed_n": int(len(committed)),
        "label_given_commit": [int(committed["label_correct"].sum()), int(len(committed))],
        "action_given_commit": [int(committed["action_correct"].sum()), int(len(committed))],
        "unsafe_given_commit": float(committed["unsafe_commit"].mean()) if len(committed) else None,
        "exact_repair_on_targets": [int(targets["exact_repair"].sum()), int(len(targets))],
    }
    return frame, metrics


def paired_contrast(left: pd.DataFrame, right: pd.DataFrame, metric: str) -> dict[str, Any]:
    merged = left[["case_id", metric, "lineage_group"]].merge(
        right[["case_id", metric]], on="case_id", suffixes=("_left", "_right"), validate="one_to_one"
    )
    diff = merged[f"{metric}_right"].astype(float) - merged[f"{metric}_left"].astype(float)
    return {
        "metric": metric,
        **cluster_robust_ci(diff, merged["lineage_group"]),
        "mcnemar": mcnemar_exact(merged[f"{metric}_left"], merged[f"{metric}_right"]),
    }


def ucr_intersection(
    cases: dict[str, dict[str, Any]],
    oracle: dict[str, dict[str, Any]],
    hidden: dict[str, dict[str, Any]],
    visible: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    hidden_ucr = visible_ucr = n = 0
    for case_id, case in cases.items():
        truth = oracle[case_id]["oracle"]
        if case["task"] not in SEMANTIC_TASKS or not truth.get("corruption_present"):
            continue
        hd, vd = hidden[case_id]["decision"], visible[case_id]["decision"]
        if hd.get("label") != truth.get("correct_label") or vd.get("label") != truth.get("correct_label"):
            continue
        prop = proposal(case)
        n += 1
        hidden_ucr += int(hd.get("action") != "recover" and hd.get("label") != prop)
        visible_ucr += int(vd.get("action") != "recover" and vd.get("label") != prop)
    return {"n": n, "hidden": hidden_ucr, "visible": visible_ucr}


def contradictory_approvals(
    cases: dict[str, dict[str, Any]],
    oracle: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, int]:
    count = n = 0
    for case_id, case in cases.items():
        gold = oracle[case_id]["oracle"]
        if case["task"] not in SEMANTIC_TASKS or not gold.get("corruption_present"):
            continue
        n += 1
        decision = predictions[case_id].get("decision", {})
        label = decision.get("label")
        count += int(
            gold.get("correct_action") == "recover"
            and label == gold.get("correct_label")
            and label != proposal(case)
            and decision.get("action") != "recover"
        )
    return {"count": count, "n": n}


def contract_compliance(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(predictions.values())
    decisions = [row.get("decision", {}) for row in rows]
    values = [decision.get("proposal_matches_my_label") for decision in decisions]
    return {
        "n": len(rows),
        "parse_success": sum(bool(row.get("parse_success")) for row in rows),
        "boolean_present": sum(value is not None for value in values),
        "boolean_true": sum(value is True for value in values),
        "boolean_false": sum(value is False for value in values),
        "label_missing": sum(decision.get("label") is None for decision in decisions),
    }
