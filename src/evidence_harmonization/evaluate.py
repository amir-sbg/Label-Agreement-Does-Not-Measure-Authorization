from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


LABEL_MISSING = "__missing__"


def _label_series(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), LABEL_MISSING).astype(str)


def index_case_rows(
    rows: list[dict[str, Any]], source: str
) -> dict[str, dict[str, Any]]:
    if not rows:
        raise RuntimeError(f"{source} contains no rows")
    indexed: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(
                f"{source} row {row_index} has no valid case_id"
            )
        if case_id in indexed:
            raise RuntimeError(
                f"{source} contains duplicate case_id: {case_id}"
            )
        indexed[case_id] = row
    return indexed


def require_exact_case_set(
    actual: dict[str, Any],
    expected: dict[str, Any],
    actual_name: str,
    expected_name: str,
) -> None:
    actual_ids = set(actual)
    expected_ids = set(expected)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing or unexpected:
        raise RuntimeError(
            f"{actual_name}/{expected_name} case-set mismatch: "
            f"{len(missing)} missing ({', '.join(missing[:5])}); "
            f"{len(unexpected)} unexpected ({', '.join(unexpected[:5])})"
        )


def validate_benchmark_oracle_panel(
    case_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = index_case_rows(case_rows, "benchmark")
    oracle = index_case_rows(oracle_rows, "oracle")
    require_exact_case_set(oracle, cases, "oracle", "benchmark")
    metadata_fields = (
        "study",
        "task",
        "error_family",
        "base_artifact",
        "lineage_group",
    )
    for case_id, case in cases.items():
        truth = oracle[case_id]
        mismatched = [
            field
            for field in metadata_fields
            if case.get(field) != truth.get(field)
        ]
        if mismatched:
            raise RuntimeError(
                f"Benchmark/oracle metadata mismatch for {case_id}: "
                f"{', '.join(mismatched)}"
            )
    return cases, oracle


def evaluate_predictions(
    predictions_path: str | Path,
    oracle_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = index_case_rows(_read_jsonl(predictions_path), "predictions")
    oracle = index_case_rows(_read_jsonl(oracle_path), "oracle")
    require_exact_case_set(predictions, oracle, "predictions", "oracle")
    rows: list[dict[str, Any]] = []
    for case_id, truth in oracle.items():
        pred = predictions[case_id]
        decision = pred.get("decision", {})
        gold = truth["oracle"]
        called = pred.get("called_tools", [])
        observed = pred.get("observed_tools", called)
        minimal_sets = gold.get("minimal_decisive_tool_sets", [])
        evidence_required = any(bool(req) for req in minimal_sets)
        acquired = any(set(req).issubset(set(observed)) for req in minimal_sets)
        minimal_tools = set().union(*(set(req) for req in minimal_sets))
        nonminimal_calls = sum(tool not in minimal_tools for tool in called)
        history = pred.get("controller_history", [])
        trajectory = pred.get("trajectory", [])
        statuses = [str(item.get("status")) for item in history]
        unavailable_calls = sum(
            step.get("type") == "tool"
            and step.get("result", {}).get("status") == "unavailable"
            for step in trajectory
        )
        relevant_call_n = sum(tool in minimal_tools for tool in called)
        relevant_tool_precision = (
            relevant_call_n / len(called) if called else np.nan
        )
        exact_repair = decision.get("repair") == gold.get("correct_repair")
        action_correct = decision.get("action") == gold["correct_action"]
        label_correct = decision.get("label") == gold.get("correct_label")
        evidence_supported = not evidence_required or acquired
        committed = decision.get("action") in {"recover", "accept_provisionally"}
        unsafe_commit = committed and not (action_correct and label_correct and exact_repair and evidence_supported)
        rows.append(
            {
                "case_id": case_id,
                "study": truth["study"],
                "task": truth["task"],
                "error_family": truth["error_family"],
                "base_artifact": truth["base_artifact"],
                "lineage_group": truth["lineage_group"],
                "oracle_status": gold.get("oracle_status", "unspecified"),
                "gold_corruption": bool(gold["corruption_present"]),
                "gold_action": gold["correct_action"],
                "pred_action": decision.get("action"),
                "gold_label": gold.get("correct_label"),
                "pred_label": decision.get("label"),
                "pred_label_missing": decision.get("label") is None,
                "action_correct": action_correct,
                "label_correct": label_correct,
                "exact_repair": exact_repair,
                "evidence_supported": evidence_supported,
                "case_success": action_correct and label_correct and exact_repair and evidence_supported,
                "committed": committed,
                "unsafe_commit": unsafe_commit,
                "decisive_evidence_acquired": acquired,
                "evidence_required": evidence_required,
                "calls": len(called),
                "observed_evidence_n": len(observed),
                "nonminimal_calls": nonminimal_calls,
                "parse_success": bool(pred.get("parse_success", True)),
                "predicted_intervention": decision.get("action") != "accept_provisionally",
                "relevant_tool_precision": relevant_tool_precision,
                "unavailable_tool_calls": unavailable_calls,
                "duplicate_tool_events": statuses.count("duplicate_tool"),
                "gate_rejections": statuses.count("rejected"),
                "invalid_json_events": statuses.count("invalid_json"),
                "budget_exhaustion_events": statuses.count("budget_exhausted"),
                "model_path": pred.get("model_path"),
                "condition": pred.get("condition"),
                "run_seed": pred.get("run_seed"),
                "excluded_tools": json.dumps(pred.get("excluded_tools", [])),
                "unresolved_n": len(decision.get("unresolved") or []),
                "inference_seconds": float(pred.get("inference_seconds", 0.0)),
                "input_tokens": int(pred.get("input_tokens", 0)),
                "output_tokens": int(pred.get("output_tokens", 0)),
            }
        )
    frame = pd.DataFrame(rows)
    summary = summarize_metrics(frame)
    return frame, summary


def _group_summary(part: pd.DataFrame) -> dict[str, Any]:
    accepted = part[part["committed"]]
    required = part[part["evidence_required"]]
    return {
        "n": int(len(part)),
        "case_success": float(part["case_success"].mean()),
        "action_accuracy": float(part["action_correct"].mean()),
        "label_accuracy": float(part["label_correct"].mean()),
        "exact_repair_accuracy": float(part["exact_repair"].mean()),
        "unsafe_commit_rate": float(part["unsafe_commit"].mean()),
        "coverage": float(part["committed"].mean()),
        "selective_risk": (
            float(accepted["unsafe_commit"].mean()) if len(accepted) else None
        ),
        "decisive_evidence_rate": float(part["decisive_evidence_acquired"].mean()),
        "decisive_evidence_rate_required": (
            float(required["decisive_evidence_acquired"].mean())
            if len(required) else None
        ),
        "mean_calls": float(part["calls"].mean()),
        "mean_nonminimal_calls": float(part["nonminimal_calls"].mean()),
        "mean_inference_seconds": float(part["inference_seconds"].mean()),
        "mean_input_tokens": float(part["input_tokens"].mean()),
        "mean_output_tokens": float(part["output_tokens"].mean()),
    }


def summarize_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    summary = _group_summary(frame)
    action_macro_f1 = f1_score(
        _label_series(frame["gold_action"]),
        _label_series(frame["pred_action"]),
        average="macro",
        zero_division=0,
    )
    label_rows = frame.dropna(subset=["gold_label"])
    controlled_label_macro_f1 = (
        f1_score(
            _label_series(label_rows["gold_label"]),
            _label_series(label_rows["pred_label"]),
            average="macro",
            zero_division=0,
        )
        if len(label_rows)
        else np.nan
    )
    detection = frame[frame["task"] != "T-GOV"]
    detect_precision, detect_recall, detect_f1, _ = precision_recall_fscore_support(
        detection["gold_corruption"].astype(bool),
        detection["predicted_intervention"].astype(bool),
        average="binary",
        zero_division=0,
    )
    recoverable = frame[frame["gold_action"] == "recover"]
    trajectory_columns = [
        "unavailable_tool_calls",
        "duplicate_tool_events",
        "gate_rejections",
        "invalid_json_events",
        "budget_exhaustion_events",
    ]
    summary.update({
        "action_macro_f1": float(action_macro_f1),
        "controlled_label_macro_f1": float(controlled_label_macro_f1),
        "intervention_detection_precision": float(detect_precision),
        "intervention_detection_recall": float(detect_recall),
        "intervention_detection_f1": float(detect_f1),
        "recovery_accuracy": (
            float(recoverable["case_success"].mean()) if len(recoverable) else None
        ),
        "mean_relevant_tool_precision": (
            float(frame["relevant_tool_precision"].mean(skipna=True))
            if frame["relevant_tool_precision"].notna().any() else None
        ),
        **{
            f"mean_{column}": float(frame[column].mean())
            for column in trajectory_columns
        },
        "abstention_rate": float((frame["pred_action"] == "abstain").mean()),
        "escalation_rate": float((frame["pred_action"] == "escalate").mean()),
        "parse_success_rate": float(frame["parse_success"].mean()),
        "by_task": {
            str(key): _group_summary(part)
            for key, part in frame.groupby("task")
        },
        "by_study": {
            str(key): _group_summary(part)
            for key, part in frame.groupby("study")
        },
        "by_error_family": {
            str(key): _group_summary(part)
            for key, part in frame.groupby("error_family")
        },
        "by_oracle_status": {
            str(key): _group_summary(part)
            for key, part in frame.groupby("oracle_status")
        },
    })
    return summary


def write_evaluation(
    frame: pd.DataFrame, summary: dict[str, Any], output_dir: str | Path
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "case_metrics.csv", index=False)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
