from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bundle import (
    DEFAULT_SOURCE_SPECS,
    _canonical_diagnosis,
    _canonical_sex,
    _find_imaging_field,
    _output_number,
    _source_records,
    masked_subject_key,
)
from .matlab import load_sfnc, load_subject_bridge
from .paths import public_locator
from .privacy import normalized_identifier
from .source_io import read_table, sha256_file


ALLOWED_ACTIONS = {"accept_provisionally", "recover"}
TARGET_FIELDS = ("age", "sex", "diagnosis", "panss_total", "cminds_composite")


def _json_candidates(raw_outputs: list[Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for raw in raw_outputs:
        if not isinstance(raw, str):
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("type") in {"candidate", "decision"}:
            if value.get("label"):
                candidates.append(value)
    return candidates


def _decision(row: dict[str, Any], mode: str) -> tuple[dict[str, Any] | None, str]:
    terminal = row.get("decision") or {}
    if mode == "gated":
        if (
            row.get("parse_success") is True
            and terminal.get("action") in ALLOWED_ACTIONS
            and terminal.get("label")
            and not terminal.get("unresolved")
        ):
            return terminal, "terminal_gated"
        return None, "terminal_blocked"

    candidates = _json_candidates(row.get("raw_outputs", []))
    if candidates:
        return candidates[-1], "raw_candidate_ungated"
    if terminal.get("label"):
        return terminal, "terminal_fallback_ungated"
    return None, "no_candidate"


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in map(json.loads, path.open())}


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in map(json.loads, path.open())}


def derive_mappings(
    cases: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    spec: Any,
    mode: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    requested = {"subject_identifier": spec.source_id_column}
    requested.update(
        {
            "current_age": spec.source_columns.get("age"),
            "sex_or_gender": spec.source_columns.get("sex"),
            "diagnosis": spec.source_columns.get("diagnosis"),
            "panss_total": spec.source_columns.get("panss_total"),
            "cognitive_domain": spec.source_columns.get("cminds_composite"),
        }
    )
    requested = {target: column for target, column in requested.items() if column}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ledger: list[dict[str, Any]] = []

    for target, expected_column in requested.items():
        case_id = None
        case = None
        for possible_case in cases.values():
            if possible_case.get("task") != "T-COL":
                continue
            if possible_case.get("base_artifact") != f"{spec.source_artifact}::{expected_column}":
                continue
            if possible_case.get("error_family") == "clean_concept":
                case_id = possible_case["case_id"]
                case = possible_case
                break

        prediction = predictions.get(case_id) if case_id else None
        selected, decision_source = _decision(prediction, mode) if prediction else (None, "missing_prediction")
        observed_label = selected.get("label") if selected else None
        action = selected.get("action") if selected else None
        row = {
            "cohort": spec.study,
            "source_artifact": spec.source_artifact,
            "source_column_observed": expected_column,
            "target_label_requested": target,
            "case_id": case_id or "",
            "model_action": action or "",
            "model_label": observed_label or "",
            "decision_source": decision_source,
            "mapping_status": "candidate" if observed_label else "blocked",
            "block_reason": "" if observed_label else decision_source,
        }
        if observed_label:
            candidates[observed_label].append(row)
        ledger.append(row)

    mappings: dict[str, str] = {}
    for target, expected_column in requested.items():
        rows = candidates.get(target, [])
        if len(rows) == 1:
            mappings[target] = rows[0]["source_column_observed"]
            rows[0]["mapping_status"] = "authorized_candidate" if mode == "gated" else "ungated_candidate"
        elif len(rows) > 1:
            for row in rows:
                row["mapping_status"] = "blocked_ambiguous_target"
                row["block_reason"] = f"{len(rows)} columns received label {target}"
        else:
            for row in ledger:
                if row["target_label_requested"] == target:
                    row["mapping_status"] = "blocked_label_mismatch_or_missing"
                    row["block_reason"] = f"no unique candidate labeled {target}"

    return mappings, ledger


def _source_canonical(value: Any, target: str, study: str) -> Any:
    if target == "current_age":
        return _output_number(value)
    if target == "sex_or_gender":
        return _canonical_sex(value, study)
    if target == "diagnosis":
        return _canonical_diagnosis(value, study)
    if target in {"panss_total", "cognitive_domain"}:
        return _output_number(value)
    raise ValueError(target)


def materialize(
    output_dir: Path,
    predictions_path: Path,
    cases_path: Path,
    salt: str,
    mode: str,
    model_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(cases_path)
    predictions = load_predictions(predictions_path)
    all_rows: list[dict[str, Any]] = []
    all_unresolved: list[dict[str, Any]] = []
    all_ledger: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for spec in DEFAULT_SOURCE_SPECS:
        mappings, ledger = derive_mappings(cases, predictions, spec, mode)
        all_ledger.extend(ledger)
        source = read_table(spec.source_path)
        source_records = _source_records(source, mappings.get("subject_identifier", "")) if mappings.get("subject_identifier") else {}
        bridge = load_subject_bridge(spec.bridge_path, spec.study)
        imaging = load_sfnc(spec.sfnc_path, spec.study)
        if list(bridge.analysis_ids) != list(imaging.analysis_ids):
            raise ValueError(f"{spec.study}: bridge/sFNC order is not exact")

        fields = imaging.field_names
        score_index = {field: index for index, field in enumerate(fields)}
        imaging_fields = {concept: _find_imaging_field(fields, concept) for concept in ("age", "sex", "diagnosis", "site")}
        site_values = imaging.scores[score_index[imaging_fields["site"]]] if imaging_fields["site"] else None
        matched = duplicate = 0
        field_counts = {
            "current_age": 0,
            "sex_or_gender": 0,
            "diagnosis": 0,
            "panss_total": 0,
            "cognitive_domain": 0,
        }
        for position, raw_identifier in enumerate(imaging.analysis_ids):
            key = normalized_identifier(raw_identifier)
            matches = source_records.get(key, [])
            record = matches[0] if len(matches) == 1 else None
            if len(matches) == 1:
                matched += 1
            elif len(matches) > 1:
                duplicate += 1
            row: dict[str, Any] = {
                "cohort": spec.study,
                "subject_key": masked_subject_key(spec.study, raw_identifier, salt),
                "age": "",
                "sex": "",
                "diagnosis": "",
                "site": _output_number(site_values[position]) if site_values is not None else "",
                "bridge_status": "exact_id_order",
                "sfnc_available": "yes" if bool(np.isfinite(imaging.sfnc[:, :, position]).all()) else "no",
                "agent_source_artifact": spec.source_artifact,
                "agent_join_status": "matched_unique" if record is not None else ("duplicate_key" if len(matches) > 1 else "not_in_source"),
                "analysis_reference_age": _output_number(imaging.scores[score_index[imaging_fields["age"]]][position]),
                "analysis_reference_sex": _canonical_sex(imaging.scores[score_index[imaging_fields["sex"]]][position], spec.study),
                "analysis_reference_diagnosis": _canonical_diagnosis(imaging.scores[score_index[imaging_fields["diagnosis"]]][position], spec.study),
                "source_age": "",
                "source_sex": "",
                "source_diagnosis": "",
                "source_panss_total": "",
                "source_cminds_composite": "",
            }
            for target, source_column in mappings.items():
                if target == "subject_identifier" or record is None:
                    continue
                value = _source_canonical(record[source_column], target, spec.study)
                output_field = {
                    "current_age": "age",
                    "sex_or_gender": "sex",
                    "diagnosis": "diagnosis",
                    "panss_total": "source_panss_total",
                    "cognitive_domain": "source_cminds_composite",
                }[target]
                row[output_field] = value
                if value != "":
                    field_counts[target] += 1
            all_rows.append(row)

        blocked = [entry for entry in ledger if entry["mapping_status"].startswith("blocked")]
        for entry in blocked:
            all_unresolved.append(
                {
                    "issue_id": f"{spec.study.lower()}_{entry['target_label_requested']}_mapping",
                    "cohort": spec.study,
                    "scope": entry["target_label_requested"],
                    "affected_n": len(imaging.analysis_ids),
                    "issue_type": "agent_mapping_blocked",
                    "source_artifact": spec.source_artifact,
                    "source_column": entry["source_column_observed"],
                    "case_id": entry["case_id"],
                    "decision_source": entry["decision_source"],
                    "severity": "high" if entry["target_label_requested"] in {"subject_identifier", "diagnosis"} else "medium",
                    "action": "human_review_or_add_evidence",
                    "reason": entry["block_reason"],
                }
            )
        if not mappings.get("subject_identifier"):
            all_unresolved.append(
                {
                    "issue_id": f"{spec.study.lower()}_subject_identifier_join",
                    "cohort": spec.study,
                    "scope": "subject_identifier",
                    "affected_n": len(imaging.analysis_ids),
                    "issue_type": "agent_mapping_blocked",
                    "source_artifact": spec.source_artifact,
                    "source_column": "",
                    "case_id": "",
                    "decision_source": "",
                    "severity": "high",
                    "action": "human_review_or_add_evidence",
                    "reason": "No authorized unique subject_identifier mapping; source values cannot be joined.",
                }
            )
        summaries.append(
            {
                "cohort": spec.study,
                "n_subjects": len(imaging.analysis_ids),
                "source_rows": len(source),
                "source_matched_n": matched,
                "source_duplicate_n": duplicate,
                "mapping_count": len(mappings),
                "mapping_targets": sorted(mappings),
                "age_coverage_n": field_counts["current_age"],
                "sex_coverage_n": field_counts["sex_or_gender"],
                "diagnosis_coverage_n": field_counts["diagnosis"],
                "model": model_name,
                "mode": mode,
            }
        )

    columns = [
        "cohort", "subject_key", "age", "sex", "diagnosis", "site",
        "bridge_status", "sfnc_available", "agent_source_artifact",
        "agent_join_status", "analysis_reference_age", "analysis_reference_sex",
        "analysis_reference_diagnosis", "source_age", "source_sex",
        "source_diagnosis", "source_panss_total", "source_cminds_composite",
    ]
    pd.DataFrame(all_rows, columns=columns).to_csv(output_dir / "canonical_subject_bundle.csv", index=False)
    pd.DataFrame(all_ledger).to_csv(output_dir / "agent_mapping_ledger.csv", index=False)
    unresolved_columns = [
        "issue_id", "cohort", "scope", "affected_n", "issue_type", "source_artifact",
        "source_column", "case_id", "decision_source", "severity", "action", "reason",
    ]
    pd.DataFrame(all_unresolved, columns=unresolved_columns).to_csv(output_dir / "unresolved.csv", index=False)
    summary = {
        "version": "agent_materialized_bundle_v1",
        "mode": mode,
        "model": model_name,
        "predictions": public_locator(str(predictions_path)),
        "oracle_read": False,
        "raw_subject_identifiers_serialized": False,
        "row_count": len(all_rows),
        "source_summaries": summaries,
        "mapping_ledger_count": len(all_ledger),
        "authorized_mapping_count": sum(x["mapping_status"] in {"authorized_candidate", "ungated_candidate"} for x in all_ledger),
        "blocked_mapping_count": sum(x["mapping_status"].startswith("blocked") for x in all_ledger),
        "unresolved_issue_count": len(all_unresolved),
        "scope": "Agent-derived candidate bundle. It is not a validated scientific gold standard.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Agent-Materialized Bundle v1\n\n"
        "This read-only derived bundle consumes agent column decisions. The "
        "analysis-reference fields are retained for comparison only; canonical "
        "age, sex, and diagnosis are populated from an agent-selected source "
        "column only when the mapping is available. Blocked mappings remain "
        "unresolved and are never silently replaced by the deterministic bundle.\n\n"
        f"Mode: `{mode}`. Model: `{model_name}`. Oracle read: `false`.\n",
        encoding="utf-8",
    )
    return summary
