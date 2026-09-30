from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .environment import CaseEnvironment


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tokens(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def infer_file_role(name: str, structure: dict[str, Any] | None = None) -> str:
    token = _tokens(name)
    if "dictionary" in token or "codebook" in token:
        return "dictionary_or_codebook"
    sheet_names = []
    column_names = []
    if structure:
        for sheet in structure.get("sheets", []):
            sheet_names.append(_tokens(sheet.get("sheet_name", "")))
            column_names.extend(_tokens(x) for x in sheet.get("column_names", []))
        column_names.extend(
            _tokens(x) for x in structure.get("column_names", [])
        )
    joined = " ".join([token, *sheet_names, *column_names[:120]])
    role_hits = {
        "dictionary_or_codebook": any(
            x in joined for x in ("dictionary", "codebook", "valid values", "field name")
        ),
        "medication": any(x in joined for x in (" medication ", " med ", "cpz", "antipsychotic", "dose")),
        "clinical_scale": any(x in joined for x in ("panss", "delusions", "clinical", "calgary")),
        "cognitive": any(x in joined for x in ("cognitive", "working memory", "wasi", "wtar", "cminds")),
        "diagnosis_or_labels": any(x in joined for x in ("diagnosis", " dx ", "subject type")),
        "demographics": any(x in joined for x in ("demographic", " age ", " gender ", " sex ")),
    }
    substantive = [role for role, hit in role_hits.items() if hit]
    non_dictionary = [x for x in substantive if x != "dictionary_or_codebook"]
    if role_hits["dictionary_or_codebook"] and not non_dictionary:
        return "dictionary_or_codebook"
    if len(set(non_dictionary)) >= 2 or len(sheet_names) >= 3:
        return "mixed"
    if non_dictionary:
        return non_dictionary[0]
    if any(x in joined for x in ("subjectid", "subject id", "subjid", "ursi")):
        return "subject_list"
    return "unknown"


def infer_column_concept(name: str) -> str:
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    spaced = _tokens(name)
    if any(x in compact for x in ("subjectid", "subjid", "ursi", "anonymizedid")) or compact in {"id", "doi", "usrd"}:
        return "subject_identifier"
    if "siteid" in compact or compact == "site":
        return "site_identifier"
    if "visitid" in compact or compact == "visit":
        return "visit_identifier"
    if "experimentid" in compact:
        return "experiment_identifier"
    if any(x in compact for x in ("assessmentid", "asmtid", "segmentid", "instrumentid")):
        return "assessment_identifier"
    if any(x in compact for x in ("diagnos",)) or compact == "dx":
        return "diagnosis"
    if "gender" in compact or compact == "sex" or "sex" in spaced.split():
        return "sex_or_gender"
    if "ageonset" in compact or compact in {"ao"} or "ageatfirst" in compact:
        return "age_at_onset"
    if "age" in compact:
        return "current_age"
    if "panss" in compact and any(x in compact for x in ("tot", "total", "gentotal")):
        return "panss_total"
    if "panss" in compact and ("pos" in compact or "positive" in compact):
        return "panss_positive"
    if "panss" in compact and ("neg" in compact or "negative" in compact):
        return "panss_negative"
    panss_items = {
        "delusions", "hallucinatorybehavior", "bluntedaffect", "poorattention",
        "conceptualdisorganization",
    }
    if compact in panss_items or "npanss" in compact:
        return "panss_item"
    if any(x in compact for x in ("drug", "medication")):
        return "medication_name"
    if "dose" in compact or compact.endswith("mg"):
        return "medication_dose"
    if compact in {"cpz", "cpz1", "cpz2"}:
        return "cpz_candidate"
    cognitive = (
        "cognitive", "workingmemory", "speedofprocessing", "attentionvigilance",
        "cmindscomposite", "wtar", "wasi", "iq",
    )
    if any(x in compact for x in cognitive):
        return "cognitive_domain"
    if compact in {"positive"}:
        return "panss_positive"
    if compact in {"negative"}:
        return "panss_negative"
    return "unknown_or_study_specific"


def _decision(
    action: str,
    label: str | None,
    repair: dict[str, Any] | None,
    evidence_tools: list[str],
    unresolved: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "label": label,
        "repair": repair,
        "evidence_tools": evidence_tools,
        "unresolved": unresolved or [],
    }


def deterministic_gate(case: dict[str, Any], budget: int = 6) -> dict[str, Any]:
    env = CaseEnvironment(case, budget)
    obs = case["initial_observation"]
    called: list[str] = []

    if case["task"] == "T-FILE":
        result = env.call("reparse_artifact")
        called.append("reparse_artifact")
        evidence = result.get("evidence", {})
        label = infer_file_role(
            obs.get("artifact_name", ""), evidence.get("structure", {})
        )
        proposed = obs.get("proposed_primary_role")
        action = "accept_provisionally" if label == proposed else "recover"
        repair = None if action == "accept_provisionally" else {"primary_role": label}
        if action == "recover" and obs.get("parser_status") == "injected_parser_failure":
            repair = {"parser_status": "parsed", "primary_role": label}
        if evidence.get("status") != "parsed":
            action, repair = "abstain", None

    elif case["task"] == "T-COL":
        label = infer_column_concept(obs.get("column_name", ""))
        proposed = obs.get("proposed_concept")
        if label in {"sex_or_gender", "diagnosis", "current_age", "age_at_onset", "medication_dose"}:
            env.call("profile_values")
            called.append("profile_values")
        if label == "unknown_or_study_specific" and "profile_values" in case["tools"]:
            env.call("profile_values")
            called.append("profile_values")
        action = "accept_provisionally" if label == proposed else "recover"
        repair = None if action == "accept_provisionally" else {"concept": label}

    elif case["task"] == "T-VAL":
        label = "value_encoding"
        verifier = next(
            (
                name
                for name in ("get_dictionary_entry", "compare_redundant_values")
                if name in case["tools"]
            ),
            None,
        )
        if verifier is None:
            env.call("profile_values")
            called.append("profile_values")
            action, repair = "abstain", None
        else:
            result = env.call(verifier)
            called.append(verifier)
            verified = result["evidence"]["mapping"]
            proposed = obs.get("proposed_mapping")
            action = "accept_provisionally" if proposed == verified else "recover"
            repair = None if action == "accept_provisionally" else {"mapping": verified}

    elif case["task"] == "T-ID":
        scope = obs.get("id_scope", "imaging_row_alignment")
        label = scope
        if scope == "imaging_row_alignment":
            result = env.call("check_imaging_order")
            called.append("check_imaging_order")
            aligned = bool(result["evidence"]["order_agreement"])
            action = "accept_provisionally" if aligned else "recover"
            repair = None if aligned else {"operation": "restore_analysis_id_order"}
        elif scope == "identifier_linkage":
            result = env.call("check_id_overlap")
            called.append("check_id_overlap")
            evidence = result["evidence"]
            verified = evidence["verified_relationship"]
            valid = (
                obs.get("proposed_relationship") == verified
                and obs.get("proposed_identity_proven") is False
            )
            action = "accept_provisionally" if valid else "recover"
            repair = None if valid else {
                "identity_proven": False,
                "relationship": verified,
            }
        else:
            result = env.call("check_lineage")
            called.append("check_lineage")
            verified = result["evidence"]["verified_independence_status"]
            valid = obs.get("proposed_independence_status") == verified
            action = "accept_provisionally" if valid else "recover"
            repair = None if valid else {"independence_status": verified}

    else:
        label = obs.get("governance_scope")
        env.call("escalate_to_human")
        called.append("escalate_to_human")
        action, repair = "escalate", None

    decision = _decision(action, label, repair, called)
    env.terminate(**decision)
    return {
        "case_id": case["case_id"],
        "decision": decision,
        "called_tools": called,
        "parse_success": True,
        "trajectory": env.trajectory.steps,
    }


def run_deterministic_reference(
    cases_path: str | Path, output_path: str | Path, budget: int = 6
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for case in _read_jsonl(cases_path):
            handle.write(json.dumps(deterministic_gate(case, budget), sort_keys=True) + "\n")
