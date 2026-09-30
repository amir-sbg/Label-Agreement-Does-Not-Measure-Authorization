from __future__ import annotations

from typing import Any


FILE_LABELS = [
    "subject_list", "demographics", "diagnosis_or_labels", "clinical_scale",
    "cognitive", "medication", "dictionary_or_codebook",
    "scanner_or_site_metadata", "visit_metadata", "imaging_derivative",
    "mixed", "ambiguous", "unknown",
]
COLUMN_LABELS = [
    "subject_identifier", "site_identifier", "visit_identifier",
    "experiment_identifier", "assessment_identifier", "diagnosis",
    "sex_or_gender", "current_age", "age_at_onset", "panss_item",
    "panss_positive", "panss_negative", "panss_total", "medication_name",
    "medication_dose", "cpz_candidate", "cognitive_domain",
    "imaging_feature", "unknown_or_study_specific",
]
ACTIONS = {"recover", "accept_provisionally", "abstain", "escalate"}


def normalize_decision(
    case: dict[str, Any], parsed: dict[str, Any], called: list[str]
) -> dict[str, Any]:
    task = case["task"]
    action = parsed.get("action")
    if action not in ACTIONS:
        action = "abstain"
    label = parsed.get("label")
    if task == "T-VAL":
        label = "value_encoding"
    elif task == "T-ID":
        label = case["initial_observation"].get("id_scope", "imaging_row_alignment")
    elif task == "T-GOV":
        label = case["initial_observation"].get("governance_scope")
    repair = parsed.get("repair")
    if action != "recover":
        repair = None
    elif task == "T-FILE":
        repair = dict(repair) if isinstance(repair, dict) else {}
        if label:
            repair["primary_role"] = label
        if (
            case["initial_observation"].get("parser_status") == "injected_parser_failure"
            and "reparse_artifact" in called
        ):
            repair["parser_status"] = "parsed"
    elif task == "T-COL":
        repair = {"concept": label}
    elif task == "T-ID":
        scope = case["initial_observation"].get("id_scope", "imaging_row_alignment")
        if scope == "imaging_row_alignment":
            repair = {"operation": "restore_analysis_id_order"}
        elif scope == "identifier_linkage" and "check_id_overlap" in called:
            evidence = case["tools"]["check_id_overlap"]
            repair = {
                "identity_proven": False,
                "relationship": evidence["verified_relationship"],
            }
        elif scope == "source_independence" and "check_lineage" in called:
            evidence = case["tools"]["check_lineage"]
            repair = {
                "independence_status": evidence["verified_independence_status"]
            }
    unresolved = parsed.get("unresolved")
    if not isinstance(unresolved, list):
        unresolved = [str(unresolved)] if unresolved else []
    return {
        "action": action,
        "label": label,
        "repair": repair,
        "evidence_tools": list(called),
        "unresolved": unresolved,
    }


def gate_violation(case: dict[str, Any], decision: dict[str, Any], called: list[str]) -> str | None:
    action = decision["action"]
    committed = action in {"recover", "accept_provisionally"}
    task = case["task"]
    if task == "T-GOV":
        if action != "escalate":
            return "Governance scope requires escalation."
        if "escalate_to_human" not in called:
            return "Call escalate_to_human before escalation."
    if not committed:
        return None
    if task == "T-FILE" and "reparse_artifact" not in called:
        return "A file-role commit requires verified parsing and structure."
    if task == "T-COL" and decision["label"] in {
        "sex_or_gender", "diagnosis", "current_age", "age_at_onset", "medication_dose"
    } and "profile_values" not in called:
        return "This column-concept commit requires an aggregate value profile."
    if task == "T-VAL" and not ({"get_dictionary_entry", "compare_redundant_values"} & set(called)):
        return "A value-encoding commit requires dictionary or redundant-source evidence."
    if task == "T-ID":
        scope = case["initial_observation"].get("id_scope", "imaging_row_alignment")
        required_tool = {
            "imaging_row_alignment": "check_imaging_order",
            "identifier_linkage": "check_id_overlap",
            "source_independence": "check_lineage",
        }[scope]
        if required_tool not in called:
            return f"A {scope} commit requires {required_tool}."
    return None


def fallback_decision(case: dict[str, Any], called: list[str], reason: str) -> dict[str, Any]:
    if case["task"] == "T-GOV":
        return {
            "action": "escalate", "label": case["initial_observation"].get("governance_scope"),
            "repair": None, "evidence_tools": called, "unresolved": [reason],
        }
    if case["task"] == "T-VAL":
        label = "value_encoding"
    elif case["task"] == "T-ID":
        label = case["initial_observation"].get("id_scope", "imaging_row_alignment")
    else:
        label = None
    return {
        "action": "abstain", "label": label, "repair": None,
        "evidence_tools": called, "unresolved": [reason],
    }
