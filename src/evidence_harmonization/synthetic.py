from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path


FILE_ROLES = [
    "subject_list", "demographics", "diagnosis_or_labels", "clinical_scale",
    "cognitive", "medication", "dictionary_or_codebook", "mixed",
]
COLUMN_CONCEPTS = [
    "subject_identifier", "site_identifier", "visit_identifier", "diagnosis",
    "sex_or_gender", "current_age", "age_at_onset", "panss_total",
    "medication_name", "medication_dose", "cpz_candidate", "cognitive_domain",
]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def base_case(
    case_id: str,
    study: str,
    task: str,
    family: str,
    seed: int,
    initial: dict,
    views: dict,
    tools: dict,
    oracle: dict,
) -> tuple[dict, dict]:
    case = {
        "base_artifact": family,
        "case_id": case_id,
        "error_family": oracle["error_family"],
        "evidence_views": views,
        "initial_observation": initial,
        "lineage_group": f"synthetic_{family}",
        "perturbation_seed": seed,
        "study": study,
        "task": task,
        "tools": tools,
    }
    oracle_row = {
        "base_artifact": family,
        "case_id": case_id,
        "error_family": oracle["error_family"],
        "lineage_group": f"synthetic_{family}",
        "oracle": {k: v for k, v in oracle.items() if k != "error_family"},
        "study": study,
        "task": task,
    }
    return case, oracle_row


def build(output: Path, n_pairs: int, seed: int) -> None:
    rng = random.Random(seed)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    subjects = [[f"SYN_A_{i:04d}", 20 + (i * 7) % 45, 1 + i % 2, "A"] for i in range(1, 61)]
    write_csv(tables / "site_a_demographics.csv", ["participant_code", "age_years", "sex_code", "site"], subjects)
    clinical = [[row[0], 1 + i % 3, i % 2, 8 + (i * 3) % 28, round(50 + (i * 2.5), 1)] for i, row in enumerate(subjects)]
    write_csv(tables / "site_a_clinical.csv", ["subj_id", "visit", "dx_code", "panss_total", "cpz_mg_day"], clinical)
    dictionary = [["dx_code", "0", "control"], ["dx_code", "1", "schizophrenia"], ["sex_code", "1", "male"], ["sex_code", "2", "female"]]
    write_csv(tables / "site_a_dictionary.csv", ["variable", "code", "meaning"], dictionary)
    imaging = [[row[0], i, f"{0.1 + i / 100:.3f}", f"{0.2 + (i % 11) / 100:.3f}"] for i, row in enumerate(subjects)]
    write_csv(tables / "site_a_sfnc_summary.csv", ["participant_code", "row_index", "edge_001", "edge_002"], imaging)
    mixed = [[row[0], row[1], row[2], clinical[i][3], clinical[i][4]] for i, row in enumerate(subjects)]
    write_csv(tables / "site_a_mixed_export.csv", ["ID", "Age", "Sex", "PANSS_total", "CPZ"], mixed)

    cases: list[dict] = []
    oracle: list[dict] = []

    for i in range(n_pairs):
        role = FILE_ROLES[i % len(FILE_ROLES)]
        family = f"file_{i:02d}"
        artifact = f"synthetic_{role}_{i:02d}.csv"
        columns = {
            "subject_list": ["participant_code"],
            "demographics": ["participant_code", "age_years", "sex_code"],
            "diagnosis_or_labels": ["subj_id", "dx_code"],
            "clinical_scale": ["subj_id", "panss_total"],
            "cognitive": ["subj_id", "WTAR_standard_score"],
            "medication": ["subj_id", "medication_name", "cpz_mg_day"],
            "dictionary_or_codebook": ["variable", "code", "meaning"],
            "mixed": ["ID", "Age", "Sex", "PANSS_total", "CPZ"],
        }[role]
        structure = {"column_names": columns, "columns": len(columns), "rows": 60}
        for corrupted in (False, True):
            proposed = "demographics" if corrupted else role
            case_id = digest(f"{family}|file|{corrupted}")
            initial = {"artifact_name": artifact, "path_context": f"SYNTHETIC/{artifact}", "parser_status": "not_yet_requested", "proposed_primary_role": proposed}
            views = {"names_only": {"artifact_name": artifact, "proposed_primary_role": proposed}, "context": {**initial, "component_hints": [role]}, "structure": {**initial, "structure": structure}, "full": {**initial, "structure": structure, "validity_status": "structure_verified"}}
            tools = {"reparse_artifact": {"status": "parsed", "artifact_id": family, "kind": "table", "structure": structure, "parser_provenance": "synthetic_parser"}}
            c, o = base_case(case_id, "SYNTH-A", "T-FILE", family, i, initial, views, tools, {"error_family": "role_misclassification" if corrupted else "clean_role", "correct_action": "recover" if corrupted else "accept_provisionally", "correct_label": role, "correct_repair": {"primary_role": role} if corrupted else None, "corruption_present": corrupted, "evidence_basis": "synthetic schema with exact role assignment", "minimal_decisive_tool_sets": [["reparse_artifact"]], "oracle_status": "synthetic_exact"})
            cases.append(c); oracle.append(o)

    for i in range(n_pairs * 2):
        concept = COLUMN_CONCEPTS[i % len(COLUMN_CONCEPTS)]
        family = f"column_{i:02d}"
        artifact = "site_a_mixed_export.csv" if i % 2 else "site_a_clinical.csv"
        names = {"subject_identifier": "subj_id", "site_identifier": "site", "visit_identifier": "visit", "diagnosis": "dx_code", "sex_or_gender": "sex_code", "current_age": "age_years", "age_at_onset": "age_onset", "panss_total": "PANSS_total", "medication_name": "medication_name", "medication_dose": "dose_mg_day", "cpz_candidate": "CPZ", "cognitive_domain": "WTAR_standard_score"}
        col = names[concept]
        for corrupted in (False, True):
            proposed = "diagnosis" if corrupted else concept
            case_id = digest(f"{family}|column|{corrupted}")
            signature = {"dtype": "int64", "n": 60, "missing_fraction": 0.0, "unique_n": 2 if concept in {"diagnosis", "sex_or_gender"} else 60, "numeric_fraction": 1.0}
            initial = {"artifact_name": artifact, "column_name": col, "parent_primary_role": "mixed", "parser_status": "parsed", "proposed_concept": proposed}
            views = {"names_only": {"artifact_name": artifact, "column_name": col, "proposed_concept": proposed}, "context": {**initial, "neighboring_columns": ["ID", "Age", "Sex", "PANSS_total", "CPZ"]}, "structure": {**initial, "value_signature": signature}, "value_signature": {**initial, "value_signature": signature}, "full": {**initial, "value_signature": signature, "dictionary_hint": "available_via_tool"}}
            tools = {"profile_values": {"column_name": col, "signature": signature, "status": "verified"}}
            if concept in {"diagnosis", "sex_or_gender", "age_at_onset", "medication_dose", "cpz_candidate"}:
                tools["get_dictionary_entry"] = {"status": "verified", "mapping": {"0": "control", "1": "schizophrenia"} if concept == "diagnosis" else {"1": "male", "2": "female"} if concept == "sex_or_gender" else {"unit": "source_specific"}}
            c, o = base_case(case_id, "SYNTH-A", "T-COL", family, i, initial, views, tools, {"error_family": "column_misclassification" if corrupted else "clean_column", "correct_action": "recover" if corrupted else "accept_provisionally", "correct_label": concept, "correct_repair": {"concept": concept} if corrupted else None, "corruption_present": corrupted, "evidence_basis": "synthetic column schema and value signature", "minimal_decisive_tool_sets": [["profile_values"]], "oracle_status": "synthetic_exact"})
            cases.append(c); oracle.append(o)

    for i in range(max(4, n_pairs // 2)):
        family = f"value_{i:02d}"
        corrupted = i % 2 == 1
        mapping = {"0": "control", "1": "schizophrenia"} if not corrupted else {"0": "schizophrenia", "1": "control"}
        case_id = digest(f"{family}|value|{corrupted}")
        initial = {"artifact_name": "site_a_dictionary.csv", "column_name": "dx_code", "proposed_mapping": mapping, "reference_status": "not_yet_requested"}
        signature = {"category_counts": {"0": 30, "1": 30}, "dtype": "int64", "unique_n": 2, "missing_fraction": 0.0}
        views = {"names_only": initial, "value_signature": {**initial, "value_signature": signature}, "reference": initial, "full": {**initial, "value_signature": signature}}
        tools = {"profile_values": {"status": "verified", "signature": signature}, "get_dictionary_entry": {"status": "verified", "mapping": {"0": "control", "1": "schizophrenia"}, "source": "synthetic dictionary"}}
        c, o = base_case(case_id, "SYNTH-A", "T-VAL", family, i, initial, views, tools, {"error_family": "encoding_direction" if corrupted else "clean_encoding", "correct_action": "recover" if corrupted else "accept_provisionally", "correct_label": "value_encoding", "correct_repair": {"mapping": {"0": "control", "1": "schizophrenia"}} if corrupted else None, "corruption_present": corrupted, "evidence_basis": "synthetic dictionary with exact code direction", "minimal_decisive_tool_sets": [["get_dictionary_entry"]], "oracle_status": "synthetic_exact"})
        cases.append(c); oracle.append(o)

    for i in range(max(6, n_pairs // 2)):
        family = f"id_{i:02d}"
        scope = ["imaging_row_alignment", "identifier_linkage", "source_independence"][i % 3]
        corrupted = i % 2 == 1
        case_id = digest(f"{family}|id|{corrupted}")
        if scope == "imaging_row_alignment":
            initial = {"operation": "join_metadata_to_sfnc_by_current_row_order", "metadata_rows": 60, "imaging_subjects": 60, "claimed_alignment": not corrupted}
            views = {"names_only": initial, "context": {**initial, "study": "SYNTH-A"}, "overlap": {**initial, "order_check_hint": "available_via_tool"}, "full": {**initial, "order_check_hint": "available_via_tool"}}
            tools = {"check_imaging_order": {"order_agreement": not corrupted, "mismatch_fraction": 0.0 if not corrupted else 0.2, "mismatch_n": 0 if not corrupted else 12, "status": "verified"}}
            label, repair = "imaging_row_alignment", {"operation": "restore_analysis_id_order"}
        elif scope == "identifier_linkage":
            initial = {"left_id": "participant_code", "right_id": "subj_id", "overlap_fraction": 1.0, "claimed_relationship": "same_subject_key"}
            views = {"names_only": initial, "context": initial, "overlap": {**initial, "overlap_hint": "available_via_tool"}, "full": {**initial, "overlap_hint": "available_via_tool"}}
            tools = {"check_id_overlap": {"overlap_fraction": 1.0 if not corrupted else 0.2, "overlap_n": 60 if not corrupted else 12, "status": "verified", "relationship": "operational_overlap_only"}}
            label, repair = "identifier_linkage", {"identity_proven": False, "relationship": "operational_overlap_only"}
        else:
            initial = {"source_a": "site_a_demographics.csv", "source_b": "site_a_clinical.csv", "claimed_independence": not corrupted}
            views = {"names_only": initial, "context": initial, "full": {**initial, "lineage_hint": "available_via_tool"}}
            tools = {"check_lineage": {"independence_status": "independent" if not corrupted else "same_lineage_not_independent", "status": "verified"}}
            label, repair = "source_independence", {"independence_status": "independent"}
        c, o = base_case(case_id, "SYNTH-A", "T-ID", family, i, initial, views, tools, {"error_family": "id_or_integrity" if corrupted else "clean_id_or_integrity", "correct_action": "recover" if corrupted else "accept_provisionally", "correct_label": label, "correct_repair": repair if corrupted else None, "corruption_present": corrupted, "evidence_basis": "synthetic exact linkage/order/lineage relation", "minimal_decisive_tool_sets": [[next(iter(tools))]], "oracle_status": "synthetic_exact"})
        cases.append(c); oracle.append(o)

    for i in range(max(4, n_pairs // 2)):
        family = f"gov_{i:02d}"
        case_id = digest(f"{family}|gov")
        initial = {"artifact_name": "site_a_clinical.csv", "governance_scope": "diagnosis_inclusion_policy", "proposal": "collapse codes into a binary cohort"}
        views = {"names_only": {"proposal": initial["proposal"]}, "context": initial, "full": {**initial, "limitation": "synthetic data does not authorize clinical inclusion policy"}}
        tools = {"escalate_to_human": {"allowed": True, "required_scope": "diagnosis_inclusion_policy"}}
        c, o = base_case(case_id, "SYNTH-A", "T-GOV", family, i, initial, views, tools, {"error_family": "governance", "correct_action": "escalate", "correct_label": "diagnosis_inclusion_policy", "correct_repair": None, "corruption_present": False, "evidence_basis": "policy requires human authorization", "minimal_decisive_tool_sets": [["escalate_to_human"]], "oracle_status": "synthetic_exact"})
        cases.append(c); oracle.append(o)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "oracle.jsonl").open("w", encoding="utf-8") as handle:
        for row in oracle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"version": "synthetic_v1", "seed": seed, "n_cases": len(cases), "n_base_artifacts": len({x["base_artifact"] for x in cases}), "paired_counterfactual": True, "raw_tables": [str(p.relative_to(output)) for p in sorted(tables.glob("*.csv"))], "task_counts": {task: sum(1 for x in cases if x["task"] == task) for task in sorted({x["task"] for x in cases})}, "oracle_status": "synthetic_exact_operational_only"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# Synthetic harmonization benchmark v1\n\nThis benchmark contains synthetic identifiers and values only. Paired clean/corrupted cases have an exact operational oracle. It is used to test recovery, abstention, tool routing, and uncertainty instrumentation; it does not establish semantic validity on COBRE or FBIRN.\n", encoding="utf-8")
