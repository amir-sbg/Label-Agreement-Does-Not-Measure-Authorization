from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .cards import load_manifest
from .matlab import load_subject_bridge
from .privacy import normalized_identifier, stable_digest


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    study: str
    task: str
    error_family: str
    base_artifact: str
    lineage_group: str
    perturbation_seed: int
    initial_observation: dict[str, Any]
    tools: dict[str, dict[str, Any]]
    oracle: dict[str, Any]
    evidence_views: dict[str, dict[str, Any]]


_WRONG_ROLE = {
    "mixed": "demographics",
    "subject_list": "diagnosis_or_labels",
    "demographics": "subject_list",
    "diagnosis_or_labels": "demographics",
    "clinical_scale": "subject_list",
    "cognitive": "clinical_scale",
    "medication": "scanner_or_site_metadata",
    "dictionary_or_codebook": "subject_list",
}

_WRONG_CONCEPT = {
    "subject_identifier": "site_identifier",
    "site_identifier": "subject_identifier",
    "visit_identifier": "subject_identifier",
    "experiment_identifier": "subject_identifier",
    "assessment_identifier": "subject_identifier",
    "diagnosis": "sex_or_gender",
    "sex_or_gender": "diagnosis",
    "current_age": "age_at_onset",
    "age_at_onset": "current_age",
    "panss_item": "panss_total",
    "panss_positive": "panss_item",
    "panss_negative": "panss_item",
    "panss_total": "panss_item",
    "medication_name": "diagnosis",
    "medication_dose": "cpz_candidate",
    "cpz_candidate": "medication_dose",
    "cognitive_domain": "clinical_scale",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


_UNSPECIFIED_SHEET = object()


def _unique_rows(
    rows: list[dict[str, Any]], key_name: str, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        key = row.get(key_name)
        if not isinstance(key, str) or not key:
            raise RuntimeError(
                f"{source} row {row_index} has no valid {key_name}"
            )
        if key in indexed:
            raise RuntimeError(f"{source} contains duplicate {key_name}: {key}")
        indexed[key] = row
    return indexed


def _cards_by_id(
    cards_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str | None, str], Any],
    list[dict[str, Any]],
]:
    file_rows = _read_jsonl(cards_dir / "file_cards.jsonl")
    files = _unique_rows(file_rows, "artifact_id", "FileCards")
    columns: dict[tuple[str, str | None, str], Any] = {}
    for row_index, row in enumerate(
        _read_jsonl(cards_dir / "column_cards.jsonl")
    ):
        artifact_id = row.get("artifact_id")
        column_name = row.get("column_name")
        sheet_name = row.get("sheet_name")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise RuntimeError(
                f"ColumnCards row {row_index} has no valid artifact_id"
            )
        if not isinstance(column_name, str) or not column_name:
            raise RuntimeError(
                f"ColumnCards row {row_index} has no valid column_name"
            )
        if sheet_name is not None and not isinstance(sheet_name, str):
            raise RuntimeError(
                f"ColumnCards row {row_index} has an invalid sheet_name"
            )
        key = (artifact_id, sheet_name, column_name)
        if key in columns:
            raise RuntimeError(
                "ColumnCards contains duplicate evidence unit: "
                f"{artifact_id}::{sheet_name}::{column_name}"
            )
        columns[key] = row
    linkages = _read_jsonl(cards_dir / "linkage_cards.jsonl")
    _unique_rows(linkages, "card_id", "LinkageCards")
    return files, columns, linkages


def _resolve_column_card(
    columns: dict[tuple[str, str | None, str], Any],
    artifact_id: str,
    column_name: str,
    sheet_name: str | None | object = _UNSPECIFIED_SHEET,
) -> dict[str, Any]:
    if sheet_name is not _UNSPECIFIED_SHEET:
        key = (artifact_id, sheet_name, column_name)
        if key not in columns:
            raise KeyError(
                "Missing ColumnCard for "
                f"{artifact_id}::{sheet_name}::{column_name}"
            )
        return columns[key]

    matches = [
        row
        for (artifact, _, column), row in columns.items()
        if artifact == artifact_id and column == column_name
    ]
    if not matches:
        raise KeyError(f"Missing ColumnCard for {artifact_id}::{column_name}")
    if len(matches) > 1:
        sheets = sorted(str(row.get("sheet_name")) for row in matches)
        raise RuntimeError(
            "Ambiguous workbook column requires an explicit sheet: "
            f"{artifact_id}::{column_name}; candidates={sheets}"
        )
    return matches[0]


def _case_id(*parts: Any) -> str:
    return stable_digest("|".join(map(str, parts)), length=24)


def build_benchmark(
    source_manifest: str | Path,
    concept_manifest: str | Path,
    cards_dir: str | Path,
) -> list[BenchmarkCase]:
    source = load_manifest(source_manifest)
    with Path(concept_manifest).open("r", encoding="utf-8") as handle:
        concepts = yaml.safe_load(handle)
    files, columns, linkages = _cards_by_id(Path(cards_dir))
    specs = _unique_rows(source["artifacts"], "artifact_id", "source manifest")
    cases: list[BenchmarkCase] = []

    metadata_ids = [
        x["artifact_id"]
        for x in source["artifacts"]
        if x["kind"] in {"table", "workbook"}
    ]
    for artifact_id in metadata_ids:
        cases.extend(_file_cases(specs[artifact_id], files[artifact_id]))

    for entry in concepts["columns"]:
        if len(entry) == 4:
            artifact_id, column_name, concept, basis = entry
            sheet_name: str | None | object = _UNSPECIFIED_SHEET
        elif len(entry) == 5:
            artifact_id, sheet_name, column_name, concept, basis = entry
        else:
            raise RuntimeError(
                "Concept entries must be [artifact, column, concept, basis] "
                "or [artifact, sheet, column, concept, basis]"
            )
        card = _resolve_column_card(
            columns, str(artifact_id), str(column_name), sheet_name
        )
        cases.extend(
            _column_cases(specs[artifact_id], card, str(concept), str(basis))
        )

    cases.extend(_value_cases(files, columns, specs))
    cases.extend(_identifier_cases(specs))
    cases.extend(_metadata_linkage_cases(linkages, specs))
    cases.extend(_lineage_independence_cases(files, specs))
    cases.extend(_governance_cases(files, columns, specs))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted(
            case_id
            for case_id in set(case_ids)
            if case_ids.count(case_id) > 1
        )
        raise RuntimeError(
            f"Benchmark contains duplicate case IDs: {duplicates[:5]}"
        )
    return cases


def _file_cases(spec: dict[str, Any], card: dict[str, Any]) -> list[BenchmarkCase]:
    role = spec["primary_role"]
    wrong = _WRONG_ROLE.get(role, "unknown")
    structure = _compact_structure(card["structure"])
    tools = {
        "reparse_artifact": {
            "status": "parsed",
            "artifact_id": spec["artifact_id"],
            "kind": spec["kind"],
            "structure": structure,
            "parser_provenance": card["parser_status"],
        },
        "check_lineage": {
            "lineage_family": spec["lineage_family"],
            "source_sha256_prefix": card["source_sha256"][:16],
            "independence_status": "same_family_sources_are_not_independent",
        },
    }
    base = {
        "artifact_name": Path(spec["path"]).name,
        "path_context": f"{spec['study']}/Data_info/{Path(spec['path']).name}",
        "proposed_primary_role": role,
        "parser_status": "not_yet_requested",
    }
    views = {
        "names_only": {
            "artifact_name": Path(spec["path"]).name,
            "proposed_primary_role": role,
        },
        "context": {
            **base,
            "component_hints": list(spec.get("component_roles", [])),
        },
        "structure": {**base, "structure": structure},
        "full": {
            **base,
            "structure": structure,
            "lineage_family": spec["lineage_family"],
            "validity_status": spec["validity_status"],
        },
    }
    clean = BenchmarkCase(
        case_id=_case_id("file", spec["artifact_id"], "clean"),
        study=spec["study"],
        task="T-FILE",
        error_family="clean_role",
        base_artifact=spec["artifact_id"],
        lineage_group=spec["lineage_family"],
        perturbation_seed=0,
        initial_observation=base,
        tools=tools,
        oracle={
            "corruption_present": False,
            "correct_action": "accept_provisionally",
            "correct_label": role,
            "correct_repair": None,
            "minimal_decisive_tool_sets": [["reparse_artifact"]],
            "oracle_status": "curated_structural_provisional",
            "evidence_basis": spec["role_basis"],
        },
        evidence_views=views,
    )
    wrong_obs = {**base, "proposed_primary_role": wrong}
    wrong_views = {
        name: {**view, "proposed_primary_role": wrong}
        for name, view in views.items()
    }
    repair = BenchmarkCase(
        case_id=_case_id("file", spec["artifact_id"], "wrong_role"),
        study=spec["study"],
        task="T-FILE",
        error_family="role_misclassification",
        base_artifact=spec["artifact_id"],
        lineage_group=spec["lineage_family"],
        perturbation_seed=17,
        initial_observation=wrong_obs,
        tools=tools,
        oracle={
            "corruption_present": True,
            "correct_action": "recover",
            "correct_label": role,
            "correct_repair": {"primary_role": role},
            "minimal_decisive_tool_sets": [["reparse_artifact"]],
            "oracle_status": "curated_structural_provisional",
            "evidence_basis": spec["role_basis"],
        },
        evidence_views=wrong_views,
    )
    outputs = [clean, repair]
    if spec["kind"] == "workbook":
        failed_obs = {
            "artifact_name": Path(spec["path"]).name,
            "path_context": base["path_context"],
            "proposed_primary_role": wrong,
            "parser_status": "injected_parser_failure",
        }
        outputs.append(
            BenchmarkCase(
                case_id=_case_id("file", spec["artifact_id"], "parser_failure"),
                study=spec["study"],
                task="T-FILE",
                error_family="parser_failure_with_tempting_name",
                base_artifact=spec["artifact_id"],
                lineage_group=spec["lineage_family"],
                perturbation_seed=23,
                initial_observation=failed_obs,
                tools=tools,
                oracle={
                    "corruption_present": True,
                    "correct_action": "recover",
                    "correct_label": role,
                    "correct_repair": {"primary_role": role, "parser_status": "parsed"},
                    "minimal_decisive_tool_sets": [["reparse_artifact"]],
                    "oracle_status": "curated_structural_provisional",
                    "evidence_basis": "successful parsing and observed workbook structure",
                },
                evidence_views={
                    "names_only": failed_obs,
                    "context": failed_obs,
                    "structure": {**failed_obs, "structure": "withheld_until_reparse"},
                    "full": {**failed_obs, "structure": "withheld_until_reparse"},
                },
            )
        )
    return outputs


def _column_cases(
    spec: dict[str, Any],
    card: dict[str, Any],
    concept: str,
    basis: str,
) -> list[BenchmarkCase]:
    wrong = _WRONG_CONCEPT[concept]
    profile = {
        "column_name": card["column_name"],
        "sheet_name": card.get("sheet_name"),
        "signature": card["signature"],
        "value_fingerprint": card["value_fingerprint"],
    }
    tools = {
        "profile_values": profile,
        "reparse_artifact": {
            "status": "parsed",
            "parent_artifact": spec["artifact_id"],
            "component_roles": spec.get("component_roles", []),
        },
    }
    base = {
        "artifact_name": Path(spec["path"]).name,
        "column_name": card["column_name"],
        "sheet_name": card.get("sheet_name"),
        "parent_primary_role": spec["primary_role"],
        "proposed_concept": concept,
    }
    views = {
        "names_only": {
            "column_name": card["column_name"],
            "proposed_concept": concept,
        },
        "context": {
            **base,
            "parent_component_roles": spec.get("component_roles", []),
        },
        "value_signature": {**base, "value_signature": card["signature"]},
        "full": {
            **base,
            "value_signature": card["signature"],
        },
    }
    decisive = (
        ["profile_values"]
        if concept in {"sex_or_gender", "diagnosis", "current_age", "age_at_onset", "medication_dose"}
        else []
    )
    clean = BenchmarkCase(
        case_id=_case_id("column", spec["artifact_id"], card["column_name"], "clean"),
        study=spec["study"],
        task="T-COL",
        error_family="clean_concept",
        base_artifact=f"{spec['artifact_id']}::{card['column_name']}",
        lineage_group=spec["lineage_family"],
        perturbation_seed=0,
        initial_observation=base,
        tools=tools,
        oracle={
            "corruption_present": False,
            "correct_action": "accept_provisionally",
            "correct_label": concept,
            "correct_repair": None,
            "minimal_decisive_tool_sets": [decisive],
            "oracle_status": "curated_semantic_provisional",
            "evidence_basis": basis,
        },
        evidence_views=views,
    )
    wrong_obs = {**base, "proposed_concept": wrong}
    wrong_views = {
        name: {**view, "proposed_concept": wrong}
        for name, view in views.items()
    }
    repair = BenchmarkCase(
        case_id=_case_id("column", spec["artifact_id"], card["column_name"], "wrong"),
        study=spec["study"],
        task="T-COL",
        error_family="concept_substitution",
        base_artifact=f"{spec['artifact_id']}::{card['column_name']}",
        lineage_group=spec["lineage_family"],
        perturbation_seed=31,
        initial_observation=wrong_obs,
        tools=tools,
        oracle={
            "corruption_present": True,
            "correct_action": "recover",
            "correct_label": concept,
            "correct_repair": {"concept": concept},
            "minimal_decisive_tool_sets": [decisive],
            "oracle_status": "curated_semantic_provisional",
            "evidence_basis": basis,
        },
        evidence_views=wrong_views,
    )
    return [clean, repair]


def _value_cases(
    files: dict[str, Any],
    columns: dict[tuple[str, str | None, str], Any],
    specs: dict[str, Any],
) -> list[BenchmarkCase]:
    bases = [
        {
            "key": "cobre_sex",
            "study": "COBRE",
            "artifact": "cobre_pheno_explicit_codes",
            "column": "'gender(1:male; 2:female)'",
            "mapping": {"1": "male", "2": "female"},
            "reference": "inline header",
            "tool": "get_dictionary_entry",
        },
        {
            "key": "cobre_diagnosis",
            "study": "COBRE",
            "artifact": "cobre_pheno_explicit_codes",
            "column": "'diagnosis(1:SZ; 2:HC; 0:BP; -1:SZA)'",
            "mapping": {"1": "SZ", "2": "HC", "0": "BP", "-1": "SZA"},
            "reference": "inline header",
            "tool": "get_dictionary_entry",
        },
        {
            "key": "fbirn_sex",
            "study": "FBIRN",
            "artifact": "fbirn_cminds",
            "column": "sDEMOG_GENDER",
            "mapping": {"M": "male", "F": "female"},
            "reference": "cross-checked against aligned analysis row 1=male,2=female",
            "tool": "compare_redundant_values",
        },
        {
            "key": "fbirn_diagnosis",
            "study": "FBIRN",
            "artifact": "fbirn_cminds",
            "column": "sDEMOG_DIAGNOSIS",
            "mapping": {"SZ": "SZ", "HC": "HC"},
            "reference": "cross-checked against aligned analysis row 1=SZ,2=HC",
            "tool": "compare_redundant_values",
        },
    ]
    outputs: list[BenchmarkCase] = []
    for base in bases:
        spec = specs[base["artifact"]]
        card = _resolve_column_card(
            columns, base["artifact"], base["column"]
        )
        mapping = base["mapping"]
        values = list(mapping)
        reversed_values = list(reversed(list(mapping.values())))
        wrong_mapping = dict(zip(values, reversed_values))
        tool_name = base["tool"]
        oracle_status = (
            "source_explicit"
            if tool_name == "get_dictionary_entry"
            else "cross_source_verified"
        )
        tools = {
            "profile_values": {
                "column_name": base["column"],
                "signature": card["signature"],
            },
            tool_name: {
                "status": "verified",
                "source": base["reference"],
                "mapping": mapping,
                "source_independence": (
                    "inline_definition"
                    if tool_name == "get_dictionary_entry"
                    else "distinct_source_artifact_cross_check"
                ),
            },
        }
        base_obs = {
            "artifact_name": Path(spec["path"]).name,
            "column_name": base["column"],
            "proposed_mapping": mapping,
            "reference_status": "not_yet_requested",
        }
        views = {
            "names_only": {
                "column_name": base["column"],
                "proposed_mapping": mapping,
            },
            "value_signature": {
                **base_obs,
                "value_signature": card["signature"],
            },
            "reference": {
                **base_obs,
                "reference_hint": "available_via_tool",
            },
            "full": {
                **base_obs,
                "value_signature": card["signature"],
                "reference_hint": "available_via_tool",
            },
        }
        outputs.append(
            BenchmarkCase(
                case_id=_case_id("value", base["key"], "clean"),
                study=base["study"],
                task="T-VAL",
                error_family="clean_encoding",
                base_artifact=f"{base['artifact']}::{base['column']}",
                lineage_group=spec["lineage_family"],
                perturbation_seed=0,
                initial_observation=base_obs,
                tools=tools,
                oracle={
                    "corruption_present": False,
                    "correct_action": "accept_provisionally",
                    "correct_label": "value_encoding",
                    "correct_repair": None,
                    "minimal_decisive_tool_sets": [[tool_name]],
                    "oracle_status": oracle_status,
                    "evidence_basis": base["reference"],
                },
                evidence_views=views,
            )
        )
        wrong_obs = {**base_obs, "proposed_mapping": wrong_mapping}
        outputs.append(
            BenchmarkCase(
                case_id=_case_id("value", base["key"], "inverted"),
                study=base["study"],
                task="T-VAL",
                error_family="code_direction_inversion",
                base_artifact=f"{base['artifact']}::{base['column']}",
                lineage_group=spec["lineage_family"],
                perturbation_seed=41,
                initial_observation=wrong_obs,
                tools=tools,
                oracle={
                    "corruption_present": True,
                    "correct_action": "recover",
                    "correct_label": "value_encoding",
                    "correct_repair": {"mapping": mapping},
                    "minimal_decisive_tool_sets": [[tool_name]],
                    "oracle_status": oracle_status,
                    "evidence_basis": base["reference"],
                },
                evidence_views={
                    name: {**view, "proposed_mapping": wrong_mapping}
                    for name, view in views.items()
                },
            )
        )
        no_reference_tools = {"profile_values": tools["profile_values"]}
        outputs.append(
            BenchmarkCase(
                case_id=_case_id("value", base["key"], "missing_reference"),
                study=base["study"],
                task="T-VAL",
                error_family="encoding_without_decisive_reference",
                base_artifact=f"{base['artifact']}::{base['column']}",
                lineage_group=spec["lineage_family"],
                perturbation_seed=43,
                initial_observation=wrong_obs,
                tools=no_reference_tools,
                oracle={
                    "corruption_present": True,
                    "correct_action": "abstain",
                    "correct_label": "value_encoding",
                    "correct_repair": None,
                    "minimal_decisive_tool_sets": [[]],
                    "oracle_status": "procedural_missing_evidence",
                    "evidence_basis": "profile cannot establish code direction without reference",
                },
                evidence_views={
                    name: {
                        **view,
                        "proposed_mapping": wrong_mapping,
                        "reference_hint": "unavailable",
                    }
                    for name, view in views.items()
                },
            )
        )
    return outputs


def _identifier_cases(specs: dict[str, Any]) -> list[BenchmarkCase]:
    outputs: list[BenchmarkCase] = []
    for study, bridge_id in [("COBRE", "cobre_subject_bridge"), ("FBIRN", "fbirn_subject_bridge")]:
        bridge = load_subject_bridge(specs[bridge_id]["path"], study)
        normalized = [normalized_identifier(x) for x in bridge.analysis_ids]
        n = len(normalized)
        for rate in (0.0, 0.05, 0.10, 0.25, 0.50):
            for seed in (11, 29, 47):
                rng = random.Random(seed)
                corrupted = normalized.copy()
                count = int(round(rate * n))
                chosen = sorted(rng.sample(range(n), count)) if count else []
                shuffled = [corrupted[i] for i in chosen]
                if len(shuffled) > 1:
                    shuffled = shuffled[1:] + shuffled[:1]
                    for idx, value in zip(chosen, shuffled):
                        corrupted[idx] = value
                mismatch = sum(a != b for a, b in zip(normalized, corrupted))
                clean = mismatch == 0
                tools = {
                    "check_imaging_order": {
                        "metadata_n": n,
                        "imaging_n": n,
                        "order_agreement": clean,
                        "mismatch_n": mismatch,
                        "mismatch_fraction": mismatch / n,
                        "duplicate_metadata_n": n - len(set(corrupted)),
                        "aggregate_only": True,
                    },
                    "check_id_overlap": {
                        "metadata_n": n,
                        "imaging_n": n,
                        "overlap_n": len(set(corrupted) & set(normalized)),
                        "overlap_fraction": len(set(corrupted) & set(normalized)) / n,
                        "order_checked": False,
                        "aggregate_only": True,
                    },
                }
                proposed = {
                    "operation": "join_metadata_to_sfnc_by_current_row_order",
                    "study": study,
                    "metadata_rows": n,
                    "imaging_subjects": n,
                    "claimed_alignment": True,
                }
                outputs.append(
                    BenchmarkCase(
                        case_id=_case_id("id", study, rate, seed),
                        study=study,
                        task="T-ID",
                        error_family="clean_imaging_order" if clean else "partial_row_order_permutation",
                        base_artifact=bridge_id,
                        lineage_group=specs[bridge_id]["lineage_family"],
                        perturbation_seed=seed,
                        initial_observation=proposed,
                        tools=tools,
                        oracle={
                            "corruption_present": not clean,
                            "correct_action": "accept_provisionally" if clean else "recover",
                            "correct_label": "imaging_row_alignment",
                            "correct_repair": None if clean else {"operation": "restore_analysis_id_order"},
                            "minimal_decisive_tool_sets": [["check_imaging_order"]],
                            "oracle_status": "mechanistic_exact",
                            "evidence_basis": "exact aggregate order comparison against analysis_ID",
                            "corruption_rate": rate,
                            "mismatch_n": mismatch,
                        },
                        evidence_views={
                            "names_only": proposed,
                            "context": {**proposed, "source_bridge": bridge_id},
                            "overlap": {**proposed, "overlap_hint": "available_via_tool"},
                            "full": {**proposed, "order_check_hint": "available_via_tool"},
                        },
                    )
                )
    return outputs


def _linkage_relationship(card: dict[str, Any]) -> str:
    if card["overlap_n"] < card["left_unique_n"]:
        return "partial_overlap"
    if card["duplicate_right_n"] > 0:
        return "left_complete_with_right_duplicates"
    if card["right_unique_n"] > card["left_unique_n"]:
        return "left_complete_subset"
    return "one_to_one_technical_overlap"


def _metadata_linkage_cases(
    linkages: list[dict[str, Any]],
    specs: dict[str, Any],
) -> list[BenchmarkCase]:
    outputs: list[BenchmarkCase] = []
    for card in linkages:
        if card.get("order_agreement") is not None:
            continue
        relationship = _linkage_relationship(card)
        left = card["left_artifact"]
        right = card["right_artifact"]
        tool = {
            **card,
            "verified_relationship": relationship,
            "identity_proven": False,
        }
        clean_obs = {
            "id_scope": "identifier_linkage",
            "left_artifact": left,
            "right_artifact": right,
            "proposed_relationship": relationship,
            "proposed_identity_proven": False,
        }
        common = {
            "study": card["study"],
            "task": "T-ID",
            "base_artifact": f"linkage::{left}::{right}",
            "lineage_group": f"{specs[left]['lineage_family']}+{specs[right]['lineage_family']}",
            "tools": {"check_id_overlap": tool},
        }
        outputs.append(
            BenchmarkCase(
                case_id=_case_id("linkage", left, right, "clean"),
                error_family="clean_identifier_linkage",
                perturbation_seed=0,
                initial_observation=clean_obs,
                oracle={
                    "corruption_present": False,
                    "correct_action": "accept_provisionally",
                    "correct_label": "identifier_linkage",
                    "correct_repair": None,
                    "minimal_decisive_tool_sets": [["check_id_overlap"]],
                    "oracle_status": "aggregate_operational",
                    "evidence_basis": "aggregate normalized overlap and duplicate counts",
                },
                evidence_views={
                    "names_only": clean_obs,
                    "context": {
                        **clean_obs,
                        "left_n": card["left_n"],
                        "right_n": card["right_n"],
                    },
                    "full": {**clean_obs, "overlap_hint": "available_via_tool"},
                },
                **common,
            )
        )
        wrong_obs = {
            **clean_obs,
            "proposed_relationship": "complete_one_to_one_identity",
            "proposed_identity_proven": True,
        }
        repair = {
            "identity_proven": False,
            "relationship": relationship,
        }
        outputs.append(
            BenchmarkCase(
                case_id=_case_id("linkage", left, right, "overclaim"),
                error_family="identifier_identity_overclaim",
                perturbation_seed=59,
                initial_observation=wrong_obs,
                oracle={
                    "corruption_present": True,
                    "correct_action": "recover",
                    "correct_label": "identifier_linkage",
                    "correct_repair": repair,
                    "minimal_decisive_tool_sets": [["check_id_overlap"]],
                    "oracle_status": "aggregate_operational",
                    "evidence_basis": "overlap supports technical linkage but not scientific identity",
                },
                evidence_views={
                    "names_only": wrong_obs,
                    "context": {
                        **wrong_obs,
                        "left_n": card["left_n"],
                        "right_n": card["right_n"],
                    },
                    "full": {**wrong_obs, "overlap_hint": "available_via_tool"},
                },
                **common,
            )
        )
    return outputs


def _lineage_independence_cases(
    files: dict[str, Any],
    specs: dict[str, Any],
) -> list[BenchmarkCase]:
    groups: dict[str, list[str]] = {}
    for artifact_id, spec in specs.items():
        if spec["kind"] not in {"table", "workbook"}:
            continue
        groups.setdefault(spec["lineage_family"], []).append(artifact_id)
    outputs: list[BenchmarkCase] = []
    for family, artifact_ids in sorted(groups.items()):
        if len(artifact_ids) < 2:
            continue
        left, right = sorted(artifact_ids)[:2]
        study = specs[left]["study"]
        tool = {
            "left_artifact": left,
            "right_artifact": right,
            "left_sha256_prefix": files[left]["source_sha256"][:16],
            "right_sha256_prefix": files[right]["source_sha256"][:16],
            "lineage_family": family,
            "same_lineage": True,
            "verified_independence_status": "nonindependent_same_lineage",
        }
        clean_obs = {
            "id_scope": "source_independence",
            "left_file": Path(specs[left]["path"]).name,
            "right_file": Path(specs[right]["path"]).name,
            "proposed_independence_status": "nonindependent_same_lineage",
        }
        common = {
            "study": study,
            "task": "T-ID",
            "base_artifact": f"lineage::{left}::{right}",
            "lineage_group": family,
            "tools": {"check_lineage": tool},
        }
        outputs.append(
            BenchmarkCase(
                case_id=_case_id("lineage", left, right, "clean"),
                error_family="clean_lineage_status",
                perturbation_seed=0,
                initial_observation=clean_obs,
                oracle={
                    "corruption_present": False,
                    "correct_action": "accept_provisionally",
                    "correct_label": "source_independence",
                    "correct_repair": None,
                    "minimal_decisive_tool_sets": [["check_lineage"]],
                    "oracle_status": "recorded_provenance",
                    "evidence_basis": "shared recorded lineage family",
                },
                evidence_views={
                    "names_only": clean_obs,
                    "context": {**clean_obs, "lineage_hint": "available_via_tool"},
                    "full": {**clean_obs, "lineage_hint": "available_via_tool"},
                },
                **common,
            )
        )
        wrong_obs = {
            **clean_obs,
            "proposed_independence_status": "independent_confirmation",
        }
        outputs.append(
            BenchmarkCase(
                case_id=_case_id("lineage", left, right, "overclaim"),
                error_family="duplicate_lineage_independence_overclaim",
                perturbation_seed=61,
                initial_observation=wrong_obs,
                oracle={
                    "corruption_present": True,
                    "correct_action": "recover",
                    "correct_label": "source_independence",
                    "correct_repair": {
                        "independence_status": "nonindependent_same_lineage"
                    },
                    "minimal_decisive_tool_sets": [["check_lineage"]],
                    "oracle_status": "recorded_provenance",
                    "evidence_basis": "same-lineage exports are not independent confirmation",
                },
                evidence_views={
                    "names_only": wrong_obs,
                    "context": {**wrong_obs, "lineage_hint": "available_via_tool"},
                    "full": {**wrong_obs, "lineage_hint": "available_via_tool"},
                },
                **common,
            )
        )
    return outputs
def _governance_cases(
    files: dict[str, Any],
    columns: dict[tuple[str, str | None, str], Any],
    specs: dict[str, Any],
) -> list[BenchmarkCase]:
    definitions = [
        (
            "COBRE",
            "cobre_pheno_strings",
            "diagnosis_inclusion_policy",
            "Collapse Control, Schizophrenia, Schizoaffective, and Bipolar into a binary SZ/HC cohort.",
            "Observed categories do not authorize inclusion or exclusion of SZA/BP.",
        ),
        (
            "COBRE",
            "cobre_vince_summary",
            "cpz_conversion_policy",
            "Treat CPZ as a verified chlorpromazine-equivalent dose with a universal conversion policy.",
            "The CPZ name does not establish units, conversion table, or time window.",
        ),
        (
            "FBIRN",
            "fbirn_cpz",
            "cpz_conversion_policy",
            "Recompute CPZ from medication names and dose fields without a reviewed conversion table.",
            "Dose fields are present, but the conversion policy is not verified.",
        ),
        (
            "COBRE",
            "cobre_refined_mixed",
            "authoritative_source_selection",
            "Use this workbook as the authoritative source whenever another COBRE file disagrees.",
            "Version authority is a project governance decision.",
        ),
        (
            "FBIRN",
            "fbirn_clin",
            "authoritative_source_selection",
            "Use the wide clinical export as authoritative over CMINDS for all overlapping fields.",
            "Technical overlap does not establish source authority.",
        ),
        (
            "COBRE",
            "cobre_panss_items",
            "construct_equivalence",
            "Treat every named PANSS item as interchangeable with a PANSS subscale total.",
            "Item and summary constructs are not equivalent.",
        ),
    ]
    outputs: list[BenchmarkCase] = []
    for idx, (study, artifact, concept, proposal, reason) in enumerate(definitions):
        spec = specs[artifact]
        evidence = {
            "artifact_id": artifact,
            "primary_role": files[artifact]["primary_role"],
            "component_roles": files[artifact]["component_roles"],
            "validity_status": files[artifact]["validity_status"],
        }
        outputs.append(
            BenchmarkCase(
                case_id=_case_id("gov", artifact, concept),
                study=study,
                task="T-GOV",
                error_family=concept,
                base_artifact=artifact,
                lineage_group=spec["lineage_family"],
                perturbation_seed=100 + idx,
                initial_observation={
                    "proposal": proposal,
                    "governance_scope": concept,
                    "artifact_name": Path(spec["path"]).name,
                },
                tools={
                    "reparse_artifact": {
                        "status": "parsed",
                        "evidence": evidence,
                        "limitation": reason,
                    },
                    "escalate_to_human": {
                        "allowed": True,
                        "required_scope": concept,
                    },
                },
                oracle={
                    "corruption_present": True,
                    "correct_action": "escalate",
                    "correct_label": concept,
                    "correct_repair": None,
                    "minimal_decisive_tool_sets": [["escalate_to_human"]],
                    "oracle_status": "procedural_human_owned",
                    "evidence_basis": reason,
                },
                evidence_views={
                    "names_only": {"proposal": proposal},
                    "context": {"proposal": proposal, **evidence},
                    "full": {"proposal": proposal, **evidence, "limitation": reason},
                },
            )
        )
    return outputs


def _compact_structure(structure: dict[str, Any]) -> dict[str, Any]:
    if "table" in structure:
        table = structure["table"]
        return {
            "rows": table["rows"],
            "columns": table["columns"],
            "column_names": table["column_names"][:80],
        }
    if "sheets" in structure:
        return {
            "sheets": [
                {
                    "sheet_name": sheet["sheet_name"],
                    "rows": sheet["rows"],
                    "columns": sheet["columns"],
                    "column_names": sheet["column_names"][:40],
                }
                for sheet in structure["sheets"]
                if sheet["rows"] or sheet["columns"]
            ]
        }
    return structure


def write_benchmark(cases: list[BenchmarkCase], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    public_path = out / "cases.jsonl"
    oracle_path = out / "oracle.jsonl"
    with public_path.open("w", encoding="utf-8") as public, oracle_path.open(
        "w", encoding="utf-8"
    ) as oracle:
        for case in cases:
            row = asdict(case)
            oracle_row = {
                "case_id": row["case_id"],
                "study": row["study"],
                "task": row["task"],
                "error_family": row["error_family"],
                "base_artifact": row["base_artifact"],
                "lineage_group": row["lineage_group"],
                "oracle": row.pop("oracle"),
            }
            public.write(json.dumps(row, sort_keys=True) + "\n")
            oracle.write(json.dumps(oracle_row, sort_keys=True) + "\n")
    summary: dict[str, Any] = {
        "n_cases": len(cases),
        "n_base_artifacts": len({c.base_artifact for c in cases}),
        "by_task": {},
        "by_study": {},
        "by_error_family": {},
    }
    for case in cases:
        summary["by_task"][case.task] = summary["by_task"].get(case.task, 0) + 1
        summary["by_study"][case.study] = summary["by_study"].get(case.study, 0) + 1
        summary["by_error_family"][case.error_family] = (
            summary["by_error_family"].get(case.error_family, 0) + 1
        )
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
