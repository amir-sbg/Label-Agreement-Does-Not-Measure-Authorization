from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .matlab import load_sfnc, load_subject_bridge
from .paths import data_path, public_locator
from .privacy import normalized_identifier
from .source_io import read_table, sha256_file


@dataclass(frozen=True)
class SourceSpec:
    study: str
    bridge_artifact: str
    bridge_file: str
    sfnc_artifact: str
    sfnc_file: str
    source_artifact: str
    source_file: str
    source_id_column: str
    source_columns: dict[str, str]

    @property
    def bridge_path(self) -> str:
        return data_path(self.bridge_file)

    @property
    def sfnc_path(self) -> str:
        return data_path(self.sfnc_file)

    @property
    def source_path(self) -> str:
        return data_path(self.source_file)


DEFAULT_SOURCE_SPECS = (
    SourceSpec(
        study="COBRE",
        bridge_artifact="cobre_subject_bridge",
        bridge_file="Results/Subject_selection/COBRE/sub_info_COBRE.mat",
        sfnc_artifact="cobre_sfnc",
        sfnc_file="Results/SFNC/COBRE/COBRE.mat",
        source_artifact="cobre_pheno_explicit_codes",
        source_file="Data/COBRE/Data_info/COBRE_Phenotypic_File_ZeningFu.xlsx",
        source_id_column="ID",
        source_columns={
            "age": "'age'",
            "sex": "'gender(1:male; 2:female)'",
            "diagnosis": "'diagnosis(1:SZ; 2:HC; 0:BP; -1:SZA)'",
            "panss_total": "'PANSS(gentotal)'",
        },
    ),
    SourceSpec(
        study="FBIRN",
        bridge_artifact="fbirn_subject_bridge",
        bridge_file="Results/Subject_selection/FBIRN/sub_info_FBIRN.mat",
        sfnc_artifact="fbirn_sfnc",
        sfnc_file="Results/SFNC/FBIRN/FBIRN.mat",
        source_artifact="fbirn_cminds",
        source_file="Data/FBIRN/Data_info/FBIRN/FBIRN_CMINDs_20190801.xlsx",
        source_id_column="SubjectID",
        source_columns={
            "age": "nDEMOG_CUR_AGE",
            "sex": "sDEMOG_GENDER",
            "diagnosis": "sDEMOG_DIAGNOSIS",
            "cminds_composite": "CMINDS_composite",
        },
    ),
)


def masked_subject_key(study: str, raw_identifier: Any, salt: str) -> str:
    normalized = normalized_identifier(raw_identifier)
    if not normalized:
        raise ValueError(f"Empty identifier for {study}")
    digest = hashlib.sha256(
        f"{salt}|{study}|{normalized}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{study.lower()}_{digest}"


def _field_name(
    fields: tuple[str, ...],
    exact: str | None = None,
    prefix: str | None = None,
) -> str | None:
    if exact is not None and exact in fields:
        return exact
    if prefix is not None:
        for field in fields:
            if field.startswith(prefix):
                return field
    return None


def _as_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return None
    return float(converted)


def _output_number(value: Any) -> int | float | str:
    number = _as_number(value)
    if number is None:
        return ""
    return int(number) if number.is_integer() else number


def _canonical_sex(value: Any, study: str) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if study == "FBIRN":
        if text in {"M", "F"}:
            return text
        number = _as_number(value)
        return {1.0: "M", 2.0: "F"}.get(number, f"UNRESOLVED:{text}")
    number = _as_number(value)
    return {1.0: "M", 2.0: "F"}.get(number, f"UNRESOLVED:{text}")


def _canonical_diagnosis(value: Any, study: str) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if study == "FBIRN":
        if text in {"SZ", "HC"}:
            return text
        number = _as_number(value)
        return {1.0: "SZ", 2.0: "HC"}.get(number, f"UNRESOLVED:{text}")
    number = _as_number(value)
    return {
        1.0: "SZ",
        2.0: "HC",
        0.0: "BP",
        -1.0: "SZA",
    }.get(number, f"UNRESOLVED:{text}")


def _same_value(left: Any, right: Any, *, kind: str, study: str) -> bool:
    if kind == "age":
        left_number = _as_number(left)
        right_number = _as_number(right)
        if left_number is None or right_number is None:
            return left_number is None and right_number is None
        return bool(np.isclose(left_number, right_number, rtol=0.0, atol=1e-9))
    if kind == "sex":
        return _canonical_sex(left, study) == _canonical_sex(right, study)
    if kind == "diagnosis":
        return _canonical_diagnosis(left, study) == _canonical_diagnosis(right, study)
    raise ValueError(f"Unsupported comparison kind: {kind}")


def _source_records(frame: pd.DataFrame, id_column: str) -> dict[str, list[pd.Series]]:
    if id_column not in frame.columns:
        raise KeyError(f"Source identifier column missing: {id_column}")
    records: dict[str, list[pd.Series]] = {}
    for _, row in frame.iterrows():
        key = normalized_identifier(row[id_column])
        if key:
            records.setdefault(key, []).append(row)
    return records


def _source_value(
    record: pd.Series | None,
    records: dict[str, list[pd.Series]],
    key: str,
    column: str,
) -> Any:
    if record is None or len(records.get(key, [])) != 1:
        return None
    return record[column]


def _find_imaging_field(fields: tuple[str, ...], concept: str) -> str | None:
    if concept == "age":
        return _field_name(fields, exact="age")
    if concept == "sex":
        return _field_name(fields, exact="gender(1:male; 2:female)")
    if concept == "diagnosis":
        return _field_name(fields, prefix="diagnosis")
    if concept == "site":
        return _field_name(fields, exact="Site")
    raise ValueError(concept)


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, na_rep="")


def _provenance_row(
    *,
    study: str,
    field: str,
    source_artifact: str,
    source_locator: str,
    transformation: str,
    evidence_ids: str,
    oracle_strength: str,
    authorization_state: str,
    coverage_n: int,
    conflict_n: int,
    notes: str,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "cohort": study,
        "canonical_field": field,
        "source_artifact": source_artifact,
        "source_locator": public_locator(source_locator),
        "transformation": transformation,
        "evidence_ids": evidence_ids,
        "oracle_strength": oracle_strength,
        "authorization_state": authorization_state,
        "coverage_n": coverage_n,
        "conflict_n": conflict_n,
        "source_sha256": source_sha256,
        "notes": notes,
    }


def materialize_bundle(
    output_dir: str | Path,
    salt: str,
    specs: tuple[SourceSpec, ...] = DEFAULT_SOURCE_SPECS,
) -> dict[str, Any]:
    if not salt:
        raise ValueError("A non-empty salt is required and must not be written to output.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    canonical_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    study_summaries: list[dict[str, Any]] = []

    canonical_columns = [
        "cohort",
        "subject_key",
        "age",
        "sex",
        "diagnosis",
        "site",
        "bridge_status",
        "imaging_order_status",
        "sfnc_available",
        "clinical_source_artifact",
        "clinical_source_join_status",
        "source_age",
        "source_sex",
        "source_diagnosis",
        "source_panss_total",
        "source_cminds_composite",
    ]

    for spec in specs:
        bridge = load_subject_bridge(spec.bridge_path, spec.study)
        imaging = load_sfnc(spec.sfnc_path, spec.study)
        source = read_table(spec.source_path)
        source_sha256 = sha256_file(spec.source_path)
        bridge_sha256 = sha256_file(spec.bridge_path)
        records = _source_records(source, spec.source_id_column)

        if not bridge.analysis_ids:
            raise ValueError(f"{spec.study} has no analysis identifiers")
        exact_order = list(bridge.analysis_ids) == list(imaging.analysis_ids)
        if not exact_order:
            raise ValueError(f"{spec.study} bridge and sFNC identifier order differ")

        fields = imaging.field_names
        imaging_fields = {
            concept: _find_imaging_field(fields, concept)
            for concept in ("age", "sex", "diagnosis", "site")
        }
        missing_required = [
            concept
            for concept in ("age", "sex", "diagnosis")
            if imaging_fields[concept] is None
        ]
        if missing_required:
            raise ValueError(
                f"{spec.study} missing imaging metadata fields: {missing_required}"
            )

        score_index = {field: index for index, field in enumerate(fields)}
        source_counts = {
            "matched": 0,
            "missing": 0,
            "duplicate": 0,
            "conflict": 0,
        }
        conflict_counts = {"age": 0, "sex": 0, "diagnosis": 0}
        site_values = (
            imaging.scores[score_index[imaging_fields["site"]]]
            if imaging_fields["site"] is not None
            else None
        )

        for position, raw_identifier in enumerate(imaging.analysis_ids):
            lookup_key = normalized_identifier(raw_identifier)
            matching = records.get(lookup_key, [])
            record = matching[0] if len(matching) == 1 else None
            if not matching:
                source_status = "not_in_source_table"
                source_counts["missing"] += 1
            elif len(matching) > 1:
                source_status = "ambiguous_duplicate_source_key"
                source_counts["duplicate"] += 1
            else:
                source_counts["matched"] += 1
                mismatch_kinds = []
                for concept in ("age", "sex", "diagnosis"):
                    source_column = spec.source_columns[concept]
                    source_value = record[source_column]
                    imaging_value = imaging.scores[
                        score_index[imaging_fields[concept]]
                    ][position]
                    if not _same_value(
                        source_value,
                        imaging_value,
                        kind=concept,
                        study=spec.study,
                    ):
                        mismatch_kinds.append(concept)
                        conflict_counts[concept] += 1
                if mismatch_kinds:
                    source_status = "matched_source_conflict_review"
                    source_counts["conflict"] += 1
                else:
                    source_status = "matched_verified"

            age = imaging.scores[score_index[imaging_fields["age"]]][position]
            sex = imaging.scores[score_index[imaging_fields["sex"]]][position]
            diagnosis = imaging.scores[
                score_index[imaging_fields["diagnosis"]]
            ][position]
            sfnc_available = bool(np.isfinite(imaging.sfnc[:, :, position]).all())
            site = (
                _output_number(site_values[position])
                if site_values is not None
                else ""
            )

            source_age = _source_value(
                record, records, lookup_key, spec.source_columns["age"]
            )
            source_sex = _source_value(
                record, records, lookup_key, spec.source_columns["sex"]
            )
            source_diagnosis = _source_value(
                record, records, lookup_key, spec.source_columns["diagnosis"]
            )
            source_panss_total = ""
            source_cminds_composite = ""
            if "panss_total" in spec.source_columns:
                source_panss_total = _output_number(
                    _source_value(
                        record,
                        records,
                        lookup_key,
                        spec.source_columns["panss_total"],
                    )
                )
            if "cminds_composite" in spec.source_columns:
                source_cminds_composite = _output_number(
                    _source_value(
                        record,
                        records,
                        lookup_key,
                        spec.source_columns["cminds_composite"],
                    )
                )

            canonical_rows.append(
                {
                    "cohort": spec.study,
                    "subject_key": masked_subject_key(
                        spec.study, raw_identifier, salt
                    ),
                    "age": _output_number(age),
                    "sex": _canonical_sex(sex, spec.study),
                    "diagnosis": _canonical_diagnosis(diagnosis, spec.study),
                    "site": site,
                    "bridge_status": "exact_id_order",
                    "imaging_order_status": "exact_id_order",
                    "sfnc_available": "yes" if sfnc_available else "no",
                    "clinical_source_artifact": spec.source_artifact,
                    "clinical_source_join_status": source_status,
                    "source_age": _output_number(source_age),
                    "source_sex": _canonical_sex(source_sex, spec.study),
                    "source_diagnosis": _canonical_diagnosis(
                        source_diagnosis, spec.study
                    ),
                    "source_panss_total": source_panss_total,
                    "source_cminds_composite": source_cminds_composite,
                }
            )

        n_subjects = len(imaging.analysis_ids)
        source_coverage = source_counts["matched"]
        source_conflict_n = source_counts["conflict"] + source_counts["duplicate"]
        study_summaries.append(
            {
                "cohort": spec.study,
                "n_subjects": n_subjects,
                "source_rows": int(len(source)),
                "source_matched_n": source_coverage,
                "source_missing_n": source_counts["missing"],
                "source_duplicate_n": source_counts["duplicate"],
                "source_conflict_n": source_counts["conflict"],
                "age_conflict_n": conflict_counts["age"],
                "sex_conflict_n": conflict_counts["sex"],
                "diagnosis_conflict_n": conflict_counts["diagnosis"],
                "bridge_sfnc_order": "exact",
                "source_artifact": spec.source_artifact,
            }
        )

        linkage_evidence = f"linkage:{spec.bridge_artifact}:{spec.sfnc_artifact}"
        source_evidence = f"table_crosscheck:{spec.source_artifact}"
        sfnc_sha256 = sha256_file(spec.sfnc_path)
        for field, locator, transform, strength, auth, conflict, notes in [
            (
                "subject_key",
                f"{spec.bridge_path}::analysis_ID",
                "cohort-scoped salted SHA-256 pseudonym",
                "mechanical",
                "authorized_for_derived_bundle",
                0,
                "Raw analysis IDs are never serialized.",
            ),
            (
                "age",
                f"{spec.sfnc_path}::analysis_SCORE::age",
                "direct analysis-score value; source-table cross-check retained",
                "mechanical_plus_crosscheck",
                "authorized_for_analysis_bundle",
                conflict_counts["age"],
                "Cross-source conflicts remain in unresolved.csv.",
            ),
            (
                "sex",
                f"{spec.sfnc_path}::analysis_SCORE::gender",
                "numeric code converted using the explicit field header",
                "explicit_header_plus_crosscheck",
                "authorized_for_analysis_bundle",
                conflict_counts["sex"],
                "The source-table value is retained separately.",
            ),
            (
                "diagnosis",
                f"{spec.sfnc_path}::analysis_SCORE::diagnosis",
                "numeric or string code converted using the explicit field header",
                "explicit_header_plus_crosscheck",
                "authorized_for_analysis_bundle",
                conflict_counts["diagnosis"],
                "Cohort inclusion policy is not inferred from this artifact.",
            ),
            (
                "site",
                (
                    f"{spec.sfnc_path}::analysis_SCORE::Site"
                    if imaging_fields["site"] is not None
                    else "not_available_in_sfnc_artifact"
                ),
                "direct observed site code; no codebook transformation",
                "mechanical_observation",
                (
                    "provisional_observed_code"
                    if imaging_fields["site"] is not None
                    else "deferred_missing_source"
                ),
                0,
                "Site meaning or missing COBRE site values require further source evidence.",
            ),
            (
                "bridge_status",
                f"{spec.bridge_path} + {spec.sfnc_path}",
                "exact identifier equality and row-order check",
                "mechanical",
                "authorized_for_analysis_bundle",
                0,
                "This verifies operational alignment, not scientific identity in every source.",
            ),
            (
                "imaging_order_status",
                f"{spec.bridge_path} + {spec.sfnc_path}",
                "exact analysis-ID order equality",
                "mechanical",
                "authorized_for_analysis_bundle",
                0,
                "Required before attaching metadata to sFNC rows.",
            ),
            (
                "sfnc_available",
                f"{spec.sfnc_path}::sFNC",
                "finite tensor check at the subject index",
                "mechanical",
                "authorized_for_derived_bundle",
                0,
                "This records availability, not imaging quality.",
            ),
            (
                "clinical_source_join_status",
                spec.source_path,
                "normalized identifier lookup with duplicate and value checks",
                "mechanical_plus_crosscheck",
                "review_on_conflict_or_missing",
                source_conflict_n,
                "Missing source rows and conflicts are not auto-repaired.",
            ),
        ]:
            evidence = linkage_evidence
            if field in {"age", "sex", "diagnosis", "clinical_source_join_status"}:
                evidence = f"{linkage_evidence};{source_evidence}"
            provenance_rows.append(
                _provenance_row(
                    study=spec.study,
                    field=field,
                    source_artifact=(
                        spec.bridge_artifact
                        if field == "subject_key"
                        else (
                            spec.source_artifact
                            if field == "clinical_source_join_status"
                            else spec.sfnc_artifact
                        )
                    ),
                    source_locator=locator,
                    transformation=transform,
                    evidence_ids=evidence,
                    oracle_strength=strength,
                    authorization_state=auth,
                    coverage_n=(
                        n_subjects
                        if field != "clinical_source_join_status"
                        else source_coverage
                    ),
                    conflict_n=conflict,
                    notes=notes,
                    source_sha256=(
                        bridge_sha256
                        if field == "subject_key"
                        else (
                            source_sha256
                            if field == "clinical_source_join_status"
                            else sfnc_sha256
                        )
                    ),
                )
            )

        for field, column in spec.source_columns.items():
            output_field = {
                "age": "source_age",
                "sex": "source_sex",
                "diagnosis": "source_diagnosis",
                "panss_total": "source_panss_total",
                "cminds_composite": "source_cminds_composite",
            }[field]
            provenance_rows.append(
                _provenance_row(
                    study=spec.study,
                    field=output_field,
                    source_artifact=spec.source_artifact,
                    source_locator=f"{spec.source_path}::{column}",
                    transformation=(
                        "numeric value retained"
                        if field not in {"sex", "diagnosis"}
                        else "source code normalized to canonical display category"
                    ),
                    evidence_ids=source_evidence,
                    oracle_strength="source_observation_only",
                    authorization_state="provisional_source_value",
                    coverage_n=source_coverage,
                    conflict_n=conflict_counts.get(field, 0),
                    notes="Not used to replace the canonical analysis value automatically.",
                    source_sha256=source_sha256,
                )
            )

        if imaging_fields["site"] is None:
            unresolved_rows.append(
                {
                    "issue_id": f"{spec.study.lower()}_site_missing",
                    "cohort": spec.study,
                    "scope": "canonical_site",
                    "affected_n": n_subjects,
                    "issue_type": "missing_source_evidence",
                    "source_artifact": spec.sfnc_artifact,
                    "evidence_ids": linkage_evidence,
                    "severity": "medium",
                    "action": "retrieve_site_metadata_or_authoritative_crosswalk",
                    "reason": "The selected analysis artifact has no Site field for this cohort.",
                }
            )
        elif spec.study == "FBIRN":
            unresolved_rows.append(
                {
                    "issue_id": "fbirn_site_codebook_missing",
                    "cohort": spec.study,
                    "scope": "canonical_site",
                    "affected_n": n_subjects,
                    "issue_type": "missing_reference",
                    "source_artifact": spec.sfnc_artifact,
                    "evidence_ids": linkage_evidence,
                    "severity": "low",
                    "action": "attach_site_dictionary_before_cross_site_interpretation",
                    "reason": "Site codes are observed and preserved, but their external labels are not verified here.",
                }
            )

        if source_counts["missing"]:
            unresolved_rows.append(
                {
                    "issue_id": f"{spec.study.lower()}_source_coverage",
                    "cohort": spec.study,
                    "scope": spec.source_artifact,
                    "affected_n": source_counts["missing"],
                    "issue_type": "partial_source_coverage",
                    "source_artifact": spec.source_artifact,
                    "evidence_ids": source_evidence,
                    "severity": "medium",
                    "action": "check_cohort_inclusion_or_identifier_crosswalk",
                    "reason": "The clinical source table does not contain every analysis subject.",
                }
            )
        if source_counts["duplicate"]:
            unresolved_rows.append(
                {
                    "issue_id": f"{spec.study.lower()}_duplicate_source_keys",
                    "cohort": spec.study,
                    "scope": spec.source_artifact,
                    "affected_n": source_counts["duplicate"],
                    "issue_type": "duplicate_join_key",
                    "source_artifact": spec.source_artifact,
                    "evidence_ids": source_evidence,
                    "severity": "high",
                    "action": "resolve_entity_or_visit_granularity_before_join",
                    "reason": "Multiple source rows share a normalized key; automatic selection is blocked.",
                }
            )
        for concept, count in conflict_counts.items():
            if count:
                unresolved_rows.append(
                    {
                        "issue_id": f"{spec.study.lower()}_{concept}_cross_source_conflict",
                        "cohort": spec.study,
                        "scope": concept,
                        "affected_n": count,
                        "issue_type": "source_crosscheck_conflict",
                        "source_artifact": spec.source_artifact,
                        "evidence_ids": f"{linkage_evidence};{source_evidence}",
                        "severity": "high" if concept == "diagnosis" else "medium",
                        "action": "human_adjudication_before_source_value_replacement",
                        "reason": "The clinical source value disagrees with the analysis artifact; no automatic overwrite is allowed.",
                    }
                )

        if "panss_total" in spec.source_columns:
            unresolved_rows.append(
                {
                    "issue_id": f"{spec.study.lower()}_panss_semantics",
                    "cohort": spec.study,
                    "scope": "source_panss_total",
                    "affected_n": source_coverage,
                    "issue_type": "semantic_policy_unverified",
                    "source_artifact": spec.source_artifact,
                    "evidence_ids": source_evidence,
                    "severity": "medium",
                    "action": "verify_scale_definition_and_missing_value_policy",
                    "reason": "The field name identifies a PANSS total candidate, but this materializer does not establish scale semantics.",
                }
            )
        if "cminds_composite" in spec.source_columns:
            unresolved_rows.append(
                {
                    "issue_id": f"{spec.study.lower()}_cminds_composite_semantics",
                    "cohort": spec.study,
                    "scope": "source_cminds_composite",
                    "affected_n": source_coverage,
                    "issue_type": "semantic_policy_unverified",
                    "source_artifact": spec.source_artifact,
                    "evidence_ids": source_evidence,
                    "severity": "medium",
                    "action": "verify_dictionary_scope_and_composite_definition",
                    "reason": "The composite is carried as a source observation; it is not treated as a harmonized cross-cohort construct.",
                }
            )

        if spec.study == "COBRE":
            unresolved_rows.append(
                {
                    "issue_id": "cobre_non_analysis_diagnosis_codes",
                    "cohort": spec.study,
                    "scope": "diagnosis_inclusion_policy",
                    "affected_n": int(
                        source[spec.source_columns["diagnosis"]]
                        .isin([0, -1])
                        .sum()
                    ),
                    "issue_type": "cohort_policy_unverified",
                    "source_artifact": spec.source_artifact,
                    "evidence_ids": source_evidence,
                    "severity": "medium",
                    "action": "document_analysis_inclusion_policy",
                    "reason": "The full source table contains BP/SZA codes, while the selected analysis bridge contains only SZ/HC subjects.",
                }
            )

        unresolved_rows.append(
            {
                "issue_id": f"{spec.study.lower()}_cpz_conversion_policy",
                "cohort": spec.study,
                "scope": "CPZ",
                "affected_n": n_subjects,
                "issue_type": "unsupported_transformation",
                "source_artifact": spec.sfnc_artifact,
                "evidence_ids": linkage_evidence,
                "severity": "high",
                "action": "attach_authoritative_units_and_conversion_policy",
                "reason": "CPZ-like fields are intentionally excluded because units and conversion policy are not established by this bundle.",
            }
        )

    _write_csv(
        canonical_rows,
        output / "canonical_subject_bundle.csv",
        canonical_columns,
    )
    provenance_columns = [
        "cohort",
        "canonical_field",
        "source_artifact",
        "source_locator",
        "transformation",
        "evidence_ids",
        "oracle_strength",
        "authorization_state",
        "coverage_n",
        "conflict_n",
        "source_sha256",
        "notes",
    ]
    _write_csv(
        provenance_rows,
        output / "field_provenance.csv",
        provenance_columns,
    )
    unresolved_columns = [
        "issue_id",
        "cohort",
        "scope",
        "affected_n",
        "issue_type",
        "source_artifact",
        "evidence_ids",
        "severity",
        "action",
        "reason",
    ]
    _write_csv(unresolved_rows, output / "unresolved.csv", unresolved_columns)

    summary = {
        "version": "materialized_bundle_v1",
        "privacy": {
            "raw_subject_identifiers_serialized": False,
            "subject_key": "cohort_scoped_salted_sha256_prefix",
            "salt_serialized": False,
        },
        "row_count": len(canonical_rows),
        "row_count_by_cohort": {
            study["cohort"]: study["n_subjects"] for study in study_summaries
        },
        "source_summaries": study_summaries,
        "unresolved_issue_count": len(unresolved_rows),
        "canonical_columns": canonical_columns,
        "scope": (
            "Derived analysis bundle anchored to exact bridge-to-sFNC order. "
            "It is not a universal harmonized database or a clinical decision dataset."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        _readme_text(summary),
        encoding="utf-8",
    )
    return summary


def _readme_text(summary: dict[str, Any]) -> str:
    rows = "\n".join(
        f"- {item['cohort']}: {item['n_subjects']} analysis subjects; "
        f"{item['source_matched_n']} matched to the selected clinical table; "
        f"{item['source_conflict_n']} source conflicts; "
        f"{item['source_missing_n']} source rows absent."
        for item in summary["source_summaries"]
    )
    return f"""# Materialized Subject Bundle v1

This is a derived, privacy-safe integration artifact. It is anchored to the existing subject-selection bridge and sFNC
analysis order. The original read-only source files were never modified.
Source locators are relative to NEUROMARK_ROOT; absolute mount paths are
not serialized into this artifact.

## Outputs

- canonical_subject_bundle.csv: one row per analysis subject, with a
  cohort-scoped pseudonymous key, canonical analysis metadata, source-table
  coverage status, and source observations retained separately.
- field_provenance.csv: source locator, transformation, evidence IDs,
  authorization state, and conflict counts for every output field.
- unresolved.csv: aggregated review blockers. No source conflict is silently
  overwritten.
- summary.json: counts and privacy scope.

## Coverage

{rows}

## Interpretation

The canonical age, sex, and diagnosis values are taken from the subject-indexed
analysis artifact after exact bridge-to-sFNC order verification. Selected
clinical tables are joined only for evidence and additional source fields.
When a source value is missing or conflicts, the analysis value is retained for
the derived bundle and the source value is routed to unresolved review.

The bundle does not establish a universal ontology, hospital interoperability,
CPZ conversion, or clinical validity. Subject keys are salted hashes and the
salt is not stored in this directory.
"""
