from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import yaml

from .matlab import load_sfnc, load_subject_bridge
from .paths import expand, public_locator
from .privacy import normalized_identifier, stable_digest
from .source_io import (
    artifact_structure,
    read_table,
    select_existing_columns,
    sha256_file,
    summarize_column,
    workbook_sheets,
)


@dataclass(frozen=True)
class FileCard:
    card_id: str
    artifact_id: str
    study: str
    source_path: str
    source_sha256: str
    source_size: int
    kind: str
    parser_status: str
    primary_role: str
    component_roles: tuple[str, ...]
    role_basis: str
    validity_status: str
    lineage_family: str
    structure: dict[str, Any]
    missing_requested_columns: tuple[str, ...]


@dataclass(frozen=True)
class ColumnCard:
    card_id: str
    artifact_id: str
    study: str
    source_locator: str
    column_name: str
    sheet_name: str | None
    signature: dict[str, Any]
    value_fingerprint: str
    privacy_scope: str = "aggregate_only"


@dataclass(frozen=True)
class ReferenceCard:
    card_id: str
    artifact_id: str
    study: str
    source_locator: str
    reference_scope: str
    verified_fields: tuple[str, ...]
    limitation: str


@dataclass(frozen=True)
class LinkageCard:
    card_id: str
    study: str
    left_artifact: str
    right_artifact: str
    normalization: str
    left_n: int
    right_n: int
    overlap_n: int
    left_unique_n: int
    right_unique_n: int
    duplicate_left_n: int
    duplicate_right_n: int
    order_agreement: bool | None
    identity_status: str
    privacy_scope: str = "aggregate_only"


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    for key, value in manifest.get("source_roots", {}).items():
        manifest["source_roots"][key] = expand(value)
    for spec in manifest.get("artifacts", []):
        spec["path"] = expand(spec["path"])
    return manifest


def build_cards(manifest_path: str | Path) -> dict[str, list[Any]]:
    manifest = load_manifest(manifest_path)
    files: list[FileCard] = []
    columns: list[ColumnCard] = []
    references: list[ReferenceCard] = []

    for spec in manifest["artifacts"]:
        path = Path(spec["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        missing: list[str] = []
        try:
            structure = artifact_structure(path, spec["kind"])
            parser_status = "parsed"
            if spec["kind"] in {"table", "workbook"}:
                new_columns, new_missing = _column_cards_for_spec(spec)
                columns.extend(new_columns)
                missing.extend(new_missing)
        except Exception as exc:
            structure = {"error_type": type(exc).__name__, "error": str(exc)}
            parser_status = "failed"

        file_card = FileCard(
            card_id=f"file:{spec['artifact_id']}",
            artifact_id=spec["artifact_id"],
            study=spec["study"],
            source_path=public_locator(str(path)),
            source_sha256=sha256_file(path),
            source_size=path.stat().st_size,
            kind=spec["kind"],
            parser_status=parser_status,
            primary_role=spec["primary_role"],
            component_roles=tuple(spec.get("component_roles", [])),
            role_basis=spec["role_basis"],
            validity_status=spec["validity_status"],
            lineage_family=spec["lineage_family"],
            structure=structure,
            missing_requested_columns=tuple(sorted(set(missing))),
        )
        files.append(file_card)
        references.extend(_reference_cards(spec, file_card))

    links = _build_linkage_cards(manifest["artifacts"])
    return {"file_cards": files, "column_cards": columns, "reference_cards": references, "linkage_cards": links}


def _column_cards_for_spec(spec: dict[str, Any]) -> tuple[list[ColumnCard], list[str]]:
    path = Path(spec["path"])
    requested = list(spec.get("selected_columns", []))
    output: list[ColumnCard] = []
    missing_all = set(requested)
    sheets: Iterable[str | None]
    if spec["kind"] == "workbook":
        selected = spec.get("selected_sheets") or list(workbook_sheets(path))
        sheets = [str(x) for x in selected]
    else:
        sheets = [None]

    for sheet in sheets:
        df = read_table(path, sheet)
        selected, _ = select_existing_columns(df, requested)
        for column in selected:
            summary = summarize_column(df, column)
            locator = f"{path}::{sheet}::{column}" if sheet else f"{path}::{column}"
            output.append(
                ColumnCard(
                    card_id=f"column:{spec['artifact_id']}:{stable_digest(str(sheet) + ':' + column)}",
                    artifact_id=spec["artifact_id"],
                    study=spec["study"],
                    source_locator=public_locator(locator),
                    column_name=column,
                    sheet_name=sheet,
                    signature=summary["signature"],
                    value_fingerprint=summary["value_fingerprint"],
                )
            )
            missing_all.discard(column)
    return output, sorted(missing_all)


def _require_reference_fields(
    file_card: FileCard,
    required: set[str],
    *,
    sheet_name: str | None = None,
    every_nonempty_sheet: bool = False,
) -> None:
    if file_card.parser_status != "parsed":
        raise ValueError(
            f"Cannot verify reference fields for unparsed {file_card.artifact_id}"
        )
    sheets = [
        sheet
        for sheet in file_card.structure.get("sheets", [])
        if sheet.get("rows") or sheet.get("columns")
    ]
    if sheet_name is not None:
        sheets = [
            sheet for sheet in sheets
            if sheet.get("sheet_name") == sheet_name
        ]
        if not sheets:
            raise ValueError(
                f"Reference sheet {sheet_name} missing from {file_card.artifact_id}"
            )
    if not sheets:
        raise ValueError(
            f"No parsed reference sheets in {file_card.artifact_id}"
        )
    checked = sheets if every_nonempty_sheet else sheets[:1]
    for sheet in checked:
        missing = required - set(sheet.get("column_names", []))
        if missing:
            raise ValueError(
                f"Reference fields missing from {file_card.artifact_id}::"
                f"{sheet.get('sheet_name')}: {sorted(missing)}"
            )


def _reference_cards(spec: dict[str, Any], file_card: FileCard) -> list[ReferenceCard]:
    artifact_id = spec["artifact_id"]
    cards: list[ReferenceCard] = []
    if artifact_id == "cobre_pheno_explicit_codes":
        _require_reference_fields(
            file_card,
            {
                "'gender(1:male; 2:female)'",
                "'diagnosis(1:SZ; 2:HC; 0:BP; -1:SZA)'",
            },
            sheet_name="Sheet1",
        )
        cards.append(
            ReferenceCard(
                card_id="reference:cobre_inline_sex_diagnosis",
                artifact_id=artifact_id,
                study="COBRE",
                source_locator=public_locator(f"{spec['path']}::Sheet1::header"),
                reference_scope="inline sex and diagnosis encodings",
                verified_fields=("sex:1=male,2=female", "diagnosis:1=SZ,2=HC,0=BP,-1=SZA"),
                limitation="Headers verify code direction but not cohort inclusion policy.",
            )
        )
    if artifact_id == "fbirn_cminds_dictionary":
        _require_reference_fields(
            file_card,
            {"Field_Name", "Description", "Valid_Values"},
            every_nonempty_sheet=True,
        )
        cards.append(
            ReferenceCard(
                card_id="reference:fbirn_cminds_dictionary_scope",
                artifact_id=artifact_id,
                study="FBIRN",
                source_locator=public_locator(str(spec["path"])),
                reference_scope="CMINDS cognitive test fields and valid values",
                verified_fields=("Field_Name", "Description", "Valid_Values"),
                limitation=(
                    "This workbook documents CMINDS cognitive instruments. It is not evidence "
                    "for the separate sDEMOG sex or diagnosis code direction."
                ),
            )
        )
    if artifact_id == "cobre_med_dictionary_workbook":
        _require_reference_fields(
            file_card,
            {
                "Question ID",
                "Question Description",
                "Response Description",
                "Response Value",
            },
            sheet_name="Dictionary",
        )
        cards.append(
            ReferenceCard(
                card_id="reference:cobre_med_internal_dictionary",
                artifact_id=artifact_id,
                study="COBRE",
                source_locator=public_locator(f"{spec['path']}::Dictionary"),
                reference_scope="medication instrument questions and response values",
                verified_fields=("Question ID", "Question Description", "Response Description", "Response Value"),
                limitation="Dictionary presence does not by itself establish a CPZ conversion policy.",
            )
        )
    return cards


def _artifact_specs(specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {spec["artifact_id"]: spec for spec in specs}


def _build_linkage_cards(specs: list[dict[str, Any]]) -> list[LinkageCard]:
    by_id = _artifact_specs(specs)
    cards: list[LinkageCard] = []
    cards.append(
        _link_bridge_to_table(
            "COBRE",
            by_id["cobre_subject_bridge"],
            by_id["cobre_pheno_explicit_codes"],
            "ID",
        )
    )
    cards.append(
        _link_bridge_to_table(
            "FBIRN",
            by_id["fbirn_subject_bridge"],
            by_id["fbirn_clin"],
            "SubjectID",
        )
    )
    cards.append(
        _link_bridge_to_table(
            "FBIRN",
            by_id["fbirn_subject_bridge"],
            by_id["fbirn_cminds"],
            "SubjectID",
        )
    )
    cards.append(
        _link_bridge_to_sfnc(
            "COBRE", by_id["cobre_subject_bridge"], by_id["cobre_sfnc"]
        )
    )
    cards.append(
        _link_bridge_to_sfnc(
            "FBIRN", by_id["fbirn_subject_bridge"], by_id["fbirn_sfnc"]
        )
    )
    return cards


def _link_bridge_to_table(
    study: str,
    bridge_spec: dict[str, Any],
    table_spec: dict[str, Any],
    id_column: str,
) -> LinkageCard:
    bridge = load_subject_bridge(bridge_spec["path"], study)
    sheet = (table_spec.get("selected_sheets") or [None])[0]
    table = read_table(table_spec["path"], sheet)
    left = [normalized_identifier(x) for x in bridge.analysis_ids]
    right = [normalized_identifier(x) for x in table[id_column].tolist()]
    return _linkage_card(
        study,
        bridge_spec["artifact_id"],
        table_spec["artifact_id"],
        left,
        right,
        order=None,
        status="technical_overlap_not_identity_proof",
    )


def _link_bridge_to_sfnc(
    study: str, bridge_spec: dict[str, Any], sfnc_spec: dict[str, Any]
) -> LinkageCard:
    bridge = load_subject_bridge(bridge_spec["path"], study)
    imaging = load_sfnc(sfnc_spec["path"], study)
    left = [normalized_identifier(x) for x in bridge.analysis_ids]
    right = [normalized_identifier(x) for x in imaging.analysis_ids]
    order = left == right and len(left) == imaging.sfnc.shape[-1]
    return _linkage_card(
        study,
        bridge_spec["artifact_id"],
        sfnc_spec["artifact_id"],
        left,
        right,
        order=order,
        status="operational_alignment_verified" if order else "alignment_conflict",
    )


def _linkage_card(
    study: str,
    left_name: str,
    right_name: str,
    left: list[str],
    right: list[str],
    order: bool | None,
    status: str,
) -> LinkageCard:
    left_nonempty = [x for x in left if x]
    right_nonempty = [x for x in right if x]
    left_set, right_set = set(left_nonempty), set(right_nonempty)
    return LinkageCard(
        card_id=f"linkage:{left_name}:{right_name}",
        study=study,
        left_artifact=left_name,
        right_artifact=right_name,
        normalization="uppercase_alnum_and_numeric_leading_zero_removal",
        left_n=len(left),
        right_n=len(right),
        overlap_n=len(left_set & right_set),
        left_unique_n=len(left_set),
        right_unique_n=len(right_set),
        duplicate_left_n=len(left_nonempty) - len(left_set),
        duplicate_right_n=len(right_nonempty) - len(right_set),
        order_agreement=order,
        identity_status=status,
    )


def write_cards(cards: dict[str, list[Any]], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, values in cards.items():
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(asdict(value), sort_keys=True) + "\n")
    summary = {name: len(values) for name, values in cards.items()}
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
