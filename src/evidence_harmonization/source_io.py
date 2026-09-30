from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .matlab import root_manifest
from .privacy import safe_fingerprint, value_signature


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet or 0)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t", engine="python")
    if suffix == ".sav":
        return pd.read_spss(p)
    raise ValueError(f"Unsupported tabular format: {p}")


def workbook_sheets(path: str | Path) -> tuple[str, ...]:
    return tuple(pd.ExcelFile(path).sheet_names)


def table_structure(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
    }


def select_existing_columns(
    df: pd.DataFrame, requested: Iterable[str]
) -> tuple[list[str], list[str]]:
    exact = {str(c): c for c in df.columns}
    selected: list[str] = []
    missing: list[str] = []
    for name in requested:
        if name in exact:
            selected.append(name)
        else:
            missing.append(name)
    return selected, missing


def summarize_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    series = df[column]
    return {
        "column_name": str(column),
        "signature": value_signature(series, str(column)),
        "value_fingerprint": safe_fingerprint(series, str(column)),
    }


def artifact_structure(path: str | Path, kind: str) -> dict[str, Any]:
    p = Path(path)
    if kind in {"matlab_bridge", "matlab_imaging"}:
        return {"variables": root_manifest(p)}
    if p.suffix.lower() in {".xlsx", ".xls"}:
        sheets = []
        for name in workbook_sheets(p):
            df = read_table(p, name)
            sheets.append({"sheet_name": name, **table_structure(df)})
        return {"sheets": sheets}
    df = read_table(p)
    return {"table": table_structure(df)}
