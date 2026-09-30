from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class AnalysisBundle:
    study: str
    analysis_ids: tuple[str, ...]
    field_names: tuple[str, ...]
    scores: np.ndarray
    sfnc: np.ndarray | None = None


def _decode_reference(f: h5py.File, ref: h5py.Reference) -> str:
    arr = np.asarray(f[ref]).squeeze()
    if arr.dtype.kind in "ui":
        return "".join(chr(int(x)) for x in arr.ravel() if int(x))
    if arr.dtype.kind in "SU":
        return "".join(str(x) for x in arr.ravel())
    return str(arr)


def read_cell_strings(f: h5py.File, key: str) -> tuple[str, ...]:
    refs = np.asarray(f[key]).ravel()
    return tuple(_decode_reference(f, ref) for ref in refs)


def load_subject_bridge(path: str | Path, study: str) -> AnalysisBundle:
    with h5py.File(path, "r") as f:
        ids = read_cell_strings(f, "analysis_ID")
        fields = read_cell_strings(f, "FILE_ID")
        scores = np.asarray(f["analysis_SCORE"], dtype=float)
    if scores.shape != (len(fields), len(ids)):
        raise ValueError(
            f"Bridge shape mismatch for {study}: {scores.shape}, "
            f"{len(fields)} fields, {len(ids)} IDs"
        )
    return AnalysisBundle(study, ids, fields, scores)


def load_sfnc(path: str | Path, study: str) -> AnalysisBundle:
    with h5py.File(path, "r") as f:
        ids = read_cell_strings(f, "analysis_ID")
        fields = read_cell_strings(f, "FILE_ID")
        scores = np.asarray(f["analysis_SCORE"], dtype=float)
        sfnc = np.asarray(f["sFNC"], dtype=float)
    if scores.shape != (len(fields), len(ids)):
        raise ValueError(
            f"sFNC score shape mismatch for {study}: {scores.shape}, "
            f"{len(fields)} fields, {len(ids)} IDs"
        )
    if (
        sfnc.ndim != 3
        or sfnc.shape[0] != sfnc.shape[1]
        or sfnc.shape[-1] != len(ids)
    ):
        raise ValueError(f"sFNC shape mismatch for {study}: {sfnc.shape}")
    return AnalysisBundle(study, ids, fields, scores, sfnc)


def root_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            if key == "#refs#":
                continue
            obj = f[key]
            rows.append(
                {
                    "name": key,
                    "shape": list(getattr(obj, "shape", [])),
                    "dtype": str(getattr(obj, "dtype", "group")),
                    "matlab_class": _attr_text(obj.attrs.get("MATLAB_class")),
                }
            )
    return rows


def _attr_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)
