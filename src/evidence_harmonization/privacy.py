from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

_ID_RE = re.compile(
    r"(^|[^a-z])(id|ursi|subject|subj|patient|participant|visit|site|"
    r"experiment|segment|assessment|asmt|doi|usrd)([^a-z]|$)",
    re.IGNORECASE,
)


def is_identifier_name(name: str) -> bool:
    text = str(name)
    if _ID_RE.search(text.replace("_", " ")):
        return True
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    exact = {"id", "doi", "ursi", "usrd", "subjectid", "subjid", "patientid", "participantid"}
    prefixes = ("subject", "subj", "patient", "participant", "visit", "site", "experiment", "segment", "assessment", "asmt")
    return compact in exact or (
        compact.endswith("id") and compact.startswith(prefixes)
    )


def normalized_identifier(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    if text.isdigit():
        return str(int(text))
    return text


def stable_digest(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def value_signature(series: pd.Series, name: str) -> dict[str, Any]:
    s = series
    n = int(len(s))
    missing = int(s.isna().sum())
    nonmissing = s.dropna()
    out: dict[str, Any] = {
        "n": n,
        "missing_n": missing,
        "missing_fraction": 0.0 if n == 0 else missing / n,
        "dtype": str(s.dtype),
        "unique_n": int(nonmissing.nunique(dropna=True)),
        "is_identifier_like": is_identifier_name(name),
    }
    if out["is_identifier_like"]:
        texts = [str(v).strip() for v in nonmissing]
        out["unique_fraction"] = 0.0 if not texts else len(set(texts)) / len(texts)
        out["length_counts"] = dict(sorted(Counter(map(len, texts)).items()))
        out["format_patterns"] = dict(
            Counter(
                re.sub(r"[A-Z]", "A", re.sub(r"[a-z]", "a", re.sub(r"[0-9]", "#", t)))
                for t in texts
            ).most_common(8)
        )
        return out

    numeric = pd.to_numeric(nonmissing, errors="coerce")
    numeric_fraction = 0.0 if len(nonmissing) == 0 else float(numeric.notna().mean())
    out["numeric_fraction"] = numeric_fraction
    if numeric_fraction >= 0.95 and numeric.notna().any():
        vals = numeric.dropna().astype(float)
        out["numeric_summary"] = {
            "min": float(vals.min()),
            "q25": float(vals.quantile(0.25)),
            "median": float(vals.median()),
            "q75": float(vals.quantile(0.75)),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        }

    unique_n = int(nonmissing.nunique(dropna=True))
    text_values = [str(v) for v in nonmissing]
    short_values = all(len(v) <= 80 for v in text_values[:1000])
    if unique_n <= 20 and short_values:
        counts = nonmissing.astype(str).value_counts(dropna=False).head(20)
        out["category_counts"] = {str(k): int(v) for k, v in counts.items()}
    return out


def safe_fingerprint(series: pd.Series, name: str) -> str:
    normalized = []
    for value in series.tolist():
        if pd.isna(value):
            normalized.append("<NA>")
        else:
            normalized.append(str(_safe_scalar(value)).strip())
    return stable_digest(name + "\n" + "\n".join(normalized), length=24)
