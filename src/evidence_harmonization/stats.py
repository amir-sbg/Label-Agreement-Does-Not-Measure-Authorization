from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t


def cluster_robust_ci(
    diff: pd.Series, groups: pd.Series, level: float = 0.95
) -> dict[str, Any]:
    diff = diff.astype(float).reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    estimate = float(diff.mean())
    scores = (diff - estimate).groupby(groups, sort=True).sum().to_numpy()
    g, n = len(scores), len(diff)
    se = float(np.sqrt((g / (g - 1)) * np.sum(scores**2) / n**2))
    critical = float(t.ppf(0.5 + level / 2, g - 1))
    return {
        "estimate": estimate,
        "n": n,
        "n_clusters": g,
        "df": g - 1,
        "se": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
    }


def mcnemar_exact(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    left = left.astype(bool).to_numpy()
    right = right.astype(bool).to_numpy()
    left_only = int((left & ~right).sum())
    right_only = int((~left & right).sum())
    total = left_only + right_only
    p = float(binomtest(min(left_only, right_only), total, 0.5).pvalue) if total else 1.0
    return {"left_only": left_only, "right_only": right_only, "exact_p": p}
