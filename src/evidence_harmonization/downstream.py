from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .matlab import load_sfnc
from .paths import data_path


@dataclass(frozen=True)
class CohortData:
    study: str
    features: np.ndarray
    diagnosis: np.ndarray
    age: np.ndarray
    sex: np.ndarray
    site: np.ndarray


def load_cohort(study: str, sfnc_path: str | Path) -> CohortData:
    bundle = load_sfnc(sfnc_path, study)
    fields = list(bundle.field_names)
    diagnosis_idx = next(i for i, x in enumerate(fields) if x.startswith("diagnosis("))
    age_idx = fields.index("age")
    sex_idx = fields.index("gender(1:male; 2:female)")
    site = (
        bundle.scores[fields.index("Site")].astype(float)
        if "Site" in fields
        else np.ones(bundle.scores.shape[1], dtype=float)
    )
    tri = np.tril_indices(bundle.sfnc.shape[0], k=-1)
    features = bundle.sfnc[tri[0], tri[1], :].T
    diagnosis = bundle.scores[diagnosis_idx].astype(float)
    age = bundle.scores[age_idx].astype(float)
    sex = bundle.scores[sex_idx].astype(float)
    arrays = {
        "features": features,
        "diagnosis": diagnosis,
        "age": age,
        "sex": sex,
        "site": site,
    }
    if features.ndim != 2 or any(
        len(value) != len(diagnosis) for value in arrays.values()
    ):
        raise ValueError(f"{study} feature/metadata row counts differ")
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{study} {name} contains non-finite values")
    if set(np.unique(diagnosis)) != {1.0, 2.0}:
        raise ValueError(f"{study} analysis diagnosis is not binary 1/2")
    if set(np.unique(sex)) != {1.0, 2.0}:
        raise ValueError(f"{study} analysis sex is not binary 1/2")
    return CohortData(
        study=study,
        features=features,
        diagnosis=diagnosis,
        age=age,
        sex=sex,
        site=site,
    )


def binary_sz(diagnosis: np.ndarray) -> np.ndarray:
    return (np.asarray(diagnosis) == 1).astype(int)


def standardized_mean_difference(features: np.ndarray, y: np.ndarray) -> np.ndarray:
    x1 = features[y == 1]
    x0 = features[y == 0]
    n1, n0 = len(x1), len(x0)
    var1 = x1.var(axis=0, ddof=1)
    var0 = x0.var(axis=0, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
    return np.divide(
        x1.mean(axis=0) - x0.mean(axis=0),
        pooled,
        out=np.zeros(features.shape[1], dtype=float),
        where=pooled > 0,
    )


def adjusted_diagnosis_coefficient(
    features: np.ndarray,
    y: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    site: np.ndarray,
) -> np.ndarray:
    age_z = (age - np.nanmean(age)) / np.nanstd(age)
    sex_z = (sex - np.nanmean(sex)) / np.nanstd(sex)
    site_values = np.unique(site[np.isfinite(site)])
    site_cols = [
        (site == value).astype(float)
        for value in site_values[1:]
    ]
    design = np.column_stack(
        [
            np.ones(len(y)),
            y.astype(float),
            np.nan_to_num(age_z),
            np.nan_to_num(sex_z),
            *site_cols,
        ]
    )
    beta = np.linalg.pinv(design) @ features
    return beta[1]


def cross_validated_auc(
    features: np.ndarray, y: np.ndarray, seed: int, folds: int = 5
) -> float:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.full(len(y), np.nan)
    for train, test in splitter.split(features, y):
        k = min(200, features.shape[1])
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(features[train], y[train])
        predictions[test] = model.predict_proba(features[test])[:, 1]
    return float(roc_auc_score(y, predictions))


def corrupt_joint_row_order(
    cohort: CohortData, rate: float, seed: int
) -> tuple[CohortData, int]:
    rng = np.random.default_rng(seed)
    n = len(cohort.diagnosis)
    count = int(round(rate * n))
    chosen = np.sort(rng.choice(n, size=count, replace=False)) if count else np.array([], int)
    permutation = chosen.copy()
    if len(permutation) > 1:
        permutation = np.roll(permutation, 1)
    diagnosis = cohort.diagnosis.copy()
    age = cohort.age.copy()
    sex = cohort.sex.copy()
    site = cohort.site.copy()
    if len(chosen) > 1:
        diagnosis[chosen] = cohort.diagnosis[permutation]
        age[chosen] = cohort.age[permutation]
        sex[chosen] = cohort.sex[permutation]
        site[chosen] = cohort.site[permutation]
    mismatch = int(np.sum(chosen != permutation))
    return CohortData(cohort.study, cohort.features, diagnosis, age, sex, site), mismatch


def partial_diagnosis_inversion(
    cohort: CohortData, rate: float, seed: int
) -> tuple[CohortData, int]:
    rng = np.random.default_rng(seed)
    n = len(cohort.diagnosis)
    count = int(round(rate * n))
    chosen = np.sort(rng.choice(n, size=count, replace=False)) if count else np.array([], int)
    diagnosis = cohort.diagnosis.copy()
    diagnosis[chosen] = 3 - diagnosis[chosen]
    return CohortData(
        cohort.study, cohort.features, diagnosis, cohort.age, cohort.sex, cohort.site
    ), len(chosen)


def global_sex_inversion(cohort: CohortData) -> CohortData:
    return CohortData(
        cohort.study,
        cohort.features,
        cohort.diagnosis,
        cohort.age,
        3 - cohort.sex,
        cohort.site,
    )


def effect_comparison(clean: np.ndarray, altered: np.ndarray, top_k: int = 50) -> dict[str, float]:
    corr = float(np.corrcoef(clean, altered)[0, 1])
    rmse = float(np.sqrt(np.mean((clean - altered) ** 2)))
    valid = (np.abs(clean) > 1e-12) | (np.abs(altered) > 1e-12)
    sign_flip = float(np.mean(np.sign(clean[valid]) != np.sign(altered[valid])))
    k = min(top_k, len(clean))
    top_clean = set(np.argpartition(np.abs(clean), -k)[-k:])
    top_alt = set(np.argpartition(np.abs(altered), -k)[-k:])
    jaccard = len(top_clean & top_alt) / len(top_clean | top_alt)
    return {
        "effect_correlation": corr,
        "effect_rmse": rmse,
        "effect_sign_flip_rate": sign_flip,
        "top_k_jaccard": float(jaccard),
    }


def run_sensitivity(
    cohorts: list[CohortData],
    rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50),
    seeds: tuple[int, ...] = (11, 29, 47, 61, 83),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        clean_y = binary_sz(cohort.diagnosis)
        clean_effect = standardized_mean_difference(cohort.features, clean_y)
        clean_adjusted = adjusted_diagnosis_coefficient(
            cohort.features, clean_y, cohort.age, cohort.sex, cohort.site
        )
        auc_cache = {
            seed: cross_validated_auc(cohort.features, clean_y, seed)
            for seed in seeds
        }
        for family in ("row_order", "diagnosis_inversion"):
            for rate in rates:
                for seed in seeds:
                    if family == "row_order":
                        altered, changed = corrupt_joint_row_order(cohort, rate, seed)
                    else:
                        altered, changed = partial_diagnosis_inversion(cohort, rate, seed)
                    y = binary_sz(altered.diagnosis)
                    effect = standardized_mean_difference(altered.features, y)
                    adjusted = adjusted_diagnosis_coefficient(
                        altered.features, y, altered.age, altered.sex, altered.site
                    )
                    row = {
                        "study": cohort.study,
                        "family": family,
                        "rate": rate,
                        "seed": seed,
                        "n_subjects": len(y),
                        "changed_n": changed,
                        "clean_auc": auc_cache[seed],
                        "altered_auc": cross_validated_auc(cohort.features, y, seed),
                    }
                    row["auc_delta"] = row["altered_auc"] - row["clean_auc"]
                    row.update(effect_comparison(clean_effect, effect))
                    adjusted_metrics = effect_comparison(clean_adjusted, adjusted)
                    row.update({f"adjusted_{k}": v for k, v in adjusted_metrics.items()})
                    rows.append(row)

        for seed in seeds:
            diagnosis_inverted = CohortData(
                cohort.study,
                cohort.features,
                3 - cohort.diagnosis,
                cohort.age,
                cohort.sex,
                cohort.site,
            )
            y_inv = binary_sz(diagnosis_inverted.diagnosis)
            inv_effect = standardized_mean_difference(cohort.features, y_inv)
            inv_auc = cross_validated_auc(cohort.features, y_inv, seed)
            row = {
                "study": cohort.study,
                "family": "global_diagnosis_inversion_control",
                "rate": 1.0,
                "seed": seed,
                "n_subjects": len(clean_y),
                "changed_n": len(clean_y),
                "clean_auc": auc_cache[seed],
                "altered_auc": inv_auc,
                "auc_delta": inv_auc - auc_cache[seed],
            }
            row.update(effect_comparison(clean_effect, inv_effect))
            row.update(
                {
                    f"adjusted_{k}": v
                    for k, v in effect_comparison(
                        clean_adjusted,
                        adjusted_diagnosis_coefficient(
                            cohort.features,
                            y_inv,
                            cohort.age,
                            cohort.sex,
                            cohort.site,
                        ),
                    ).items()
                }
            )
            rows.append(row)

            sex_inverted = global_sex_inversion(cohort)
            sex_adjusted = adjusted_diagnosis_coefficient(
                cohort.features,
                clean_y,
                sex_inverted.age,
                sex_inverted.sex,
                sex_inverted.site,
            )
            row = {
                "study": cohort.study,
                "family": "global_sex_recode_control",
                "rate": 1.0,
                "seed": seed,
                "n_subjects": len(clean_y),
                "changed_n": len(clean_y),
                "clean_auc": auc_cache[seed],
                "altered_auc": auc_cache[seed],
                "auc_delta": 0.0,
            }
            row.update(effect_comparison(clean_effect, clean_effect))
            row.update(
                {
                    f"adjusted_{k}": v
                    for k, v in effect_comparison(clean_adjusted, sex_adjusted).items()
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "effect_correlation",
        "effect_rmse",
        "effect_sign_flip_rate",
        "top_k_jaccard",
        "adjusted_effect_correlation",
        "adjusted_effect_rmse",
        "altered_auc",
        "auc_delta",
    ]
    grouped = frame.groupby(["study", "family", "rate"], dropna=False)
    rows = []
    for keys, part in grouped:
        row = {
            "study": keys[0],
            "family": keys[1],
            "rate": keys[2],
            "n_replicates": len(part),
            "n_subjects": int(part["n_subjects"].iloc[0]),
            "mean_changed_n": float(part["changed_n"].mean()),
        }
        for metric in metrics:
            values = part[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


SFNC_FILES = {"COBRE": "Results/SFNC/COBRE/COBRE.mat", "FBIRN": "Results/SFNC/FBIRN/FBIRN.mat"}


def evaluate_bundle(bundle_dir: str | Path, seed: int = 101) -> pd.DataFrame:
    frame = pd.read_csv(Path(bundle_dir) / "canonical_subject_bundle.csv")
    rows = []
    for study, relative in SFNC_FILES.items():
        part = frame[frame["cohort"] == study].reset_index(drop=True)
        bundle = load_sfnc(data_path(relative), study)
        tri = np.tril_indices(bundle.sfnc.shape[0], k=-1)
        features = bundle.sfnc[tri[0], tri[1], :].T
        index = next(i for i, field in enumerate(bundle.field_names) if field.startswith("diagnosis("))
        reference = np.where(bundle.scores[index].astype(float) == 1, "SZ", "HC")
        if len(part) != len(features):
            raise ValueError(f"{study}: bundle/sFNC row count differs")
        labels = part["diagnosis"].astype(str)
        labeled = labels.isin({"SZ", "HC"}).to_numpy()
        agreement = np.full(len(part), np.nan)
        agreement[labeled] = labels[labeled].to_numpy() == reference[labeled]
        y = (labels[labeled].to_numpy() == "SZ").astype(int)
        auc = float("nan")
        if labeled.sum() >= 10 and len(np.unique(y)) == 2:
            auc = cross_validated_auc(features[labeled], y, seed=seed)
        rows.append({
            "cohort": study,
            "n_rows": len(part),
            "diagnosis_labeled_n": int(labeled.sum()),
            "diagnosis_coverage": float(labeled.mean()),
            "sz_n": int((labels == "SZ").sum()),
            "hc_n": int((labels == "HC").sum()),
            "diagnosis_agreement_n": int(np.nansum(agreement)),
            "diagnosis_disagreement_n": int(np.sum(labeled) - np.nansum(agreement)),
            "agreement_with_analysis_reference": float(np.nanmean(agreement)) if labeled.any() else float("nan"),
            "cross_validated_auc": auc,
        })
    return pd.DataFrame(rows)
