import numpy as np
import pandas as pd

from evidence_harmonization.analysis import contradictory_approvals, ucr_intersection
from evidence_harmonization.stats import cluster_robust_ci, mcnemar_exact


def test_cluster_robust_ci_matches_formula():
    diff = pd.Series([1, 0, 1, 1, 0, 1, 0, 0], dtype=float)
    groups = pd.Series(["a", "a", "b", "b", "c", "c", "d", "d"])
    out = cluster_robust_ci(diff, groups)
    scores = (diff - diff.mean()).groupby(groups).sum().to_numpy()
    se = np.sqrt((4 / 3) * np.sum(scores**2) / 64)
    assert np.isclose(out["se"], se) and out["df"] == 3
    assert out["ci_low"] < out["estimate"] < out["ci_high"]


def test_mcnemar_counts_discordant_pairs():
    out = mcnemar_exact(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 1, 1]))
    assert out["left_only"] == 1 and out["right_only"] == 2


def semantic(case_id, proposed):
    return {"case_id": case_id, "task": "T-COL", "initial_observation": {"proposed_concept": proposed}}


def test_ucr_and_contradictory_approval():
    cases = {"a": semantic("a", "diagnosis"), "b": semantic("b", "diagnosis")}
    oracle = {k: {"oracle": {"corruption_present": True, "correct_action": "recover", "correct_label": "sex_or_gender"}} for k in cases}
    hidden = {k: {"decision": {"action": "accept_provisionally", "label": "sex_or_gender"}} for k in cases}
    visible = {"a": {"decision": {"action": "recover", "label": "sex_or_gender"}}, "b": {"decision": {"action": "accept_provisionally", "label": "sex_or_gender"}}}
    assert ucr_intersection(cases, oracle, hidden, visible) == {"n": 2, "hidden": 2, "visible": 1}
    assert contradictory_approvals(cases, oracle, visible) == {"count": 1, "n": 2}
