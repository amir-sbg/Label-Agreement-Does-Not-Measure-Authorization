import numpy as np

from evidence_harmonization.downstream import (
    adjusted_diagnosis_coefficient,
    effect_comparison,
    standardized_mean_difference,
)


def test_global_label_inversion_reverses_effect_vector():
    x = np.array([[0.0], [1.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1])
    a = standardized_mean_difference(x, y)
    b = standardized_mean_difference(x, 1 - y)
    assert np.allclose(a, -b)


def test_global_binary_covariate_recode_preserves_adjusted_diagnosis_effect():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 8))
    y = np.array([0, 1] * 20)
    age = np.linspace(18, 60, 40)
    sex = np.array([1, 1, 2, 2] * 10)
    site = np.ones(40)
    a = adjusted_diagnosis_coefficient(x, y, age, sex, site)
    b = adjusted_diagnosis_coefficient(x, y, age, 3 - sex, site)
    assert np.allclose(a, b, atol=1e-10)
