import pandas as pd

from evidence_harmonization.privacy import normalized_identifier, value_signature


def test_identifier_signature_never_contains_values():
    series = pd.Series(["SUBJ-0001", "SUBJ-0002", "SUBJ-0003"])
    signature = value_signature(series, "SubjectID")
    text = str(signature)
    assert "SUBJ-0001" not in text
    assert signature["is_identifier_like"] is True
    assert signature["unique_n"] == 3


def test_numeric_identifier_normalization_removes_leading_zeros():
    assert normalized_identifier("000-123") == "123"


def test_small_nonidentifier_categories_are_aggregated():
    signature = value_signature(pd.Series(["SZ", "HC", "SZ"]), "Diagnosis")
    assert signature["category_counts"] == {"SZ": 2, "HC": 1}
