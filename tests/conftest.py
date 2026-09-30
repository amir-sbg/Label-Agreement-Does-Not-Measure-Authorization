import json

import pytest

from evidence_harmonization.synthetic import build


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def synthetic_panel(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    build(root, 12, 20260821)
    return root


def imaging_case(order_agreement=False, proposal_alignment=True):
    return {
        "case_id": f"id-{order_agreement}",
        "study": "COBRE",
        "task": "T-ID",
        "error_family": "partial_row_order_permutation",
        "base_artifact": "bridge",
        "lineage_group": "g1",
        "initial_observation": {"claimed_alignment": proposal_alignment},
        "tools": {
            "check_imaging_order": {
                "order_agreement": order_agreement,
                "mismatch_n": 0 if order_agreement else 8,
                "mismatch_fraction": 0.0 if order_agreement else 0.05,
                "imaging_n": 157,
            }
        },
    }


def encoding_case():
    return {
        "case_id": "val",
        "study": "FBIRN",
        "task": "T-VAL",
        "error_family": "clean_encoding",
        "base_artifact": "dx",
        "lineage_group": "g2",
        "initial_observation": {"proposed_mapping": {"1": "SZ", "2": "HC"}},
        "tools": {"get_dictionary_entry": {"mapping": {"1": "SZ", "2": "HC"}}},
    }
