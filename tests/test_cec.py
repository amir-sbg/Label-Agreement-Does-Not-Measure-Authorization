from conftest import encoding_case, imaging_case

from evidence_harmonization.cec import evaluate_cec


def panel():
    cases = {c["case_id"]: c for c in (imaging_case(False), imaging_case(True), encoding_case())}
    oracle = {
        "id-False": {"oracle": {"correct_action": "recover", "correct_label": "imaging_row_alignment", "correct_repair": {"operation": "restore_analysis_id_order"}}},
        "id-True": {"oracle": {"correct_action": "accept_provisionally", "correct_label": "imaging_row_alignment", "correct_repair": None}},
        "val": {"oracle": {"correct_action": "accept_provisionally", "correct_label": "value_encoding", "correct_repair": None}},
    }
    return cases, oracle


def test_harness_floor_and_ceiling():
    summary = evaluate_cec(*panel())
    assert summary["n_pairs"] == 3
    assert summary["systems"]["deterministic_reference"]["all"]["cec"] == 1.0
    frozen = summary["systems"]["frozen_answer"]["all"]
    assert frozen["cec"] == 0.0 and frozen["original_side"] == 1.0
