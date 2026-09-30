from conftest import encoding_case, imaging_case

from evidence_harmonization.controller import fallback_decision, gate_violation, normalize_decision


def test_commit_without_decisive_evidence_is_blocked():
    case = imaging_case()
    decision = normalize_decision(case, {"action": "accept_provisionally"}, [])
    assert gate_violation(case, decision, []) is not None
    assert gate_violation(case, decision, ["check_imaging_order"]) is None


def test_abstention_never_needs_evidence():
    case = imaging_case()
    decision = normalize_decision(case, {"action": "abstain"}, [])
    assert gate_violation(case, decision, []) is None


def test_governance_requires_escalation_tool():
    case = {"task": "T-GOV", "initial_observation": {"governance_scope": "cpz_conversion_policy"}}
    decision = normalize_decision(case, {"action": "escalate"}, [])
    assert gate_violation(case, decision, []) == "Call escalate_to_human before escalation."
    assert gate_violation(case, decision, ["escalate_to_human"]) is None
    assert fallback_decision(case, [], "x")["action"] == "escalate"


def test_row_order_recovery_uses_fixed_repair():
    case = imaging_case()
    decision = normalize_decision(case, {"action": "recover"}, ["check_imaging_order"])
    assert decision["repair"] == {"operation": "restore_analysis_id_order"}


def test_value_commit_needs_dictionary_or_redundant_source():
    case = encoding_case()
    decision = normalize_decision(case, {"action": "accept_provisionally"}, ["profile_values"])
    assert gate_violation(case, decision, ["profile_values"]) is not None
    assert gate_violation(case, decision, ["get_dictionary_entry"]) is None
