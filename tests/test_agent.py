import json

from conftest import imaging_case, read_jsonl

from evidence_harmonization.agent import evidence_card, generator_prompt, parse_output, run_case
from evidence_harmonization.environment import CaseEnvironment
from evidence_harmonization.evaluate import evaluate_predictions
from evidence_harmonization.llm import extract_json


def scripted(outputs):
    queue = iter(outputs)
    return lambda _prompt, _seed: next(queue)


def test_extract_json_from_fence():
    fence = chr(96) * 3
    assert extract_json(f'{fence}json\n{{"type":"tool","tool_name":"x"}}\n{fence}')["tool_name"] == "x"


def test_parse_output_tool_and_candidate():
    tool, ok = parse_output('{"type":"tool","tool_name":"profile_values"}')
    assert ok and tool == {"type": "tool", "tool_name": "profile_values"}
    candidate, ok = parse_output('{"candidate":{"action":"recover","label":"diagnosis","proposal_matches_my_label":false}}')
    assert ok and candidate["label"] == "diagnosis" and candidate["proposal_matches_my_label"] is False
    assert parse_output("no json here") == (None, False)


def test_gate_rejects_then_accepts_after_tool():
    outputs = [
        '{"type":"candidate","action":"recover","label":"imaging_row_alignment"}',
        '{"type":"tool","tool_name":"check_imaging_order"}',
        '{"type":"candidate","action":"recover","label":"imaging_row_alignment"}',
    ]
    result = run_case(imaging_case(), scripted(outputs), "hidden", 0, 4)
    assert [h["status"] for h in result["controller_history"] if h["type"] == "evidence_gate"] == ["rejected"]
    assert result["decision"]["repair"] == {"operation": "restore_analysis_id_order"}


def test_unavailable_tool_does_not_satisfy_gate():
    outputs = ['{"type":"tool","tool_name":"check_id_overlap"}'] + ['{"type":"candidate","action":"recover"}'] * 5
    result = run_case(imaging_case(), scripted(outputs), "hidden", 0, 1)
    assert result["observed_tools"] == []
    assert result["decision"]["action"] == "abstain"


def test_proposal_is_redacted_when_hidden(synthetic_panel):
    case = next(c for c in read_jsonl(synthetic_panel / "cases.jsonl") if c["task"] == "T-COL")
    hidden = json.dumps(evidence_card(case))
    visible = evidence_card(case, include_proposal=True)
    assert "proposed_concept" not in hidden
    assert visible["proposal_under_review"] == {"concept": case["initial_observation"]["proposed_concept"]}


def test_contract_field_only_in_contract_prompt(synthetic_panel):
    case = read_jsonl(synthetic_panel / "cases.jsonl")[0]
    env = CaseEnvironment(case, 4)
    assert "proposal_matches_my_label" not in generator_prompt(case, env, [], proposal_visible=True)
    assert "proposal_matches_my_label" in generator_prompt(case, env, [], proposal_visible=True, contract_check=True)


def test_oracle_following_agent_scores_perfectly(synthetic_panel, tmp_path):
    cases = [c for c in read_jsonl(synthetic_panel / "cases.jsonl") if c["task"] != "T-ID"]
    oracle = {row["case_id"]: row["oracle"] for row in read_jsonl(synthetic_panel / "oracle.jsonl")}
    rows = []
    for case in cases:
        gold = oracle[case["case_id"]]
        tools = [json.dumps({"type": "tool", "tool_name": t}) for t in gold["minimal_decisive_tool_sets"][0]]
        final = json.dumps({"type": "candidate", "action": gold["correct_action"], "label": gold["correct_label"], "repair": gold["correct_repair"]})
        rows.append(run_case(case, scripted(tools + [final]), "visible", 0, 4))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    subset = tmp_path / "oracle.jsonl"
    subset.write_text("".join(json.dumps(row) + "\n" for row in read_jsonl(synthetic_panel / "oracle.jsonl") if row["case_id"] in oracle and row["task"] != "T-ID"))
    frame, summary = evaluate_predictions(predictions, subset)
    assert frame["label_correct"].all() and frame["action_correct"].all()
    assert summary["unsafe_commit_rate"] == 0.0
