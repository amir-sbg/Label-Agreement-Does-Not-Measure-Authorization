from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .controller import (
    COLUMN_LABELS,
    FILE_LABELS,
    fallback_decision,
    gate_violation,
    normalize_decision,
)
from .environment import CaseEnvironment
from .llm import extract_json, read_jsonl, stable_seed

INTERFACES = ("hidden", "visible", "contract")

REDACT_KEYS = {
    "proposed_primary_role",
    "proposed_concept",
    "component_hints",
}

Generator = Callable[[str, int], str]


def redact_provisional(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: redact_provisional(item)
            for key, item in value.items()
            if key not in REDACT_KEYS
        }
    if isinstance(value, list):
        return [redact_provisional(item) for item in value]
    return value


def evidence_card(case: dict[str, Any], include_proposal: bool = False) -> dict[str, Any]:
    views = case.get("evidence_views", {})
    card = {
        "initial_observation": redact_provisional(case.get("initial_observation", {})),
        "views": {
            name: redact_provisional(views[name])
            for name in ("names_only", "context", "structure", "full")
            if name in views
        },
        "available_tools": sorted(case.get("tools", {})),
    }
    if include_proposal:
        observation = case.get("initial_observation", {})
        proposal = {}
        if case["task"] == "T-FILE":
            proposal["primary_role"] = observation.get("proposed_primary_role")
        elif case["task"] == "T-COL":
            proposal["concept"] = observation.get("proposed_concept")
        if proposal:
            card["proposal_under_review"] = proposal
    return card


def allowed_labels(case: dict[str, Any]) -> list[str]:
    task = case["task"]
    if task == "T-FILE":
        return FILE_LABELS
    if task == "T-COL":
        return COLUMN_LABELS
    if task == "T-VAL":
        return ["value_encoding"]
    if task == "T-ID":
        return [case.get("initial_observation", {}).get("id_scope", "imaging_row_alignment")]
    return [case.get("initial_observation", {}).get("governance_scope", "governance")]


def output_contract(case: dict[str, Any], contract_check: bool = False) -> str:
    labels = json.dumps(allowed_labels(case))
    consistency_field = (
        ',"proposal_matches_my_label":true|false'
        if contract_check else ""
    )
    return f"""Return exactly one JSON object and no markdown.
For a tool request use {{\"type\":\"tool\",\"tool_name\":\"one listed tool\"}}.
For a candidate use:
{{\"type\":\"candidate\",\"action\":\"recover|accept_provisionally|abstain|escalate\",\"label\":\"allowed label or null\",\"repair\":null,\"evidence_ids\":[\"view:structure\"],\"missing_evidence\":[\"short item\"],\"unsupported_claims\":[]{consistency_field}}}
Allowed labels for this task: {labels}.
Never emit placeholder words such as LABEL, ACTION, CODE, or AVAILABLE_TOOL.
Do not output a probability or self-reported confidence. The external evaluator measures uncertainty.
"""


def generator_prompt(
    case: dict[str, Any],
    env: CaseEnvironment,
    history: list[dict[str, Any]],
    proposal_visible: bool = False,
    contract_check: bool = False,
) -> str:
    card = evidence_card(case, include_proposal=proposal_visible)
    proposal_instruction = ""
    if proposal_visible:
        proposal_instruction = """
The evidence card includes a proposal_under_review. First infer the most supported label from the
evidence, then compare it with that proposal. If the proposal does not match your label, the terminal
action must be recover and the repair must contain your label. Do not accept a proposal merely because
it is present in the record.
"""
    if contract_check:
        proposal_instruction += """
For this contract ablation, every candidate must include proposal_matches_my_label as a boolean. Set it
to true only when the proposal exactly matches your inferred label; set it to false otherwise. When it
is false, action must be recover.
"""
    evidence_view_note = (
        "provisional target-like hints are included only as proposal_under_review"
        if proposal_visible
        else "provisional target-like hints are redacted"
    )
    return f"""You are the candidate-generation agent in a neuroimaging metadata harmonization pipeline.
You are not the final authority. Use only the privacy-safe evidence card and bounded tools below.
File role and column concept are separate tasks. A mixed role is legitimate.
Do not infer code direction, units, construct equivalence, subject identity, source independence,
or row alignment from a plausible name. Treat any missing documentation as a reason to abstain
or escalate. Request a tool when it can resolve a concrete uncertainty.
{proposal_instruction}

Task: {case['task']}
Study: {case['study']}
Mode: {case.get('mode', 'v2')}
Evidence card ({evidence_view_note}):
{json.dumps(card, sort_keys=True)}
Current bounded-tool observation:
{json.dumps({'available_tools': sorted(env.case.get('tools', {})), 'calls_remaining': env.trajectory.calls_remaining}, sort_keys=True)}
Prior trajectory:
{json.dumps(history, sort_keys=True)}

{output_contract(case, contract_check=contract_check)}
Choose one next tool call or one candidate decision. Prefer a concrete evidence request over guessing.
"""


def parse_output(raw: str) -> tuple[dict[str, Any] | None, bool]:
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return None, False
    if isinstance(obj.get("candidate"), dict):
        obj = {"type": "candidate", **obj["candidate"]}
    tool_name = obj.get("tool_name") or obj.get("tool")
    kind = str(obj.get("type") or obj.get("kind") or "candidate").lower()
    if tool_name or kind in {"tool", "tool_call", "call_tool"}:
        return {"type": "tool", "tool_name": str(tool_name or "")}, True
    return {
        "type": "candidate",
        "action": str(obj.get("action") or "").lower(),
        "label": obj.get("label") or obj.get("primary_label") or obj.get("primary_candidate"),
        "repair": obj.get("repair"),
        "evidence_ids": obj.get("evidence_ids") or [],
        "missing_evidence": obj.get("missing_evidence") or [],
        "unsupported_claims": obj.get("unsupported_claims") or [],
        "proposal_matches_my_label": obj.get("proposal_matches_my_label"),
        "unresolved": obj.get("unresolved") or obj.get("uncertainty_reasons") or [],
    }, True


def normalize_candidate(
    case: dict[str, Any], parsed: dict[str, Any], called: list[str]
) -> dict[str, Any]:
    decision = normalize_decision(case, parsed, called)
    if case["task"] in {"T-FILE", "T-COL"} and decision["action"] in {
        "recover", "accept_provisionally"
    } and decision["label"] not in allowed_labels(case):
        decision["action"] = "abstain"
        decision["label"] = None
        decision["repair"] = None
        decision["unresolved"] = list(decision.get("unresolved", [])) + [
            "model emitted a label outside the controlled ontology"
        ]
    decision["proposal_matches_my_label"] = parsed.get("proposal_matches_my_label")
    return decision


def run_generator(
    case: dict[str, Any],
    generator: Generator,
    seed: int,
    budget: int,
    proposal_visible: bool = False,
    contract_check: bool = False,
) -> dict[str, Any]:
    env = CaseEnvironment(case, budget)
    history: list[dict[str, Any]] = []
    raw_outputs: list[str] = []
    called: list[str] = []
    observed: list[str] = []
    parse_success = True
    parsed: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    for turn in range(budget + 2):
        prompt = generator_prompt(
            case, env, history,
            proposal_visible=proposal_visible,
            contract_check=contract_check,
        )
        raw = generator(prompt, seed + turn)
        raw_outputs.append(raw)
        parsed, ok = parse_output(raw)
        parse_success = parse_success and ok
        if not ok or parsed is None:
            history.append({
                "type": "controller",
                "status": "invalid_json",
                "instruction": "Return one valid JSON object using the exact schema.",
            })
            continue
        if parsed["type"] == "tool":
            tool_name = parsed["tool_name"]
            if env.trajectory.calls_remaining <= 0:
                history.append({"type": "controller", "status": "budget_exhausted"})
                break
            if tool_name in called:
                history.append({"type": "controller", "status": "duplicate_tool", "tool_name": tool_name})
                continue
            result = env.call(tool_name)
            called.append(tool_name)
            if result["status"] == "ok":
                observed.append(tool_name)
            history.append({"type": "tool_result", **result})
            continue
        candidate = normalize_candidate(case, parsed, observed)
        violation = gate_violation(case, candidate, observed)
        if violation:
            history.append({"type": "evidence_gate", "status": "rejected", "reason": violation})
            candidate = None
            continue
        break
    if candidate is None:
        candidate = fallback_decision(
            case, observed, "candidate generation or evidence gate did not resolve the case"
        )
    env.terminate(
        candidate["action"], candidate["label"], candidate["repair"],
        candidate["evidence_tools"], candidate.get("unresolved", []),
    )
    return {
        "decision": candidate,
        "called_tools": called,
        "observed_tools": observed,
        "parse_success": parse_success,
        "raw_outputs": raw_outputs,
        "trajectory": env.trajectory.steps,
        "controller_history": history,
        "candidate_evidence_ids": parsed.get("evidence_ids", []) if parsed else [],
        "candidate_missing_evidence": parsed.get("missing_evidence", []) if parsed else [],
    }


def run_case(
    case: dict[str, Any],
    generator: Generator,
    interface: str,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    if interface not in INTERFACES:
        raise ValueError(f"Unknown interface: {interface}")
    generated = run_generator(
        case,
        generator,
        seed,
        budget,
        proposal_visible=interface in {"visible", "contract"},
        contract_check=interface == "contract",
    )
    called = sorted(set(generated["called_tools"]))
    observed = sorted(set(generated["observed_tools"]))
    decision = dict(generated["decision"])
    final_gate = gate_violation(case, decision, observed)
    if final_gate and decision["action"] in {"recover", "accept_provisionally"}:
        decision = {
            "action": "abstain",
            "label": None,
            "repair": None,
            "evidence_tools": called,
            "unresolved": [final_gate],
        }
    return {
        "case_id": case["case_id"],
        "decision": decision,
        "called_tools": called,
        "observed_tools": observed,
        "parse_success": generated["parse_success"],
        "raw_outputs": generated["raw_outputs"],
        "trajectory": generated["trajectory"],
        "controller_history": generated["controller_history"],
        "candidate_evidence_ids": generated["candidate_evidence_ids"],
        "candidate_missing_evidence": generated["candidate_missing_evidence"],
        "condition": interface,
    }


def run_cases(
    cases_path: str | Path,
    output_path: str | Path,
    generator: Any,
    interface: str,
    model_name: str,
    shard_id: int = 0,
    num_shards: int = 1,
    budget: int = 4,
    run_seed: int = 0,
    limit: int | None = None,
) -> None:
    cases = [
        case for index, case in enumerate(read_jsonl(cases_path))
        if index % num_shards == shard_id
    ]
    if limit is not None:
        cases = cases[:limit]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for case in cases:
            if hasattr(generator, "reset_counters"):
                generator.reset_counters()
            started = time.perf_counter()
            result = run_case(
                case, generator, interface,
                stable_seed(case["case_id"], run_seed), budget,
            )
            result.update({
                "model_path": model_name,
                "run_seed": run_seed,
                "shard_id": shard_id,
                "inference_seconds": time.perf_counter() - started,
                "input_tokens": getattr(generator, "input_tokens_total", 0),
                "output_tokens": getattr(generator, "output_tokens_total", 0),
            })
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
