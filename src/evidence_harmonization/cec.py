from __future__ import annotations

import copy
from typing import Any, Callable

from .baselines import deterministic_gate

REL_ALT = {
    "left_complete_subset": "partial_overlap",
    "partial_overlap": "left_complete_subset",
    "left_complete_with_right_duplicates": "partial_overlap",
}

System = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def mutate(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str] | None:
    c = copy.deepcopy(case)
    obs, tools, task = c["initial_observation"], c["tools"], c["task"]

    if task == "T-ID" and obs.get("id_scope") is None:
        t = tools["check_imaging_order"]
        agree = not t["order_agreement"]
        t["order_agreement"] = agree
        t["mismatch_n"] = 0 if agree else 8
        t["mismatch_fraction"] = 0.0 if agree else 8 / t["imaging_n"]
        gold = ({"correct_action": "accept_provisionally", "correct_repair": None} if agree
                else {"correct_action": "recover",
                      "correct_repair": {"operation": "restore_analysis_id_order"}})
        return c, gold, "T-ID:imaging_order", "mechanistic"

    if task == "T-ID" and obs.get("id_scope") == "identifier_linkage":
        t = tools["check_id_overlap"]
        new = REL_ALT[t["verified_relationship"]]
        t["verified_relationship"] = new
        valid = obs.get("proposed_relationship") == new and obs.get("proposed_identity_proven") is False
        gold = ({"correct_action": "accept_provisionally", "correct_repair": None} if valid
                else {"correct_action": "recover",
                      "correct_repair": {"identity_proven": False, "relationship": new}})
        return c, gold, "T-ID:linkage", "mechanistic"

    if task == "T-ID" and obs.get("id_scope") == "source_independence":
        t = tools["check_lineage"]
        t["same_lineage"] = not t.get("same_lineage", True)
        new = ("nonindependent_same_lineage" if t["same_lineage"]
               else "independent_distinct_lineage")
        t["verified_independence_status"] = new
        valid = obs.get("proposed_independence_status") == new
        gold = ({"correct_action": "accept_provisionally", "correct_repair": None} if valid
                else {"correct_action": "recover", "correct_repair": {"independence_status": new}})
        return c, gold, "T-ID:lineage", "mechanistic (OOD value)"

    if task == "T-VAL":
        verifier = next((n for n in ("get_dictionary_entry", "compare_redundant_values")
                         if n in tools), None)
        if verifier is None:
            return None
        t = tools[verifier]
        keys = list(t["mapping"])
        t["mapping"] = {k: t["mapping"][keys[(i + 1) % len(keys)]]
                        for i, k in enumerate(keys)}
        valid = obs.get("proposed_mapping") == t["mapping"]
        gold = ({"correct_action": "accept_provisionally", "correct_repair": None} if valid
                else {"correct_action": "recover", "correct_repair": {"mapping": t["mapping"]}})
        return c, gold, "T-VAL:encoding", "mechanistic"

    if task == "T-FILE" and obs.get("parser_status") == "injected_parser_failure":
        t = tools["reparse_artifact"]
        t["status"] = "parser_failure"
        t["parser_provenance"] = "failed"
        t.pop("structure", None)
        return c, {"correct_action": "abstain", "correct_repair": None}, "T-FILE:parser", "policy-derived"

    return None


def frozen_answer(case: dict[str, Any], gold_original: dict[str, Any]) -> dict[str, Any]:
    called = [t for t in case["tools"]
              if t != "escalate_to_human" or case["task"] == "T-GOV"][:6]
    return {"decision": {"action": gold_original["correct_action"],
                         "label": gold_original["correct_label"],
                         "repair": gold_original["correct_repair"],
                         "evidence_tools": called, "unresolved": []},
            "called_tools": called, "observed_tools": called}


def deterministic_reference(case: dict[str, Any], _gold: dict[str, Any]) -> dict[str, Any]:
    return deterministic_gate(case)


SYSTEMS: dict[str, System] = {
    "deterministic_reference": deterministic_reference,
    "frozen_answer": frozen_answer,
}


def correct(pred: dict[str, Any], gold_action: str, gold_repair: Any) -> bool:
    d = pred["decision"]
    return d.get("action") == gold_action and d.get("repair") == gold_repair


def build_pairs(cases: dict[str, dict[str, Any]], oracle: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for case_id, case in cases.items():
        mutated = mutate(case)
        if mutated is None:
            continue
        mcase, mgold, family, strength = mutated
        pairs.append({
            "case_id": case_id,
            "original": case,
            "mutated": mcase,
            "original_gold": oracle[case_id]["oracle"],
            "mutated_gold": mgold,
            "family": family,
            "strength": strength,
        })
    return pairs


def score(pairs: list[dict[str, Any]], system: System) -> dict[str, Any]:
    cells = {"correct_correct": 0, "correct_wrong": 0, "wrong_correct": 0, "wrong_wrong": 0}
    for pair in pairs:
        gold = pair["original_gold"]
        o_ok = correct(system(pair["original"], gold), gold["correct_action"], gold["correct_repair"])
        mgold = pair["mutated_gold"]
        m_ok = correct(system(pair["mutated"], gold), mgold["correct_action"], mgold["correct_repair"])
        key = f"{'correct' if o_ok else 'wrong'}_{'correct' if m_ok else 'wrong'}"
        cells[key] += 1
    n = len(pairs)
    return {
        "n": n,
        "cec": cells["correct_correct"] / n if n else None,
        "original_side": (cells["correct_correct"] + cells["correct_wrong"]) / n if n else None,
        "mutated_side": (cells["correct_correct"] + cells["wrong_correct"]) / n if n else None,
        "paired": cells,
    }


def evaluate_cec(cases: dict[str, dict[str, Any]], oracle: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = build_pairs(cases, oracle)
    families: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        entry = families.setdefault(pair["family"], {"n": 0, "mutated_oracle_strength": pair["strength"]})
        entry["n"] += 1
    mechanistic = [p for p in pairs if p["strength"].startswith("mechanistic")]
    systems = {}
    for name, system in SYSTEMS.items():
        systems[name] = {
            "all": score(pairs, system),
            "mechanistic_only": score(mechanistic, system),
            "by_family": {
                family: score([p for p in pairs if p["family"] == family], system)
                for family in sorted(families)
            },
        }
    return {"n_pairs": len(pairs), "families": families, "systems": systems}
