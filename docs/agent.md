# Agent, tools, and controller

The agent is a bounded candidate generator. It never writes to data and is never the final authority: every proposal passes through a deterministic controller before anything is committed.

## Loop

For each case (`agent.run_case`):

1. The model receives the case's Evidence Card, the list of available tools, the remaining call budget, and the trajectory so far.
2. It returns exactly one JSON object: a tool request or a terminal candidate.
3. The controller answers tool requests from precomputed, read-only tool outputs. It skips duplicate calls, returns `unavailable` for tools the case does not offer (which still uses budget), and enforces the budget of 4 calls within 6 turns.
4. A candidate is normalized (`controller.normalize_decision`) and checked by the evidence gate (`controller.gate_violation`). A rejected candidate is returned to the model with the reason; if nothing passes, the controller abstains (or escalates for governance cases).

## Tools

| Tool | Returns | Required before committing |
|---|---|---|
| `reparse_artifact` | parser status, sheets, shape, column names | any T-FILE commit |
| `profile_values` | aggregate value signature (type, missingness, cardinality, summary) | T-COL commits to `diagnosis`, `sex_or_gender`, `current_age`, `age_at_onset`, `medication_dose` |
| `get_dictionary_entry` | parsed dictionary code map, units, allowed values | T-VAL commits (this or `compare_redundant_values`) |
| `compare_redundant_values` | code map verified against a redundant source | T-VAL commits (this or `get_dictionary_entry`) |
| `check_id_overlap` | normalized identifier overlap and relationship | T-ID `identifier_linkage` commits |
| `check_imaging_order` | metadata-to-sFNC row-order agreement | T-ID `imaging_row_alignment` commits |
| `check_lineage` | source lineage and independence status | T-ID `source_independence` commits |
| `escalate_to_human` | review routing | all T-GOV decisions (always escalate) |

Tools see no raw subject rows; identifier checks are computed in memory and exposed only as aggregates.

## Output contract

```json
{"type": "tool", "tool_name": "profile_values"}
{"type": "candidate", "action": "recover|accept_provisionally|abstain|escalate",
 "label": "<allowed label or null>", "repair": null,
 "evidence_ids": ["view:structure"], "missing_evidence": [], "unsupported_claims": []}
```

The contract arm adds a required `"proposal_matches_my_label": true|false`. A missing Boolean is not read as `false`; it routes to review.

## Interface arms

| Arm | Upstream proposal | Extra field |
|---|---|---|
| `hidden` | redacted from the card | none |
| `visible` | shown as `proposal_under_review` | none |
| `contract` | shown as `proposal_under_review` | `proposal_matches_my_label` |

The prompts are defined in `src/evidence_harmonization/agent.py` (`generator_prompt`, `output_contract`). Decoding is greedy (`do_sample=False`), and the seed is derived from the case ID (`llm.stable_seed`).

## Controls

- **Deterministic reference** (`baselines.deterministic_gate`): a rule-based system that shares derivation with the reference labels. It is an in-domain upper reference, not an independent baseline.
- **Frozen-answer adversary** (`cec.frozen_answer`): replays the original oracle answer and calls every tool. It scores 1.0 on original cases and 0.0 CEC.
- **Null perturbations** (`downstream.run_sensitivity`): global diagnosis inversion and sex recoding, both of which should leave ΔAUC at 0.
