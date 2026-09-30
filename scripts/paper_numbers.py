import argparse
import json
from pathlib import Path

import pandas as pd

from evidence_harmonization.analysis import (
    arm_metrics,
    by_case,
    contract_compliance,
    contradictory_approvals,
    paired_contrast,
    ucr_intersection,
)

MODELS = ("gemma4", "qwen3")
ARMS = ("hidden", "visible", "contract")


def fraction(pair: list[int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def interface_section(results: Path, cases: dict, oracle_path: Path) -> dict:
    oracle = by_case(oracle_path)
    out: dict = {"arms": {}, "contrasts": {}, "ucr_intersection": {}, "contract": {}, "contradictory_approvals": {}}
    frames = {}
    for model in MODELS:
        for arm in ARMS:
            path = results / "interface" / model / arm / "predictions.jsonl"
            frame, metrics = arm_metrics(path, oracle_path)
            frames[model, arm] = frame
            out["arms"][f"{model}_{arm}"] = metrics
        hidden = by_case(results / "interface" / model / "hidden" / "predictions.jsonl")
        visible = by_case(results / "interface" / model / "visible" / "predictions.jsonl")
        contract = by_case(results / "interface" / model / "contract" / "predictions.jsonl")
        out["ucr_intersection"][model] = ucr_intersection(cases, oracle, hidden, visible)
        out["contract"][model] = contract_compliance(contract)
        out["contradictory_approvals"][model] = {
            arm: contradictory_approvals(cases, oracle, rows)
            for arm, rows in (("hidden", hidden), ("visible", visible), ("contract", contract))
        }
        out["contrasts"][model] = {
            metric: paired_contrast(frames[model, "hidden"], frames[model, "visible"], metric)
            for metric in ("case_success", "action_correct", "label_correct")
        }
    reference = results / "deterministic_reference" / "predictions.jsonl"
    if reference.exists():
        out["arms"]["deterministic_reference"] = arm_metrics(reference, oracle_path)[1]
    return out


def downstream_section(results: Path) -> dict:
    out: dict = {}
    summary = results / "downstream" / "summary.csv"
    if summary.exists():
        frame = pd.read_csv(summary)
        rows = {}
        for study in ("COBRE", "FBIRN"):
            part = frame[frame.study == study]
            clean = float(part[(part.family == "row_order") & (part.rate == 0.0)].altered_auc_mean.iloc[0])
            row_loss = abs(float(part[(part.family == "row_order") & (part.rate == 0.5)].auc_delta_mean.iloc[0]))
            flip = float(part[(part.family == "diagnosis_inversion") & (part.rate == 0.5)].auc_delta_mean.iloc[0])
            controls = {
                family: float(part[part.family == family].auc_delta_mean.iloc[0])
                for family in ("global_diagnosis_inversion_control", "global_sex_recode_control")
            }
            rows[study] = {
                "clean_auc": clean,
                "row_order_50_delta": -row_loss,
                "row_order_fraction_of_gap": row_loss / (clean - 0.5),
                "diagnosis_flip_50_delta": flip,
                **controls,
            }
        out["sfnc_sensitivity"] = rows
    bundles = results / "bundle_downstream" / "downstream_bundle_metrics.csv"
    if bundles.exists():
        frame = pd.read_csv(bundles)
        out["bundles"] = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    return out


def report(numbers: dict) -> str:
    lines = ["| Arm | Label | Action | Case | Unsafe | Commit | Label|commit | Action|commit | Repair |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m in numbers["interface"]["arms"].items():
        lines.append(
            f"| {name} | {m['label']:.3f} | {m['action']:.3f} | {m['case']:.3f} | {m['unsafe']:.3f} | {m['coverage']:.3f} | "
            f"{fraction(m['label_given_commit'])} | {m['action_given_commit'][0] / m['action_given_commit'][1]:.3f} | {fraction(m['exact_repair_on_targets'])} |"
        )
    for model in MODELS:
        ucr = numbers["interface"]["ucr_intersection"][model]
        case = numbers["interface"]["contrasts"][model]["case_success"]
        contract = numbers["interface"]["contract"][model]
        approvals = numbers["interface"]["contradictory_approvals"][model]
        lines.append("")
        lines.append(f"{model}: UCRcond hidden {ucr['hidden']}/{ucr['n']}, visible {ucr['visible']}/{ucr['n']}")
        lines.append(
            f"{model}: case success visible-hidden {case['estimate']:+.3f} "
            f"(95% CI [{case['ci_low']:.3f}, {case['ci_high']:.3f}], {case['n_clusters']} lineage families; McNemar p={case['mcnemar']['exact_p']:.2g})"
        )
        lines.append(
            f"{model}: contract parse {contract['parse_success']}/{contract['n']}, boolean present {contract['boolean_present']}/{contract['n']}, "
            f"label missing {contract['label_missing']}/{contract['n']}"
        )
        lines.append(
            f"{model}: contradictory approvals visible {approvals['visible']['count']}/{approvals['visible']['n']}, "
            f"contract {approvals['contract']['count']}/{approvals['contract']['n']}"
        )
    cec = numbers.get("cec")
    if cec:
        lines.append("")
        for name, result in cec["systems"].items():
            lines.append(
                f"CEC {name}: {result['all']['cec']:.3f} on {result['all']['n']} pairs, "
                f"{result['mechanistic_only']['cec']:.3f} on {result['mechanistic_only']['n']} mechanistic, "
                f"original-side {result['all']['original_side']:.3f}"
            )
    for study, row in numbers["downstream"].get("sfnc_sensitivity", {}).items():
        lines.append(
            f"{study}: clean AUC {row['clean_auc']:.3f}, 50% row-order {row['row_order_50_delta']:+.3f} "
            f"({100 * row['row_order_fraction_of_gap']:.1f}% of gap), 50% diagnosis flip {row['diagnosis_flip_50_delta']:+.3f}, "
            f"null controls {row['global_diagnosis_inversion_control']:+.3f}/{row['global_sex_recode_control']:+.3f}"
        )
    for row in numbers["downstream"].get("bundles", []):
        lines.append(
            f"bundle {row['bundle_name']} {row['cohort']}: diagnosis {row['diagnosis_labeled_n']}/{row['n_rows']}, "
            f"agreement {row['diagnosis_agreement_n']}, AUC {row['cross_validated_auc'] if row['cross_validated_auc'] is None else round(row['cross_validated_auc'], 3)}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute every number reported in the paper from run outputs.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--cases", type=Path, default=Path("data/real_panel/cases.jsonl"))
    parser.add_argument("--oracle", type=Path, default=Path("data/real_panel/oracle.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_numbers.json"))
    args = parser.parse_args()
    numbers = {
        "interface": interface_section(args.results, by_case(args.cases), args.oracle),
        "downstream": downstream_section(args.results),
    }
    cec = args.results / "cec" / "summary.json"
    if cec.exists():
        numbers["cec"] = json.loads(cec.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    print(report(numbers))


if __name__ == "__main__":
    main()
