import argparse
import json
from pathlib import Path

from evidence_harmonization.analysis import by_case
from evidence_harmonization.cec import evaluate_cec


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled-error consistency (CEC) harness.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate_cec(by_case(args.cases), by_case(args.oracle))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for name, result in summary["systems"].items():
        full, mech = result["all"], result["mechanistic_only"]
        print(f"{name:24s} CEC {full['cec']:.3f} (n={full['n']})  mechanistic {mech['cec']:.3f} (n={mech['n']})  original-side {full['original_side']:.3f}")


if __name__ == "__main__":
    main()
