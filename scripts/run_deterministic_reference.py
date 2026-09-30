import argparse
from pathlib import Path

from evidence_harmonization.baselines import run_deterministic_reference
from evidence_harmonization.evaluate import evaluate_predictions, write_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and score the in-domain deterministic reference.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = args.output / "predictions.jsonl"
    run_deterministic_reference(args.cases, predictions)
    frame, summary = evaluate_predictions(predictions, args.oracle)
    write_evaluation(frame, summary, args.output)


if __name__ == "__main__":
    main()
