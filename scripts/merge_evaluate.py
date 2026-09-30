import argparse
import json
from pathlib import Path

from evidence_harmonization.evaluate import evaluate_predictions, write_evaluation
from evidence_harmonization.merge import load_and_validate_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge agent shards and score them against the oracle.")
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()
    rows = load_and_validate_shards(args.shards, args.oracle, expected=args.expected)
    args.output.mkdir(parents=True, exist_ok=True)
    merged = args.output / "predictions.jsonl"
    merged.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    frame, summary = evaluate_predictions(merged, args.oracle)
    write_evaluation(frame, summary, args.output)


if __name__ == "__main__":
    main()
