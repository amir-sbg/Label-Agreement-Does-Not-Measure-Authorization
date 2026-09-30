import argparse
from pathlib import Path

import pandas as pd

from evidence_harmonization.downstream import evaluate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnosis coverage, agreement, and sFNC AUC for materialized bundles.")
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if len(args.bundle) != len(args.name):
        raise SystemExit("--bundle and --name must be given the same number of times")
    frames = []
    for name, directory in zip(args.name, args.bundle):
        frame = evaluate_bundle(directory, args.seed)
        frame.insert(0, "bundle_name", name)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "downstream_bundle_metrics.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
