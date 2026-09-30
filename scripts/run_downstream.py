import argparse
from pathlib import Path

from evidence_harmonization.downstream import load_cohort, run_sensitivity, summarize_sensitivity
from evidence_harmonization.paths import data_path

SEEDS = (11, 29, 47, 61, 83, 101, 127, 149, 173, 197, 223, 251, 277, 307, 337, 367, 397, 431, 463, 499)
SFNC = {"COBRE": "Results/SFNC/COBRE/COBRE.mat", "FBIRN": "Results/SFNC/FBIRN/FBIRN.mat"}


def main() -> None:
    parser = argparse.ArgumentParser(description="sFNC diagnosis-classification sensitivity to metadata corruption.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohorts = [load_cohort(study, data_path(path)) for study, path in SFNC.items()]
    raw = run_sensitivity(cohorts, seeds=tuple(args.seeds))
    raw.to_csv(args.output / "replicates.csv", index=False)
    summarize_sensitivity(raw).to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
