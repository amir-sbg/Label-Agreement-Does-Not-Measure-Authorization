import argparse
from pathlib import Path

from evidence_harmonization.synthetic import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic exact-oracle panel.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_panel"))
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    build(args.output, args.pairs, args.seed)


if __name__ == "__main__":
    main()
