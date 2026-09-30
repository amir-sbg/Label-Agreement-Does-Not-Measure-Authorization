import argparse
from pathlib import Path

from evidence_harmonization.benchmark import build_benchmark, write_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 230-case real panel from Evidence Cards.")
    parser.add_argument("--sources", type=Path, default=Path("configs/sources.yaml"))
    parser.add_argument("--concepts", type=Path, default=Path("configs/concept_bases.yaml"))
    parser.add_argument("--cards", type=Path, default=Path("data/cards"))
    parser.add_argument("--output", type=Path, default=Path("data/real_panel"))
    args = parser.parse_args()
    write_benchmark(build_benchmark(args.sources, args.concepts, args.cards), args.output)


if __name__ == "__main__":
    main()
