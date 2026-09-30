import argparse
from pathlib import Path

from evidence_harmonization.cards import build_cards, write_cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Build privacy-safe Evidence Cards from the source manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/sources.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/cards"))
    args = parser.parse_args()
    write_cards(build_cards(args.manifest), args.output)


if __name__ == "__main__":
    main()
