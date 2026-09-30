import argparse
from pathlib import Path

from evidence_harmonization.source_integrity import verify_source_integrity, write_source_integrity


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that every source artifact still matches its Evidence Card hash.")
    parser.add_argument("--cards", type=Path, default=Path("data/cards/file_cards.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/source_integrity.json"))
    args = parser.parse_args()
    summary = verify_source_integrity(args.cards)
    write_source_integrity(summary, args.output)
    print(f"{summary['status']}: {summary['match_count']}/{summary['artifact_count']} artifacts match")
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
