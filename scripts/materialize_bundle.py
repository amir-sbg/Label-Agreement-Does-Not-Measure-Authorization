import argparse
from pathlib import Path

from evidence_harmonization.bundle import materialize_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the deterministic reference cohort bundle.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True, help="Secret salt for subject pseudonyms; never written to disk.")
    args = parser.parse_args()
    summary = materialize_bundle(args.output, args.salt)
    print(f"wrote {summary['row_count']} rows and {summary['unresolved_issue_count']} unresolved issues to {args.output}")


if __name__ == "__main__":
    main()
