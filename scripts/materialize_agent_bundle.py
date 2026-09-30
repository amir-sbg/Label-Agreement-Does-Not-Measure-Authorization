import argparse
import json
from pathlib import Path

from evidence_harmonization.agent_bundle import materialize


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a cohort bundle from agent column decisions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--mode", choices=("gated", "ungated"), required=True)
    parser.add_argument("--model", default="unspecified")
    args = parser.parse_args()
    summary = materialize(args.output, args.predictions, args.cases, args.salt, args.mode, args.model)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
