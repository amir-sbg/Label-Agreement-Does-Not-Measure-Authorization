import argparse
from pathlib import Path

from evidence_harmonization.agent import INTERFACES, run_cases
from evidence_harmonization.llm import HFTextGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded candidate agent on one shard of a panel.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local snapshot path")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--interface", choices=INTERFACES, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    generator = HFTextGenerator(
        args.model,
        max_new_tokens=args.max_new_tokens,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    run_cases(
        args.cases, args.output, generator, args.interface, args.model,
        shard_id=args.shard_id, num_shards=args.num_shards, budget=args.budget,
        run_seed=args.run_seed, limit=args.limit,
    )


if __name__ == "__main__":
    main()
