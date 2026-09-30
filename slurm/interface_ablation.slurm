#!/bin/bash
#SBATCH --job-name=interface_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch --partition=<p> --account=<a> slurm/interface_ablation.slurm gemma4|qwen3
# Runs the hidden, visible, and contract arms on the 230-case real panel, 8 shards per arm.

set -euo pipefail
MODEL_TAG="${1:?model tag: gemma4 or qwen3}"
case "$MODEL_TAG" in
  gemma4) MODEL="google/gemma-4-12B-it"; REVISION="5926caa4ec0cac5cbfadaf4077420520de1d5205" ;;
  qwen3) MODEL="Qwen/Qwen3-4B-Instruct-2507"; REVISION="cdbee75f17c01a7cc42f958dc650907174af0554" ;;
  *) echo "unknown model tag: $MODEL_TAG" >&2; exit 2 ;;
esac

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
PANEL="${PANEL:-$ROOT/data/real_panel}"
OUT="$ROOT/results/interface/$MODEL_TAG"
SHARDS="${SHARDS:-8}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT" "$ROOT/logs"

for ARM in hidden visible contract; do
  mkdir -p "$OUT/$ARM/shards"
  pids=()
  for SHARD in $(seq 0 $((SHARDS - 1))); do
    srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=60G \
      python "$ROOT/scripts/run_agent.py" \
        --cases "$PANEL/cases.jsonl" \
        --output "$OUT/$ARM/shards/shard_$(printf '%03d' "$SHARD").jsonl" \
        --model "$MODEL" --revision "$REVISION" --interface "$ARM" \
        --shard-id "$SHARD" --num-shards "$SHARDS" \
        --budget 4 --run-seed 0 --max-new-tokens 256 \
      > "$ROOT/logs/${MODEL_TAG}_${ARM}_shard${SHARD}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  python "$ROOT/scripts/merge_evaluate.py" \
    --shards "$OUT/$ARM/shards" --oracle "$PANEL/oracle.jsonl" \
    --output "$OUT/$ARM" --expected 230
done
