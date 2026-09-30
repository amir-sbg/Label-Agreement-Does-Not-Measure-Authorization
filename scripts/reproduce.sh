#!/bin/bash
# CPU stages of the paper pipeline. Requires NEUROMARK_ROOT, BUNDLE_SALT, and, for the
# closed-loop and paper-number stages, results/interface/{gemma4,qwen3}/{hidden,visible,contract}.
set -euo pipefail
: "${NEUROMARK_ROOT:?set NEUROMARK_ROOT to the COBRE/FBIRN data root}"
STAGE="${1:-all}"

build() {
  python scripts/build_cards.py --manifest configs/sources.yaml --output data/cards
  python scripts/verify_sources.py --cards data/cards/file_cards.jsonl --output results/source_integrity.json
  python scripts/build_benchmark.py --cards data/cards --output data/real_panel
  python scripts/generate_synthetic.py --output data/synthetic_panel
}

controls() {
  python scripts/run_deterministic_reference.py --cases data/real_panel/cases.jsonl \
    --oracle data/real_panel/oracle.jsonl --output results/deterministic_reference
  python scripts/run_cec.py --cases data/real_panel/cases.jsonl \
    --oracle data/real_panel/oracle.jsonl --output results/cec
  python scripts/run_downstream.py --output results/downstream
}

closed_loop() {
  : "${BUNDLE_SALT:?set BUNDLE_SALT to a private pseudonymization salt}"
  python scripts/materialize_bundle.py --output results/bundles/deterministic --salt "$BUNDLE_SALT"
  args=(--bundle results/bundles/deterministic --name deterministic)
  for model in gemma4 qwen3; do
    for mode in ungated gated; do
      python scripts/materialize_agent_bundle.py \
        --predictions "results/interface/$model/visible/predictions.jsonl" \
        --cases data/real_panel/cases.jsonl --output "results/bundles/${model}_$mode" \
        --salt "$BUNDLE_SALT" --mode "$mode" --model "$model" > /dev/null
      args+=(--bundle "results/bundles/${model}_$mode" --name "${model}_$mode")
    done
  done
  python scripts/evaluate_bundle_downstream.py "${args[@]}" --output results/bundle_downstream
}

numbers() {
  python scripts/paper_numbers.py --results results --cases data/real_panel/cases.jsonl \
    --oracle data/real_panel/oracle.jsonl --output results/paper_numbers.json
}

case "$STAGE" in
  build) build ;;
  controls) controls ;;
  closed_loop) closed_loop ;;
  numbers) numbers ;;
  all) build; controls; closed_loop; numbers ;;
  *) echo "usage: $0 [build|controls|closed_loop|numbers|all]" >&2; exit 2 ;;
esac
