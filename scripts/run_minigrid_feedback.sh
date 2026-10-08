#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/run_minigrid_feedback.sh CHECKPOINT ENV OUTPUT_DIR' >&2
  exit 2
fi
checkpoint=$1; env_name=$2; output_dir=$3
for condition in normal zero frozen; do
  python3 eval_cv_minigrid.py --checkpoint "$checkpoint" --eval-env "$env_name" \
    --condition "$condition" --output-json "$output_dir/${condition}.json" \
    --trace-npz "$output_dir/${condition}.npz" --num-envs 20 --num-steps 8000 --seed-start 0 \
    --H 400 --embd 256 --layer 4 --head 4
 done
