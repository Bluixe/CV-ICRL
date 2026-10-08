#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo 'Usage: bash scripts/run_minigrid_baseline.sh reinformer|ic_cql ENV DATA_DIR OUTPUT_DIR' >&2
  exit 2
fi
method=$1; env_name=$2; data_dir=$3; output_dir=$4
case "$method" in
  reinformer) epoch=10; extra=(--action-head-type single_pass_linear_v3 --expectile 0.99 --rtg-gamma 1 --warmup-updates 5000 --amp-dtype bfloat16) ;;
  ic_cql) epoch=5; extra=(--tuple-mode paper --gamma 0.9 --cql-weight 0.01 --cql-label-smoothing 0.3 --target-tau 0.005 --grad-clip 1) ;;
  *) echo 'Method must be reinformer or ic_cql.' >&2; exit 2 ;;
esac
python3 "train_${method}_minigrid.py" \
  --env "$env_name" --train-data "$data_dir/train_traj-more.pkl" --val-data "$data_dir/test_traj.pkl" \
  --train-histories-per-stream 17 --val-histories-per-stream 17 --output-dir "$output_dir" \
  --H 400 --embd 256 --layer 4 --head 4 --dropout 0 --num-epochs 15 \
  --batch-size 8 --grad-accum-steps 4 --seed 0 --wandb-mode disabled "${extra[@]}"
python3 "eval_${method}_minigrid.py" \
  --checkpoint "$output_dir/epoch_${epoch}.pt" --eval-env "$env_name" --output-json "$output_dir/eval.json" \
  --num-envs 20 --num-steps 8000 --seed-start 20000 --torch-seed 0 --wandb-mode disabled
