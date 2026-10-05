#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # code root directory
# Temporal test (default): select on val period 1965-1978, test on 2001-2014.
# Spatial test (SPATIAL=true): select on the validation region (spatial_extent_val) and
# test on the held-out region (spatial_extent_test) over 1990-2014.
SPATIAL="${SPATIAL:-false}"
if [[ "$SPATIAL" == "true" ]]; then
  BASE_DIR="${BASE_DIR:-$ROOT/outputs/spatial_Adam_harmonic0/jobs_LOCAspatioTempConv1d}"
  TEST_PERIOD="${TEST_PERIOD:-1990,2014}"
  export SPATIAL_VAL="['07']"          # Upper Mississippi (validation)
  test_args=(--spatial_extent "05")    # Ohio (test)
else
  BASE_DIR="${BASE_DIR:-$ROOT/outputs/Final_repeat_nowghtdecay_5b/jobs_LOCAspatioTempConv1d}"
  TEST_PERIOD="${TEST_PERIOD:-2001,2014}"
  unset SPATIAL_VAL
  test_args=()
fi

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs."

# 1. Run model selector (pass our BASE_DIR through so it scores the same experiment)
BASE_DIR="$BASE_DIR" bash "$ROOT/run_model_selector.sh"

# 2. For each model, extract best trial info and run eval_exp.py on a different GPU
gpu=0
pids=()
for model in "$BASE_DIR"/*-livneh; do
  out_json="$model/demo_select_livneh.json"
  if [[ -f "$out_json" ]]; then
    run_id=$(jq -r '.best.run_id' "$out_json")
    best_epoch=$(jq -r '.best.best_epoch' "$out_json")
    echo "[eval] $model: run_id=$run_id, epoch=$best_epoch on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/eval_exp.py" \
      --run_id "$run_id" \
      --testepoch "$best_epoch" \
      --base_dir "$BASE_DIR" \
      --test_period "$TEST_PERIOD" \
      "${test_args[@]}" &
    pids+=($!)
    gpu=$(( (gpu + 1) % NUM_GPUS ))
  else
    echo "No best trial found for $model"
  fi
done

# Wait for all jobs to finish
for pid in "${pids[@]}"; do
  wait $pid
done
