#!/usr/bin/env bash
set -euo pipefail

ROOT="/pscratch/sd/k/kas7897/dCLIMBA-release" # your code root directory
BASE_DIR="${BASE_DIR:-$ROOT/outputs/Final_repeat_nowghtdecay_5b/jobs_LOCAspatioTempConv1d}"
VAL_PERIOD="${VAL_PERIOD:-1965,1978}"
TEST_PERIOD="${TEST_PERIOD:-2001,2014}"
SPATIAL_EXTENT="${SPATIAL_EXTENT:-}"
VAL_SPATIAL_EXTENT="${VAL_SPATIAL_EXTENT:-}"


# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs."
echo "Using test period: $TEST_PERIOD"
if [[ -n "$SPATIAL_EXTENT" ]]; then
  echo "Using spatial extent: $SPATIAL_EXTENT"
fi

# 1. Run model selector
if [[ -n "$SPATIAL_EXTENT" ]]; then
  BASE_DIR="$BASE_DIR" SPATIAL_EXTENT="${VAL_SPATIAL_EXTENT:-$SPATIAL_EXTENT}" bash "$ROOT/run_model_selector.sh"
else
  BASE_DIR="$BASE_DIR" VAL_PERIOD="$VAL_PERIOD" bash "$ROOT/run_model_selector.sh"
fi

# 2. For each model, extract best trial info and run eval_exp.py on a different GPU
gpu=0
pids=()
for model in "$BASE_DIR"/*-livneh; do
  out_json="$model/demo_select_livneh.json"
  if [[ -f "$out_json" ]]; then
    run_id=$(jq -r '.best.run_id' "$out_json")
    best_epoch=$(jq -r '.best.best_epoch' "$out_json")
    if [[ "$run_id" == "null" || "$best_epoch" == "null" ]]; then
      echo "No valid best trial found for $model"
      continue
    fi
    echo "[eval] $model: run_id=$run_id, epoch=$best_epoch on GPU $gpu"
    eval_args=(
      --run_id "$run_id"
      --testepoch "$best_epoch"
      --base_dir "$BASE_DIR"
      --test_period "$TEST_PERIOD"
    )
    if [[ -n "$SPATIAL_EXTENT" ]]; then
      eval_args+=(--spatial_extent "$SPATIAL_EXTENT")
    fi
    CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/eval_exp.py" "${eval_args[@]}" &
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