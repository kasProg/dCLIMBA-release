#!/usr/bin/env bash
set -euo pipefail

ROOT="/dCLIMBA-release" # your code root directory
BASE_DIR="$ROOT/outputs/Final_repeat_nowghtdecay_5b/jobs_LOCAspatioTempConv1d"


# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs."

# 1. Run model selector
bash "$ROOT/run_model_selector.sh"

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
      --test_period "2001,2014" &
      # --spatial_extent "05" & # uncomment if you want to run spatial test on Ohio region
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