#!/usr/bin/env bash
# run_val_batch.sh
# Scans BASE_FOLDER for run_id directories (directories containing train_config.yaml)
# and runs run_val.py for each run_id. Logs stdout/stderr into each run folder.

set -euo pipefail
IFS=$'\n\t'

BASE_FOLDER=${1:-}
PYTHON=${PYTHON:-python}
RUN_SCRIPT="run_val.py"
# Whether to launch jobs in background (&) or run them in foreground.
# Set RUN_IN_BACKGROUND=false to run interactively in the terminal.
RUN_IN_BACKGROUND=${RUN_IN_BACKGROUND:-true}

# Optional: specify val period as YYYY,YYYY
VAL_PERIOD=${2:-}

# Optional: whether to re-run completed runs (default: false)
RERUN_COMPLETED=${RERUN_COMPLETED:-false}

# Find candidate run directories: those that contain train_config.yaml
mapfile -t RUN_DIRS < <(find "$BASE_FOLDER" -maxdepth 4 -type f -name 'train_config.yaml' -printf '%h\n')

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
  echo "No run directories with train_config.yaml found under $BASE_FOLDER"
  exit 0
fi

for run_path in "${RUN_DIRS[@]}"; do
  # basename may include a year-range suffix like runid_1979_2000 -> strip that if present
  run_dir_basename=$(basename "$run_path")
  if [[ $run_dir_basename =~ ^(.+)_([0-9]{4})_([0-9]{4})$ ]]; then
    run_id=${BASH_REMATCH[1]}
  else
    run_id=$run_dir_basename
  fi
  echo "Processing run_id=$run_id at $run_path"

  out_dir="$run_path/val_runs"
  mkdir -p "$out_dir"

  

  cmd=("$PYTHON" "$RUN_SCRIPT" --run_id "$run_id" --base_dir "$BASE_FOLDER")
  if [ -n "$VAL_PERIOD" ]; then
    cmd+=(--val_period "$VAL_PERIOD")
  fi

  log_stdout="$out_dir/run_val_${run_id}.out"
  log_stderr="$out_dir/run_val_${run_id}.err"

  # Try to assign an available GPU (if nvidia-smi exists). We set CUDA_VISIBLE_DEVICES so
  # the script's internal cuda:0 maps to the selected GPU.
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_idx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
      awk 'BEGIN{min=1e12;idx=0} {if($1+0<min){min=$1+0; idx=NR-1}} END{print idx}')
    if [ -n "$gpu_idx" ]; then
      echo "  -> Running on GPU $gpu_idx: ${cmd[*]}"
      # Log GPU assignment to the job's stdout
      echo "=== GPU Assignment: Physical GPU $gpu_idx (CUDA_VISIBLE_DEVICES=$gpu_idx) ===" > "$log_stdout"
      echo "Command: ${cmd[*]}" >> "$log_stdout"
      echo "Started at: $(date)" >> "$log_stdout"
      echo "=======================================" >> "$log_stdout"
      if [ "$RUN_IN_BACKGROUND" = "true" ]; then
        CUDA_VISIBLE_DEVICES="$gpu_idx" "${cmd[@]}" >>"$log_stdout" 2>"$log_stderr" &
      else
        CUDA_VISIBLE_DEVICES="$gpu_idx" "${cmd[@]}" >>"$log_stdout" 2>"$log_stderr"
      fi
    else
      echo "  -> Running (no GPU index found): ${cmd[*]}"
      if [ "$RUN_IN_BACKGROUND" = "true" ]; then
        "${cmd[@]}" >"$log_stdout" 2>"$log_stderr" &
      else
        "${cmd[@]}" >"$log_stdout" 2>"$log_stderr"
      fi
    fi
  else
    echo "  -> Running (nvidia-smi not found): ${cmd[*]}"
    if [ "$RUN_IN_BACKGROUND" = "true" ]; then
      "${cmd[@]}" >"$log_stdout" 2>"$log_stderr" &
    else
      "${cmd[@]}" >"$log_stdout" 2>"$log_stderr"
    fi
  fi
  pid=$!
  echo "  -> PID $pid; stdout: $log_stdout; stderr: $log_stderr"

  # Optional: sleep a bit between launches to avoid overloading scheduler
  sleep 1

done

echo "Submitted ${#RUN_DIRS[@]} jobs (background processes). Check individual run logs under each run's val_runs/ folder."