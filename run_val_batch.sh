#!/usr/bin/env bash
# run_val_batch.sh -- re-validate every trained run under one or more experiment folders.
#
# Finds run directories (containing train_config.yaml and model_*.pth) and runs run_val.py
# on each, as a job queue: at most JOBS_PER_GPU jobs per GPU at a time. Each run's logs go
# to <run>/val_runs/run_val.{out,err}. Runs already re-validated (marker file present) are
# skipped unless FORCE=true, so an interrupted batch can simply be restarted.
#
# Usage (from a compute node, e.g. inside `salloc -q interactive ...`):
#   ./run_val_batch.sh outputs/Final_repeat_nowghtdecay_5b outputs/Final_repeat_nowghtdecay_5b_no_spatial
#   ./run_val_batch.sh outputs                      # every experiment under outputs/
#   VAL_PERIOD=1965,1978 ./run_val_batch.sh outputs/<exp>
#   JOBS_PER_GPU=2 FORCE=true ./run_val_batch.sh outputs/<exp>
#   DRY_RUN=true ./run_val_batch.sh outputs         # list what would run
#
# Env vars:
#   VAL_PERIOD     start,end validation years   (default: each run's val_start,val_end)
#   JOBS_PER_GPU   concurrent jobs per GPU      (default: 1)
#   GPUS           comma-separated GPU ids      (default: all visible GPUs)
#   FORCE          re-validate even if done     (default: false)
#   INCLUDE_UNFINISHED  also runs without finished.txt (default: false). Those are either
#                  still training -- re-validating would race run_exp.py's own writes to
#                  val_metrics.jsonl -- or aborted, and need training resumed anyway.
#   DRY_RUN        only list the runs           (default: false)
#   TENSORBOARD    log to runs_revised/         (default: false)
#   THREADS_PER_JOB  CPU threads per job        (default: cores / concurrent jobs)
#   PYTHON         python executable            (default: python)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python}
VAL_PERIOD=${VAL_PERIOD:-}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
FORCE=${FORCE:-false}
INCLUDE_UNFINISHED=${INCLUDE_UNFINISHED:-false}
DRY_RUN=${DRY_RUN:-false}
TENSORBOARD=${TENSORBOARD:-false}

if [[ $# -eq 0 ]]; then
  sed -n '2,27p' "$0"
  exit 1
fi

# Validation is minutes of CPU+GPU work per run; keep it off shared login nodes.
if [[ -z "${SLURM_JOB_ID:-}" && "$DRY_RUN" != "true" ]]; then
  echo "Not inside a Slurm allocation. Run from a compute node, e.g.:" >&2
  echo "  salloc -N 1 -C gpu -q interactive -t 04:00:00 -A <account>" >&2
  exit 1
fi

# ---- discover runs ---------------------------------------------------------
# Layout: <exp>/jobs_*/<gcm>-<ref>/<cfg>/<run>/train_config.yaml. Accept a parent of
# experiments (e.g. outputs/), an experiment dir, or its jobs_* dir. Fixed-depth globs,
# no recursive traversal.
shopt -s nullglob
RUN_DIRS=()
for base in "$@"; do
  base="${base%/}"
  if [[ ! -d "$base" ]]; then
    echo "Not a directory: $base" >&2
    exit 1
  fi
  for cfg in "$base"/*/jobs_*/*/*/*/train_config.yaml \
             "$base"/jobs_*/*/*/*/train_config.yaml \
             "$base"/*/*/*/train_config.yaml; do
    run_dir="$(cd "$(dirname "$cfg")" && pwd)"
    ckpts=("$run_dir"/model_*.pth)
    if [[ ${#ckpts[@]} -gt 0 ]]; then
      RUN_DIRS+=("$run_dir")
    fi
  done
done
shopt -u nullglob
if [[ ${#RUN_DIRS[@]} -gt 0 ]]; then
  mapfile -t RUN_DIRS < <(printf '%s\n' "${RUN_DIRS[@]}" | sort -u)
fi

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No trained runs (train_config.yaml + model_*.pth) found under: $*"
  exit 0
fi

# Directory run_val.py writes into for a run (mirrors run_val.py / run_exp.py naming).
val_dir_of() {
  "$PYTHON" - "$1" "$VAL_PERIOD" <<'EOF'
import sys, yaml
run, override = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(f"{run}/train_config.yaml"))
if c.get("spatial_test"):
    print(f"{run}/{c['spatial_extent_val']}")
else:
    start, end = ([s.strip() for s in override.split(",")] if override
                  else (c["val_start"], c["val_end"]))
    print(f"{run}/{start}_{end}")
EOF
}

TODO=()
UNFINISHED=()
N_DONE=0
for run in "${RUN_DIRS[@]}"; do
  if [[ "$INCLUDE_UNFINISHED" != "true" && ! -f "$run/finished.txt" ]]; then
    UNFINISHED+=("$run")
    continue
  fi
  if [[ "$FORCE" != "true" && -f "$(val_dir_of "$run")/.revalidated.json" ]]; then
    N_DONE=$((N_DONE + 1))
    continue
  fi
  TODO+=("$run")
done

echo "Found ${#RUN_DIRS[@]} trained runs: ${#TODO[@]} to validate, $N_DONE already done, ${#UNFINISHED[@]} unfinished (skipped)."
if [[ ${#UNFINISHED[@]} -gt 0 ]]; then
  echo "Skipped (no finished.txt: still training or aborted; INCLUDE_UNFINISHED=true to include):"
  printf '  %s\n' "${UNFINISHED[@]#"$ROOT"/}"
fi
if [[ ${#TODO[@]} -eq 0 ]]; then
  exit 0
fi
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Would validate:"
  printf '  %s\n' "${TODO[@]#"$ROOT"/}"
  exit 0
fi

# ---- GPU slots -------------------------------------------------------------
if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "$GPUS"
else
  mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
SLOTS=()
for g in "${GPU_LIST[@]}"; do
  for ((k = 0; k < JOBS_PER_GPU; k++)); do SLOTS+=("$g"); done
done
N_SLOTS=${#SLOTS[@]}
# Share the node's cores between concurrent jobs instead of every job grabbing all of them.
THREADS=${THREADS_PER_JOB:-$(( $(nproc) / N_SLOTS ))}; (( THREADS >= 1 )) || THREADS=1
echo "GPUs: ${GPU_LIST[*]} | $N_SLOTS concurrent jobs | $THREADS threads each"

# Some envs set GDAL/PROJ vars that break pyogrio's bundled GDAL (seen with dCLIMAD).
unset GDAL_DATA GDAL_DRIVER_PATH PROJ_DATA || true

declare -a SLOT_PID SLOT_RUN
OK=0
FAILED=()

reap() {  # collect a finished slot's exit status
  local s=$1
  if wait "${SLOT_PID[$s]}"; then
    OK=$((OK + 1))
    echo "[done] ${SLOT_RUN[$s]#"$ROOT"/}"
  else
    FAILED+=("${SLOT_RUN[$s]}")
    echo "[FAIL] ${SLOT_RUN[$s]#"$ROOT"/}  -> see val_runs/run_val.err"
  fi
  SLOT_PID[$s]=""
}

launch() {  # launch run $2 in slot $1
  local s=$1 run=$2
  local log_dir="$run/val_runs"
  mkdir -p "$log_dir"
  local cmd=("$PYTHON" "$ROOT/run_val.py" --run_path "$run")
  if [[ -n "$VAL_PERIOD" ]]; then
    cmd+=(--val_period "$VAL_PERIOD")
  fi
  if [[ "$TENSORBOARD" != "true" ]]; then
    cmd+=(--no_tensorboard)
  fi
  {
    echo "=== GPU ${SLOTS[$s]} | started $(date) ==="
    echo "${cmd[*]}"
  } > "$log_dir/run_val.out"
  # exec: the background PID is python itself, so the INT/TERM trap really stops it
  ( cd "$ROOT" || exit 1
    export CUDA_VISIBLE_DEVICES="${SLOTS[$s]}" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
    exec "${cmd[@]}" >> "$log_dir/run_val.out" 2> "$log_dir/run_val.err" ) &
  SLOT_PID[$s]=$!
  SLOT_RUN[$s]=$run
  echo "[start] GPU ${SLOTS[$s]}: ${run#"$ROOT"/}"
}

on_interrupt() {
  echo
  echo "Interrupted -- stopping running jobs (finished runs keep their results; rerun to resume)."
  for p in "${SLOT_PID[@]}"; do
    if [[ -n "$p" ]]; then kill "$p" 2>/dev/null || true; fi
  done
  exit 130
}
trap on_interrupt INT TERM

for run in "${TODO[@]}"; do
  while :; do
    for ((s = 0; s < N_SLOTS; s++)); do
      if [[ -z "${SLOT_PID[$s]:-}" ]]; then
        launch "$s" "$run"; continue 3
      elif ! kill -0 "${SLOT_PID[$s]}" 2>/dev/null; then
        reap "$s"; launch "$s" "$run"; continue 3
      fi
    done
    sleep 5
  done
done

for ((s = 0; s < N_SLOTS; s++)); do
  if [[ -n "${SLOT_PID[$s]:-}" ]]; then
    reap "$s"
  fi
done

echo
echo "Finished: $OK succeeded, ${#FAILED[@]} failed."
if [[ ${#FAILED[@]} -gt 0 ]]; then
  printf '  %s\n' "${FAILED[@]#"$ROOT"/}"
  exit 1
fi
