#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # your code root directory
BASE_DIR="${BASE_DIR:-$ROOT/outputs/Final_repeat_nowghtdecay_5b/jobs_LOCAspatioTempConv1d}"
output_dir="$BASE_DIR"
SPATIAL_EXTENT="${SPATIAL_EXTENT:-}"
VAL_SPATIAL_EXTENT="${VAL_SPATIAL_EXTENT:-}"

echo "Using output dir: $output_dir"

if [[ -n "$SPATIAL_EXTENT" ]]; then
  echo "Using spatial extent: $SPATIAL_EXTENT"
elif [[ -n "$VAL_SPATIAL_EXTENT" ]]; then
  SPATIAL_EXTENT="$VAL_SPATIAL_EXTENT"
  echo "Using spatial extent: $SPATIAL_EXTENT"
else
  VAL_PERIOD="${VAL_PERIOD:-1965,1978}"
  echo "Using validation period: $VAL_PERIOD"
fi

for exp_root in "$output_dir"/*-livneh ; do
  [[ -d "$exp_root" ]] || continue
  model="$(basename "$exp_root")"  # e.g., gfdl_esm4-gridmet
  outdir="$output_dir/$model"
  mkdir -p "$outdir"

  echo "[run] $model"
  selector_args=(
    --exp_root "$exp_root"
    --out_csv "$outdir/demo_select_livneh.csv"
    --out_json "$outdir/demo_select_livneh.json"
  )
  if [[ -n "$SPATIAL_EXTENT" ]]; then
    selector_args+=(--spatial_extent "$SPATIAL_EXTENT")
  else
    selector_args+=(--val_period "$VAL_PERIOD")
  fi
  python "$ROOT/run_model_selector.py" "${selector_args[@]}"
done
