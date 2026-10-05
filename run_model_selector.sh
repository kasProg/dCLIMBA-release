#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # code root directory
output_dir="${BASE_DIR:-$ROOT/outputs/Final_repeat_nowghtdecay_5b/jobs_LOCAspatioTempConv1d}"
# Spatial test: validation logs live in <run>/<str(spatial_extent_val)>/, e.g. "['07']"
# (Upper Mississippi). Set SPATIAL_VAL="['07']" to select on that instead of a val period.
SPATIAL_VAL="${SPATIAL_VAL:-}"
if [[ -n "$SPATIAL_VAL" ]]; then
  val_args=(--spatial_extent "$SPATIAL_VAL")
else
  val_args=(--val_period "${VAL_PERIOD:-1965,1978}")
fi
for exp_root in "$output_dir"/*-livneh ; do
  [[ -d "$exp_root" ]] || continue
  model="$(basename "$exp_root")"  # e.g., gfdl_esm4-gridmet
  outdir="$output_dir/$model"
  mkdir -p "$outdir"

  echo "[run] $model"
  python "$ROOT/run_model_selector.py" \
    --exp_root "$exp_root" \
    --out_csv  "$outdir/demo_select_livneh.csv" \
    --out_json "$outdir/demo_select_livneh.json" \
    "${val_args[@]}"
done
