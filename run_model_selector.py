# auto_select.py
import json, math, re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import os

############################
# 1) Configurable settings #
############################
# Your metrics keys (all are "lower is better" AFTER transform in _compute_J)
METRIC_WEIGHTS = {
    "tail_q95": 0.10,
    "tail_q99": 0.15,
    "tail_rx1": 0.15,
    "tail_rx5": 0.15,
    "wet_sdii": 0.15,
    "wet_cdd": 0.10,
    "wet_cwd": 0.05,
    "wet_r10": 0.05,
    "wet_r20": 0.10,
}

# Optionally weight across temporal aggregation scales
# SCALE_WEIGHTS = {"daily": 0.5, "monthly": 0.3, "seasonal": 0.2}

EMA_ALPHA = 0.3            # smoothing for J across epochs
MIN_EPOCHS = 10            # require at least this many epochs to evaluate a run
IMPROVEMENT_FLOOR = 0.02   # require >=2% improvement vs the run's early J, else mark as "stagnant"
NONINCR_TOL = 0.005        # allow tiny non-monotonicity (1%) in the tail of the curve
DEGRADE_TOL    = 0.05          # max allowed end-vs-best degradation (5%)
ALLOWED_SPIKES = 10           # max allowed upward jumps after the best
TAIL_FRAC      = 0.20          # last 20% used only for a soft slope diagnostic

# Filenames expected inside each trial directory
VAL_LOG = "val_metrics.jsonl"    # one json per line: {"epoch": int, "loss": float, "metrics": {...}}
BASELINE_FILE = "baseline.jsonl"  # raw-vs-obs metrics dict used for normalization (same keys as metrics)

#######################
# 2) Helper utilities #
#######################
def _ema(series: List[float], alpha: float) -> List[float]:
    out = []
    s = None
    for x in series:
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return out

def _safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return a / (b + eps)

def _normalize_metric(value: float, baseline: float, lower_better=True) -> float:
    """
    Map metric to [0,1] where 0 ~ perfect, 1 ~ as bad as raw baseline.
    If lower_better is False (e.g., PDF overlap), we invert appropriately.
    """
    if not lower_better:
        # convert to a "gap" first: 1 - overlap
        value = 1.0 - value
        baseline = 1.0 - baseline
    # Normalize relative to baseline (clip to [0,1.5] but we'll clamp later)
    norm = _safe_div(abs(value), abs(baseline))
    return max(0.0, min(1.0, norm))

def _compute_J(metrics: Dict[str, float]) -> float:
# def _compute_J(metrics: Dict[str, float], baseline: Dict[str, float]) -> float:

    """
    Compute single scalar J from (possibly multi-scale) metrics.
    Expect keys like "daily/pdf_overlap", "monthly/rx1_err", etc.
    """
    accum = 0.0
    wsum = 0.0
    # for scale, sw in SCALE_WEIGHTS.items():
    scale_contrib = 0.0
    scale_wsum = 0.0

    # materialize a per-scale dict if present
    def get(key_suffix: str) -> Optional[float]:
        # prefer "<scale>/<key>", else plain "<key>"
        if f"{key_suffix}" in metrics:
            return metrics[f"{key_suffix}"]
        return metrics.get(key_suffix, None)



    # # build normalized components
    # parts = {
    #     "tail_rx1": _normalize_metric(get("Rx1day") or 0.0, baseline.get(f"Rx1day", 1.0)),
    #     "tail_rx5": _normalize_metric(get("Rx5day") or 0.0, baseline.get(f"Rx5day", 1.0)),
    #     "wet_sdii": _normalize_metric(get("SDII (Monthly)") or 0.0, baseline.get(f"SDII (Monthly)", 1.0)),
    #     "wet_cdd":  _normalize_metric(get("CDD (Yearly)") or 0.0,  baseline.get(f"CDD (Yearly)", 1.0)),
    #     "wet_cwd":  _normalize_metric(get("CWD (Yearly)") or 0.0,  baseline.get(f"CWD (Yearly)", 1.0)),
    #     "wet_r10":  _normalize_metric(get("R10mm") or 0.0,  baseline.get(f"R10mm", 1.0)),
    #     "wet_r20":  _normalize_metric(get("R20mm") or 0.0,  baseline.get(f"R20mm", 1.0)),
    #     "tail_q95": _normalize_metric(get("R95pTOT") or 0.0, baseline.get(f"R95pTOT", 1.0)),
    #     "tail_q99": _normalize_metric(get("R99pTOT") or 0.0, baseline.get(f"R99pTOT", 1.0)),
    # }

    # build normalized components
    parts = {
        "tail_rx1": get("Rx1day") or 0.0,
        "tail_rx5": get("Rx5day") or 0.0,
        "wet_sdii": get("SDII (Monthly)") or 0.0,
        "wet_cdd":  get("CDD (Yearly)") or 0.0,
        "wet_cwd":  get("CWD (Yearly)") or 0.0,
        "wet_r10":  get("R10mm") or 0.0,
        "wet_r20":  get("R20mm") or 0.0,
        "tail_q95": get("R95pTOT") or 0.0,
        "tail_q99": get("R99pTOT") or 0.0,
    }

    for k, v in parts.items():
        w = METRIC_WEIGHTS.get(k, 0.0)
        accum += w * abs(v)
        wsum += w

    # if scale_wsum > 0:
    #     accum += sw * (scale_contrib / scale_wsum)
    #     wsum += sw

    return accum / wsum if wsum > 0 else 1.0

def _good_tail_toward_zero(curve, nonincr_tol=NONINCR_TOL,
                           degrade_tol=DEGRADE_TOL,
                           allowed_spikes=ALLOWED_SPIKES,
                           tail_frac=TAIL_FRAC):
    """
    Stability anchored at the global best: ensure |curve| doesn't degrade much after the best,
    and doesn't have too many upward spikes. Returns (good, diagnostics).
    """
    c = np.abs(np.asarray(curve, dtype=float))
    n = len(c)
    if n < 3:
        return True, {"reason": "short"}
    
    if np.isnan(c).any():
        return False, {"reason": "nan"}

    i_best = int(np.argmin(c))
    post = c[i_best:]  # from best to end

    # 1) Degradation from best to end (relative to best magnitude)
    degrade = (post[-1] - post[0]) / max(post[0], 1e-6)
    good_degrade = degrade <= degrade_tol

    # 2) Count upward jumps after best (relative, with floor)
    jumps = int(np.sum(np.diff(post) > nonincr_tol * np.maximum(post[:-1], 1e-6)))
    good_spikes = jumps <= allowed_spikes

    # 3) Optional: last-20% slope diagnostic (soft; not gating)
    tlen = max(1, int(tail_frac * n))
    tail = c[-tlen:]
    x = np.arange(tlen)
    slope = np.polyfit(x, tail, 1)[0] if tlen >= 2 else 0.0
    scale = max(np.median(tail), 1e-6)
    last20_slope_ok = slope <= nonincr_tol * scale / max(tlen, 1)

    good = bool(good_degrade and good_spikes)
    diag = {
        "i_best": i_best,
        "best_abs": float(c[i_best]),
        "end_abs": float(c[-1]),
        "degrade": float(max(0.0, degrade)),
        "jumps_after_best": jumps,
        "last20_slope_ok": bool(last20_slope_ok),
    }
    return good, diag


@dataclass
class EpochEval:
    epoch: int
    loss: float
    J: float
    J_ema: float

@dataclass
class TrialResult:
    trial_dir: str
    run_id: str
    best_epoch: int
    best_J: float
    best_J_ema: float
    final_loss: float
    improved: bool
    good_tail: bool
    good_Ltail: bool
    config_summary: Dict[str, Any]
    r10: float
    r20: float
    rx1: float
    rx5: float

########################
# 3) Core evaluation   #
########################
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def read_config_summary(trial_dir: Path) -> Dict[str, Any]:
    # Try a few common config filenames
    out = {"model_type": None, "layers": None, "hidden": None,
           "parameter_scale": None, "epochs": None, "trial": trial_dir.name}
    # Lightweight parse from path name (fallback)
    m = re.search(r"(MLP|CNN|LSTM)", trial_dir.name, re.I)
    if m: out["model_type"] = m.group(1).upper()
    for fname in ["config.json", "config.yaml", "train_config.yaml"]:
        p = trial_dir / fname
        if p.exists():
            try:
                if p.suffix == ".json":
                    cfg = json.loads(p.read_text())
                else:
                    # naive YAML loader to avoid pyyaml dependency
                    import yaml  # comment out if you truly can't have it
                    cfg = yaml.safe_load(p.read_text())
                out.update({k: cfg.get(k) for k in ["model_type","layers","hidden","parameter_scale","epochs"] if k in cfg})
            except Exception:
                pass
    return out

def evaluate_trial(trial_dir: Path, val_folder: Optional[str] = None) -> Optional[TrialResult]:
    # Try to find val_metrics.jsonl - first in year-range folder if val_period provided, then direct
  
    val_path = trial_dir / Path(val_folder) / VAL_LOG
    base_path = trial_dir /  BASELINE_FILE

    run_id = os.path.basename(trial_dir).split('_')[0]
    if not val_path.exists():
        return None

    logs = load_jsonl(val_path)
    if len(logs) < MIN_EPOCHS:
        return None

    # baseline = json.loads(base_path.read_text())
    epochs, losses, Js, r10, r20, rx1, rx5 = [], [], [], [], [], [], []

    for row in logs:
        ep = int(row.get("epoch", len(epochs)))
        met = row.get("metrics", {})
        loss = float(row.get("loss", math.nan))
        J = _compute_J(met)
        r10.append(met.get("R10mm", math.nan))
        r20.append(met.get("R20mm", math.nan))
        rx1.append(met.get("Rx1day", math.nan))
        rx5.append(met.get("Rx5day", math.nan))
        epochs.append(ep); losses.append(loss); Js.append(J)

    J_ema = _ema(Js, EMA_ALPHA)
    L_ema = _ema(losses, EMA_ALPHA)
    # choose best by raw J (not EMA), but keep EMA for stability diagnostics
    best_idx = min(range(len(Js)), key=lambda i: (abs(Js[i]), i))
    best_epoch = epochs[best_idx]
    best_J, best_J_ema = Js[best_idx], J_ema[best_idx]

    # diagnostics: did J meaningfully improve vs early training?
    early = Js[min(5, len(Js)-1)]
    improved = (early - best_J) / max(early, 1e-6) >= IMPROVEMENT_FLOOR

    best_r10 = r10[best_idx]
    best_r20 = r20[best_idx]
    best_rx1 = rx1[best_idx]
    best_rx5 = rx5[best_idx]    

    # # diagnostics: tail monotonic-ish check over last 20% epochs (EMA)
    # tail_start = int(0.8 * len(J_ema))
    # tail = J_ema[tail_start:]

    # L_tail_start = int(0.8 * len(L_ema))
    # L_tail = L_ema[L_tail_start:]

    # good_tail = True
    # for i in range(1, len(tail)):
    #     if tail[i] - tail[i-1] > NONINCR_TOL * max(tail[i-1], 1e-6):
    #         good_tail = False; break
    
    # good_Ltail = True
    # for i in range(1, len(L_tail)):
    #     if L_tail[i] - L_tail[i-1] > NONINCR_TOL * max(L_tail[i-1], 1e-6):
    #         good_Ltail = False; break
        
    # ---- Use it for both J_ema and L_ema ----
    good_tail, tail_diag   = _good_tail_toward_zero(J_ema)
    good_Ltail, Ltail_diag = _good_tail_toward_zero(L_ema)

    cfg = read_config_summary(trial_dir)
    return TrialResult(
        trial_dir=str(trial_dir),
        run_id=run_id,
        best_epoch=best_epoch,
        best_J=best_J,
        best_J_ema=best_J_ema,
        final_loss=float(losses[-1]),
        improved=improved,
        good_tail=True,
        good_Ltail=good_Ltail,
        config_summary=cfg,
        r10=best_r10,
        r20=best_r20,
        rx1=best_rx1,
        rx5=best_rx5,
    )

########################
# 4) Batch orchestration
########################
def scan_and_rank(root: str, val_period: Optional[str] = None, spatial_extent: Optional[str] = None) -> Dict[str, Any]: 
    rootp = Path(root)
    results: List[TrialResult] = []

    # Find trial directories - look for either direct val_metrics.jsonl or year-range folders
    trial_dirs = []
    for trial in sorted([p for p in rootp.glob("**/") if p.is_dir()]):
        # Check if val_metrics.jsonl exists directly or in val_period folder
        has_val_log = False
        if val_period:
            start_year, end_year = val_period.split(',')
            val_folder = f"{start_year.strip()}_{end_year.strip()}"
        else:
            val_folder = f"{spatial_extent}"
            
        if (trial / val_folder / VAL_LOG).exists():
            has_val_log = True

        if not has_val_log and (trial / VAL_LOG).exists():
            has_val_log = True
        
        if has_val_log:
            trial_dirs.append(trial)

    # print(trial_dirs)
    
    for trial in trial_dirs:
        r = evaluate_trial(trial, val_folder)
        if r is not None:
            results.append(r)
     

    # Filter out unstable runs first
    stable = [r for r in results if r.improved and r.good_tail and r.good_Ltail]
    finalists = stable if stable else results  # if nothing stable, fall back

    # Rank by best_J, then tail-extremes proxy if you log it (we already folded into J)
    finalists.sort(key=lambda r: (r.best_J, r.best_J_ema, r.final_loss))

    # Summaries
    table = []
    for r in finalists:
        row = {
            "trial": Path(r.trial_dir).name,
            "run_id":Path(r.trial_dir).name.split('_')[0],
            "best_epoch": r.best_epoch,
            "best_J": round(r.best_J, 6),
            "best_J_ema": round(r.best_J_ema, 6),
            "final_loss": round(r.final_loss, 6),
            "improved": r.improved,
            "good_tail": r.good_tail,
            "good_Ltail": r.good_Ltail,
            **{f"cfg_{k}": v for k, v in r.config_summary.items()},
            "r10": round(r.r10, 4),
            "r20": round(r.r20, 4),
            "rx1": round(r.rx1, 4),
            "rx5": round(r.rx5, 4),
        }
        table.append(row)

    best = finalists[0] if finalists else None
    return {
        "best": asdict(best) if best else None,
        "ranked": table,
        "n_total": len(results),
        "n_stable": len(stable),
    }

if __name__ == "__main__":
    import argparse, csv, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True, help="Directory containing many trial subfolders")
    ap.add_argument("--out_csv", default="auto_select_results.csv")
    ap.add_argument("--out_json", default="auto_select_best.json")
    ap.add_argument("--val_period", required=False, type=str, help="Validation period, format: start_year,end_year (e.g., 1965,1978)")
    ap.add_argument('--spatial_extent', type=str, required=False, help='huc ids in the form [1,2]')

    args = ap.parse_args()
    summary = scan_and_rank(args.exp_root, args.val_period, args.spatial_extent)



    # CSV table for quick viewing
    rows = summary["ranked"]
    if rows:
        keys = list(rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    # JSON with winner & counts
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[auto_select] scanned={summary['n_total']} stable={summary['n_stable']}")
    if summary["best"]:
        print(f"[auto_select] BEST trial={Path(summary['best']['trial_dir']).name} epoch={summary['best']['best_epoch']} J={summary['best']['best_J']:.4f}")
    else:
        print("[auto_select] No valid trials found.")
