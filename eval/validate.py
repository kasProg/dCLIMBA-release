"""Shared validation-epoch logic.

Extracts the block that used to be duplicated between run_exp.py's inline
validation section (run every 10 training epochs) and run_val.py's
standalone per-checkpoint validation loop: forward pass over the validation
dataloader, patch reconstruction, ETCCDI-style climate-index bias
computation against the raw and bias-corrected series, and the associated
val_metrics.jsonl / baseline_<period>.jsonl / TensorBoard side effects.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from model.loss import compute_composite_loss
from eval.metrics import ClimateIndices, get_mean_bias_percentages, get_day_bias_percentages

# Metrics kept in val_metrics.jsonl (mirrors the original inline filter in
# run_exp.py / run_val.py).
_LOGGED_METRIC_KEYS = [
    'SDII (Monthly)', 'CDD (Yearly)', 'CWD (Yearly)', 'Rx1day', 'Rx5day',
    'R10mm', 'R20mm', 'R95pTOT', 'R99pTOT',
]


def run_validation_epoch(model, dataloader_val, data_loader_val, device, val_period,
                          val_save_path, job_path, clim, ref, loss_func, emph_quantile,
                          epoch, writer=None):
    """
    Runs one full validation pass and its associated logging side effects.

    Args:
        model: the SpatioTemporalQM model, expected to already be in eval mode.
        dataloader_val: spatial-patch DataLoader for the validation period.
        data_loader_val: the DataLoaderWrapper the dataloader came from (used
            for reconstruct_from_patches).
        device: torch device.
        val_period: [start_year, end_year].
        val_save_path: directory holding the validation period's time.pt and
            where val_metrics.jsonl is appended.
        job_path: the run's job-family directory (e.g.
            f'{save_path}/jobs_LOCAspatioTemp{temp_enc}'), used as the root
            for the one-time baseline_<period>.jsonl write.
        clim, ref: climate model / reference dataset names, for the baseline
            file path.
        loss_func, emph_quantile: passed through to
            model.loss.compute_composite_loss.
        epoch: current epoch number, for logging.
        writer: optional torch.utils.tensorboard.SummaryWriter; if given,
            scalars are logged the same way run_exp.py/run_val.py did.

    Returns:
        avg_val_loss: float
        mean_bias_percentages: dict[str, (raw_bias, corrected_bias)] as
            returned by eval.metrics.get_mean_bias_percentages, filtered to
            _LOGGED_METRIC_KEYS.
    """
    val_epoch_loss = 0.0
    patch_val, xt_val, x_val, y_val = [], [], [], []

    with torch.no_grad():
        for patches, batch_input_norm, batch_x, batch_y, time_labels_val in dataloader_val:
            patches_latlon = torch.tensor(
                data_loader_val.valid_coords[patches.cpu().numpy()], dtype=batch_x.dtype
            ).to(device)

            batch_input_norm = batch_input_norm.to(device)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            time_labels_val = time_labels_val.to(device)

            transformed_x, _ = model(batch_input_norm, patches_latlon, batch_x, t_idx=time_labels_val)

            val_loss, _ = compute_composite_loss(
                transformed_x, batch_y, loss_func, device=device, emph_quantile=emph_quantile)
            val_epoch_loss += val_loss.item()

            xt_val.append(transformed_x.detach().cpu())
            patch_val.append(patches.detach().cpu())
            y_val.append(batch_y.detach().cpu())
            x_val.append(batch_x.detach().cpu())

    avg_val_loss = val_epoch_loss / len(dataloader_val)

    x_val = data_loader_val.reconstruct_from_patches(patch_val, x_val, mode='mean').numpy().T
    xt_val = data_loader_val.reconstruct_from_patches(patch_val, xt_val, mode='mean').numpy().T
    y_val = data_loader_val.reconstruct_from_patches(patch_val, y_val, mode='mean').numpy().T

    x_val_time = torch.load(f'{val_save_path}/time.pt', weights_only=False)
    x_val_time_np = np.array([pd.Timestamp(str(t)) for t in x_val_time])
    x_val_time_np = np.array(
        [pd.Timestamp(t).replace(hour=0, minute=0, second=0) for t in x_val_time_np],
        dtype='datetime64[D]')
    y_val_time_np = pd.date_range(
        start=f"{val_period[0]}-01-01", end=f"{val_period[1]}-12-31", freq="D").to_numpy()
    matched_indices = np.where(np.isin(y_val_time_np, x_val_time_np))[0]
    y_val = y_val[matched_indices, :]

    climate_indices = ClimateIndices()
    mean_bias_percentages = get_mean_bias_percentages(x_val, y_val, xt_val, x_val_time_np, climate_indices)
    day_bias_percentages = get_day_bias_percentages(x_val, y_val, xt_val, climate_indices)
    mean_bias_percentages = {k: v for k, v in mean_bias_percentages.items() if k in _LOGGED_METRIC_KEYS}

    row = {
        "epoch": int(epoch),
        "loss": float(avg_val_loss),
        "metrics": {k: float(np.nanmedian(v[1])) for k, v in mean_bias_percentages.items()},
    }
    with open(f"{val_save_path}/val_metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

    baseline_dir = f"{job_path}/{clim}-{ref}"
    baseline_path = f"{baseline_dir}/baseline_{val_period[0]}_{val_period[1]}.jsonl"
    if not os.path.exists(baseline_path):
        os.makedirs(baseline_dir, exist_ok=True)
        row_baseline = {k: float(np.nanmedian(v[0])) for k, v in mean_bias_percentages.items()}
        with open(baseline_path, "a") as f:
            f.write(json.dumps(row_baseline) + "\n")

    if writer is not None:
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        print(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}")

        for name, values in mean_bias_percentages.items():
            writer.add_scalar(f'median_adjusted/{name}', float(np.nanmedian(values[1])), epoch)
        for name, values in day_bias_percentages.items():
            writer.add_scalar(f'median_adjusted/{name}', float(np.nanmedian(values[1])), epoch)

    return avg_val_loss, mean_bias_percentages
