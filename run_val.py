"""Re-validate every saved checkpoint of a trained run.

Reproduces the validation that run_exp.py performs inline during training (same data,
loss, ETCCDI metrics and val_metrics.jsonl row format), so run_model_selector.py can
consume the result directly. Use it to regenerate val_metrics.jsonl after a metrics
change, or to validate a run on a different period.

The new log is written to a temp file and only swapped in once every checkpoint has been
validated, so an interrupted run never leaves a partial val_metrics.jsonl behind. The
original training-time log is kept once as val_metrics.orig.jsonl.

Examples:
    python run_val.py --run_path outputs/<exp>/jobs_LOCAspatioTempConv1d/<gcm>-livneh/<cfg>/<runid>_1979_2000
    python run_val.py --run_id 5fcf8609 --base_dir outputs/Final_repeat_nowghtdecay_5b
    python run_val.py --run_path <...> --val_period 1965,1978
"""
import argparse
import datetime
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

import data.helper as helper
from data.loader import DataLoaderWrapper
from eval.metrics import ClimateIndices, compute_mean_bias_percentage, get_day_bias_percentages
from model.loss import (CorrelationLoss, autocorrelation_loss, distributional_loss_interpolated,
                        fourier_spectrum_loss, rainy_day_loss, spatial_correlation_loss,
                        totalPrecipLoss)
from model.model import SpatioTemporalQM

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Must stay in sync with run_exp.py's inline validation.
VAL_METRIC_KEYS = ['SDII (Monthly)', 'CDD (Yearly)', 'CWD (Yearly)', 'Rx1day', 'Rx5day',
                   'R10mm', 'R20mm', 'R95pTOT', 'R99pTOT']
W1, W2 = 0.99, 0.01
INPUT_X = {'precipitation': ['pr', 'prec', 'prcp', 'PRCP', 'precipitation']}
DONE_MARKER = '.revalidated.json'


def parse_args():
    p = argparse.ArgumentParser(description='Re-validate all checkpoints of a trained run')
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument('--run_path', type=str, help='Trial directory (contains train_config.yaml)')
    where.add_argument('--run_id', type=str, help='Run ID, resolved under --base_dir')
    p.add_argument('--base_dir', type=str, help='Root searched for --run_id')
    p.add_argument('--val_period', type=str, default=None,
                   help='start_year,end_year (default: val_start,val_end from train_config.yaml)')
    p.add_argument('--no_tensorboard', action='store_true', help='Skip TensorBoard logging')
    p.add_argument('--seed', type=int, default=42, help='Seed for the random patch sampling')
    args = p.parse_args()
    if args.run_id and not args.base_dir:
        p.error('--run_id requires --base_dir')
    return args


def repo_path(path):
    """Configs store some paths relative to the repo root and some absolute."""
    if path and not os.path.isabs(path):
        return os.path.join(REPO_ROOT, path)
    return path


def list_checkpoints(run_path):
    ckpts = []
    for f in glob.glob(os.path.join(run_path, 'model_*.pth')):
        m = re.fullmatch(r'model_(\d+)\.pth', os.path.basename(f))
        if m:
            ckpts.append((int(m.group(1)), f))
    return sorted(ckpts)


def build_model(config, nx, device):
    return SpatioTemporalQM(
        f_in=nx, f_model=config['hidden_size'], heads=2, t_blocks=config['layers'], st_layers=1,
        degree=config['degree'], dropout=0.1,
        transform_type=config.get('transform_type', 'monotone'),
        temp_enc=config.get('temp_enc', 'Conv1d'),
        n_harmonics=config.get('n_harmonics', 0),
        spatial_attn=config.get('spatial_attn', True),
    ).to(device)


def load_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
    if 'to_params.weight' in state and 'to_coeffs.weight' not in state:
        state['to_coeffs.weight'] = state.pop('to_params.weight')
        state['to_coeffs.bias'] = state.pop('to_params.bias')
    # strict: a config/architecture mismatch must fail loudly, not validate a wrong model
    model.load_state_dict(state, strict=True)


def val_loss_fn(transformed_x, batch_y, loss_func, emph_quantile, device):
    """Same terms and weights as run_exp.py's validation loss."""
    loss = torch.zeros((), device=transformed_x.device)
    if 'quantile' in loss_func:
        loss = loss + W1 * distributional_loss_interpolated(
            transformed_x.movedim(-1, 0), batch_y.movedim(-1, 0), device=device,
            num_quantiles=1000, emph_quantile=emph_quantile)
    if 'autocorrelation' in loss_func:
        loss = loss + W2 * autocorrelation_loss(transformed_x, batch_y)
    if 'fourier' in loss_func:
        loss = loss + W2 * fourier_spectrum_loss(transformed_x, batch_y)
    if 'rainy_day' in loss_func:
        loss = loss + W2 * rainy_day_loss(transformed_x.movedim(-1, 0), batch_y.movedim(-1, 0))
    if 'correlation' in loss_func:
        loss = loss + W2 * CorrelationLoss(transformed_x, batch_y)
    if 'totalP' in loss_func:
        loss = loss + 0.0001 * totalPrecipLoss(transformed_x, batch_y)
    if 'spatial_correlation' in loss_func:
        loss = loss + spatial_correlation_loss(transformed_x, batch_y)
    return loss


def atomic_write_lines(path, lines):
    tmp = f'{path}.{os.getpid()}.tmp'  # unique: parallel runs of one GCM share baseline files
    with open(tmp, 'w') as f:
        f.writelines(lines)
    os.replace(tmp, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('WARNING: CUDA not available, validating on CPU (slow).')

    if args.run_path:
        run_path = os.path.abspath(args.run_path.rstrip('/'))
    else:
        run_path = helper.load_run_path(args.run_id, base_dir=args.base_dir)
    with open(os.path.join(run_path, 'train_config.yaml')) as f:
        config = yaml.safe_load(f)
    run_id = os.path.basename(run_path).split('_')[0]

    clim, ref = config['clim'], config['ref']
    val_period = ([int(x) for x in args.val_period.split(',')] if args.val_period
                  else [config['val_start'], config['val_end']])
    input_attrs = [a for a in config['input_attrs'].split(';') if a]
    loss_func = config['loss']
    emph_quantile = config['emph_quantile']
    autoregression, lag = config['autoregression'], config['lag']
    wet_dry_flag = config['wet_dry_flag']
    time_scale = config.get('time_scale', 'daily')

    # Spatial test validates on spatial_extent_val over the val period, exactly like
    # run_exp.py; the directory name is the str() of that list, e.g. "['07']".
    spatial_test = config.get('spatial_test', False)
    spatial_extent_val = config['spatial_extent_val'] if spatial_test else None
    shapefile_filter_path = repo_path(config['shapefile_filter_path']) if spatial_test else None
    val_dir_name = f'{spatial_extent_val}' if spatial_test else f'{val_period[0]}_{val_period[1]}'
    val_save_path = os.path.join(run_path, val_dir_name)
    os.makedirs(val_save_path, exist_ok=True)

    ckpts = list_checkpoints(run_path)
    if not ckpts:
        raise SystemExit(f'No model_*.pth checkpoints in {run_path}')
    print(f'run {run_id} ({clim}-{ref}) | val {val_dir_name} | {len(ckpts)} checkpoints: '
          f'{[e for e, _ in ckpts]}')

    data_loader_val = DataLoaderWrapper(
        clim=clim, scenario='historical', ref=ref, period=val_period,
        ref_path=repo_path(config['ref_dir']), cmip6_dir=repo_path(config['cmip_dir']),
        input_x=INPUT_X, input_attrs=input_attrs, ref_var=config['ref_var'],
        save_path=val_save_path, stat_save_path=run_path,
        crd=spatial_extent_val, shapefile_filter_path=shapefile_filter_path,
        batch_size=config['batch_size'], train=False, autoregression=autoregression, lag=lag,
        chunk=False, chunk_size=config['chunk_size'], stride=config['stride'],
        wet_dry_flag=wet_dry_flag, time_scale=time_scale, device=device)
    # Built once and reused for every checkpoint (as during training), seeded so the
    # random patch sampling is reproducible across re-runs.
    dataloader_val = data_loader_val.get_spatial_dataloader(
        K=config.get('neighbors', 16), seed=args.seed)
    valid_coords = data_loader_val.valid_coords

    nx = len(INPUT_X) + len(input_attrs) + (lag if autoregression else 0) + (1 if wet_dry_flag else 0)

    writer = None
    if not args.no_tensorboard:
        train_period = [config['train_start'], config['train_end']]
        exp = (f"{config['logging_path']}/{clim}-{ref}/{config.get('transform_type', 'monotone')}_"
               f"{config['layers']}Layers_{config['degree']}degree_quantile{emph_quantile}_scale{time_scale}/"
               f"{run_id}_{train_period[0]}_{train_period[1]}_{val_period[0]}_{val_period[1]}")
        writer = SummaryWriter(os.path.join(REPO_ROOT, 'runs_revised', exp))

    climate_indices = ClimateIndices()
    index_fns = {k: fn for k, (fn, _) in climate_indices.get_indices().items() if k in VAL_METRIC_KEYS}

    # x (raw GCM) and y (reference) are identical for every checkpoint, so their
    # reconstruction and indices are computed once and reused.
    x_val = y_val = x_time_np = None
    x_idx = y_idx = None
    rows, baseline_row = [], None

    for epoch, ckpt_path in ckpts:
        model = build_model(config, nx, device)
        load_weights(model, ckpt_path, device)
        model.eval()

        val_epoch_loss = 0.0
        patch_val, xt_batches, x_batches, y_batches = [], [], [], []
        with torch.no_grad():
            for patches, batch_input_norm, batch_x, batch_y, time_labels_val in dataloader_val:
                patches_latlon = torch.tensor(valid_coords[patches.cpu().numpy()],
                                              dtype=batch_x.dtype).to(device)
                batch_input_norm = batch_input_norm.to(device)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                time_labels_val = time_labels_val.to(device)

                transformed_x, _ = model(batch_input_norm, patches_latlon, batch_x,
                                         t_idx=time_labels_val)
                val_epoch_loss += val_loss_fn(transformed_x, batch_y, loss_func,
                                              emph_quantile, device).item()

                patch_val.append(patches.cpu())
                xt_batches.append(transformed_x.cpu())
                if x_val is None:
                    x_batches.append(batch_x.cpu())
                    y_batches.append(batch_y.cpu())

        avg_val_loss = val_epoch_loss / len(dataloader_val)
        xt_val = data_loader_val.reconstruct_from_patches(patch_val, xt_batches, mode='mean').numpy().T

        if x_val is None:
            covered = np.unique(np.concatenate([p.numpy().ravel() for p in patch_val]))
            n_loc = valid_coords.shape[0]
            print(f'patch coverage: {covered.size}/{n_loc} locations'
                  + ('' if covered.size == n_loc else ' (uncovered locations reconstruct as 0)'))

            x_val = data_loader_val.reconstruct_from_patches(patch_val, x_batches, mode='mean').numpy().T
            y_val = data_loader_val.reconstruct_from_patches(patch_val, y_batches, mode='mean').numpy().T

            x_val_time = torch.load(os.path.join(val_save_path, 'time.pt'), weights_only=False)
            x_time_np = np.array([pd.Timestamp(str(t)) for t in x_val_time])
            x_time_np = np.array([pd.Timestamp(t).replace(hour=0, minute=0, second=0)
                                  for t in x_time_np], dtype='datetime64[D]')
            y_time_np = pd.date_range(start=f'{val_period[0]}-01-01',
                                      end=f'{val_period[1]}-12-31', freq='D').to_numpy()
            y_val = y_val[np.where(np.isin(y_time_np, x_time_np))[0], :]

            x_idx = {k: fn(x_time_np, x_val) for k, fn in index_fns.items()}
            y_idx = {k: fn(x_time_np, y_val) for k, fn in index_fns.items()}

        mean_bias = {k: compute_mean_bias_percentage(None, x_idx[k], y_idx[k], fn(x_time_np, xt_val))
                     for k, fn in index_fns.items()}
        # key order matches run_exp.py's rows (ClimateIndices order)
        row = {'epoch': int(epoch), 'loss': float(avg_val_loss),
               'metrics': {k: float(np.nanmedian(v[1])) for k, v in mean_bias.items()}}
        rows.append(json.dumps(row) + '\n')
        if baseline_row is None:
            baseline_row = {k: float(np.nanmedian(v[0])) for k, v in mean_bias.items()}

        summary = ', '.join(f'{k}={row["metrics"][k]:.1f}' for k in ('Rx1day', 'R95pTOT', 'R99pTOT'))
        print(f'epoch {epoch:4d} | val loss {avg_val_loss:.4f} | {summary}', flush=True)

        if writer is not None:
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            for name, values in mean_bias.items():
                writer.add_scalar(f'median_adjusted/{name}', float(np.nanmedian(values[1])), epoch)
            for name, values in get_day_bias_percentages(x_val, y_val, xt_val, climate_indices).items():
                writer.add_scalar(f'median_adjusted/{name}', float(np.nanmedian(values[1])), epoch)

    if writer is not None:
        writer.close()

    # All checkpoints validated -> swap the new log in, keeping the training-time original once.
    log_path = os.path.join(val_save_path, 'val_metrics.jsonl')
    orig_path = os.path.join(val_save_path, 'val_metrics.orig.jsonl')
    if os.path.exists(log_path) and not os.path.exists(orig_path):
        os.replace(log_path, orig_path)
    atomic_write_lines(log_path, rows)

    # Raw-vs-reference bias, independent of the model; one per GCM dir and val period.
    gcm_dir = os.path.dirname(os.path.dirname(run_path))
    atomic_write_lines(os.path.join(gcm_dir, f'baseline_{val_period[0]}_{val_period[1]}.jsonl'),
                       [json.dumps(baseline_row) + '\n'])

    with open(os.path.join(val_save_path, DONE_MARKER), 'w') as f:
        json.dump({'finished': datetime.datetime.now().isoformat(timespec='seconds'),
                   'epochs': [e for e, _ in ckpts], 'seed': args.seed}, f)
    print(f'wrote {len(rows)} rows -> {log_path}')


if __name__ == '__main__':
    main()
