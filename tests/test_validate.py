"""Integration test for eval/validate.run_validation_epoch: wires a tiny
synthetic model + dataloader through the real reconstruction, climate-index,
and file-logging path (no real climate data / GPU required).
"""
import numpy as np
import pandas as pd
import torch

from model.build import build_model
from data.loader import DataLoaderWrapper
from eval.validate import run_validation_epoch, _LOGGED_METRIC_KEYS


class _FakeValDataLoader:
    """Duck-types just enough of DataLoaderWrapper for run_validation_epoch:
    a `valid_coords` array and the real (self-contained) reconstruct_from_patches.
    """
    def __init__(self, valid_coords):
        self.valid_coords = valid_coords

    reconstruct_from_patches = DataLoaderWrapper.reconstruct_from_patches


def test_run_validation_epoch_end_to_end(tmp_path):
    torch.manual_seed(0)
    N, P, T, F_in = 3, 3, 10, 3  # N valid coords, P=patch size (K+1), T=days, F_in=features

    valid_coords = np.array([[40.0, -80.0], [40.1, -80.1], [40.2, -80.2]])
    data_loader_val = _FakeValDataLoader(valid_coords)

    cfg = dict(hidden_size=8, layers=1, degree=2, transform_type='monotone', temp_enc='Conv1d')
    model = build_model(cfg, nx=F_in, device='cpu').eval()

    patches = torch.tensor([[0, 1, 2]])                          # (B=1, P)
    batch_input_norm = torch.randn(1, P, T, F_in)
    batch_x = torch.rand(1, P, T) * 5
    batch_y = torch.rand(1, P, T) * 5
    time_labels_val = (torch.arange(1, T + 1, dtype=torch.float32) / 365.0).unsqueeze(0)

    dataloader_val = [(patches, batch_input_norm, batch_x, batch_y, time_labels_val)]

    val_save_path = tmp_path / "val"
    val_save_path.mkdir()
    time_pt = pd.date_range("2000-01-01", periods=T, freq="D").to_numpy(dtype='datetime64[D]')
    torch.save(time_pt, val_save_path / "time.pt")

    job_path = tmp_path / "jobs"

    avg_val_loss, mean_bias_percentages = run_validation_epoch(
        model=model,
        dataloader_val=dataloader_val,
        data_loader_val=data_loader_val,
        device='cpu',
        val_period=[2000, 2000],
        val_save_path=str(val_save_path),
        job_path=str(job_path),
        clim='fake_clim',
        ref='fake_ref',
        loss_func=['quantile'],
        emph_quantile=0.5,
        epoch=10,
    )

    assert isinstance(avg_val_loss, float)
    assert np.isfinite(avg_val_loss)
    assert set(mean_bias_percentages.keys()).issubset(set(_LOGGED_METRIC_KEYS))

    val_metrics_path = val_save_path / "val_metrics.jsonl"
    assert val_metrics_path.exists()
    import json
    row = json.loads(val_metrics_path.read_text().strip())
    assert row["epoch"] == 10

    baseline_path = job_path / "fake_clim-fake_ref" / "baseline_2000_2000.jsonl"
    assert baseline_path.exists()


def test_run_validation_epoch_does_not_duplicate_baseline_on_second_call(tmp_path):
    torch.manual_seed(1)
    N, P, T, F_in = 3, 3, 8, 3

    valid_coords = np.array([[40.0, -80.0], [40.1, -80.1], [40.2, -80.2]])
    data_loader_val = _FakeValDataLoader(valid_coords)

    cfg = dict(hidden_size=8, layers=1, degree=2, transform_type='monotone', temp_enc='Conv1d')
    model = build_model(cfg, nx=F_in, device='cpu').eval()

    patches = torch.tensor([[0, 1, 2]])
    batch_input_norm = torch.randn(1, P, T, F_in)
    batch_x = torch.rand(1, P, T) * 5
    batch_y = torch.rand(1, P, T) * 5
    time_labels_val = (torch.arange(1, T + 1, dtype=torch.float32) / 365.0).unsqueeze(0)
    dataloader_val = [(patches, batch_input_norm, batch_x, batch_y, time_labels_val)]

    val_save_path = tmp_path / "val"
    val_save_path.mkdir()
    time_pt = pd.date_range("2000-01-01", periods=T, freq="D").to_numpy(dtype='datetime64[D]')
    torch.save(time_pt, val_save_path / "time.pt")
    job_path = tmp_path / "jobs"

    kwargs = dict(
        model=model, dataloader_val=dataloader_val, data_loader_val=data_loader_val,
        device='cpu', val_period=[2000, 2000], val_save_path=str(val_save_path),
        job_path=str(job_path), clim='fake_clim', ref='fake_ref',
        loss_func=['quantile'], emph_quantile=0.5,
    )
    run_validation_epoch(epoch=10, **kwargs)
    baseline_path = job_path / "fake_clim-fake_ref" / "baseline_2000_2000.jsonl"
    first_contents = baseline_path.read_text()

    run_validation_epoch(epoch=20, **kwargs)
    second_contents = baseline_path.read_text()

    # baseline is a one-time write keyed on the val period, not appended every epoch
    assert first_contents == second_contents
    assert len(second_contents.strip().splitlines()) == 1
