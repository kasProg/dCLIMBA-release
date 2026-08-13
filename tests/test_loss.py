"""Unit tests for the pure loss functions in model/loss.py.

These run on CPU with small synthetic tensors, no NetCDF data or GPU needed.
"""
import torch
import pytest

from model.loss import (
    distributional_loss_interpolated,
    rainy_day_loss,
    trend_loss,
    autocorrelation_loss,
    fourier_spectrum_loss,
    CorrelationLoss,
    totalPrecipLoss,
    spatial_correlation_loss,
    resample_time_nearest,
    kl_divergence_loss,
    wasserstein_distance_loss,
    rmse,
    compute_composite_loss,
)


def test_distributional_loss_zero_for_identical_distributions():
    torch.manual_seed(0)
    y = torch.rand(200, 4)
    loss = distributional_loss_interpolated(y, y, device="cpu", num_quantiles=50, emph_quantile=0.5)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_distributional_loss_positive_for_different_distributions():
    torch.manual_seed(0)
    x = torch.rand(200, 4)
    y = torch.rand(200, 4) + 5.0  # clearly shifted distribution
    loss = distributional_loss_interpolated(x, y, device="cpu", num_quantiles=50, emph_quantile=None)
    assert loss.item() > 0.0


def test_rainy_day_loss_zero_when_identical():
    x = torch.rand(50, 3) * 5
    loss = rainy_day_loss(x, x, threshold=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_correlation_loss_zero_for_perfectly_correlated():
    torch.manual_seed(1)
    base = torch.rand(4, 20)
    pred = base * 2.0 + 1.0  # affine transform preserves Pearson correlation
    loss = CorrelationLoss(pred, base)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_total_precip_loss_zero_when_sums_match():
    x = torch.rand(30, 5)
    loss = totalPrecipLoss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_spatial_correlation_loss_zero_for_identical_input():
    torch.manual_seed(2)
    y = torch.rand(2, 6, 10)  # (B, P, T)
    loss = spatial_correlation_loss(y, y)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_spatial_correlation_loss_resamples_mismatched_time_dim():
    torch.manual_seed(3)
    yhat = torch.rand(2, 4, 12)
    ytrue = torch.rand(2, 4, 8)
    loss = spatial_correlation_loss(yhat, ytrue)
    assert torch.isfinite(loss)


def test_resample_time_nearest_shape():
    x = torch.rand(2, 5, 20)  # (B, P, T_in)
    out = resample_time_nearest(x, T_out=10)
    assert out.shape == (2, 5, 10)


def test_autocorrelation_loss_zero_for_identical_series():
    torch.manual_seed(4)
    x = torch.rand(3, 40)
    loss = autocorrelation_loss(x, x, lags=[1, 2, 3])
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_fourier_spectrum_loss_zero_for_identical_series():
    torch.manual_seed(5)
    x = torch.rand(3, 32)
    loss = fourier_spectrum_loss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_trend_loss_zero_for_identical_linear_trend():
    t = torch.arange(20, dtype=torch.float32).unsqueeze(1)
    x = 2.0 * t + 1.0
    loss = trend_loss(x, x, device="cpu")
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_rmse_matches_manual_computation():
    import numpy as np
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    expected = np.sqrt(np.mean((a - b) ** 2, axis=0))
    assert rmse(a, b) == pytest.approx(expected)


def test_kl_divergence_loss_zero_for_identical_logits():
    x = torch.rand(4, 10)
    loss = kl_divergence_loss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_wasserstein_distance_loss_zero_for_identical_input():
    x = torch.rand(4, 10)
    loss = wasserstein_distance_loss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_compute_composite_loss_sums_requested_terms():
    torch.manual_seed(0)
    transformed_x = torch.rand(2, 5, 20)  # (B, P, T)
    batch_y = torch.rand(2, 5, 20)
    loss_func = ['quantile', 'rainy_day', 'spatial_correlation']

    total, components = compute_composite_loss(
        transformed_x, batch_y, loss_func, device='cpu', emph_quantile=0.5)

    assert set(components.keys()) == set(loss_func)
    assert total.item() == pytest.approx(sum(v.item() for v in components.values()), rel=1e-5)


def test_compute_composite_loss_zero_for_identical_series():
    torch.manual_seed(1)
    x = torch.rand(2, 5, 20)
    loss_func = ['quantile', 'rainy_day', 'correlation', 'totalP', 'spatial_correlation']

    total, components = compute_composite_loss(x, x, loss_func, device='cpu', emph_quantile=0.5)

    assert total.item() == pytest.approx(0.0, abs=1e-3)


def test_compute_composite_loss_rejects_unknown_loss_func():
    x = torch.rand(2, 3, 10)
    with pytest.raises(ValueError):
        compute_composite_loss(x, x, ['not_a_real_term'], device='cpu', emph_quantile=0.5)
