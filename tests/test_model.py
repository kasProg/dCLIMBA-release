"""Unit tests for model/model.py: shape contracts and the monotonicity
guarantee the whole method relies on (MonotoneMap1D must be non-decreasing
in x for any parameter values, since alpha/w/s are forced >= 0 via softplus).
"""
import torch
import pytest

from model.model import MonotoneMap1D, SpatioTemporalQM


def test_monotone_map_is_nondecreasing_in_x():
    torch.manual_seed(0)
    K = 4
    P = 2 + 3 * K
    mono = MonotoneMap1D(n_bumps=K)

    params = torch.randn(1, P)
    x = torch.linspace(-5, 5, steps=200).unsqueeze(0)  # (1, 200)
    params_expanded = params.expand(1, 200, P) if False else params  # broadcastable via (...,P)
    # broadcast params over the 200 x-points: shape (1, 200, P)
    params_b = params.unsqueeze(1).expand(1, 200, P)

    y = mono(x, params_b)
    diffs = y[0, 1:] - y[0, :-1]
    assert (diffs >= -1e-5).all(), "MonotoneMap1D output must be non-decreasing in x"


def test_monotone_map_requires_positive_k():
    mono = MonotoneMap1D(n_bumps=0)
    with pytest.raises(AssertionError):
        mono(torch.zeros(1, 1), torch.zeros(1, 1, 2))  # K = (2-2)//3 = 0


@pytest.mark.parametrize("temp_enc", ["Conv1d", "LSTM", "MLP"])
@pytest.mark.parametrize("transform_type", ["monotone", "poly"])
def test_spatiotemporal_qm_forward_shapes(temp_enc, transform_type):
    torch.manual_seed(0)
    B, P, T, F_in, degree, f_model = 2, 3, 6, 5, 4, 8

    model = SpatioTemporalQM(
        f_in=F_in, f_model=f_model, heads=2, t_blocks=1, st_layers=1,
        degree=degree, dropout=0.0, transform_type=transform_type,
        temp_enc=temp_enc, n_harmonics=0,
    )
    inps = torch.randn(B, P, T, F_in)
    patches_latlon = torch.rand(B, P, 2) * 10 + 30  # plausible lat/lon degrees
    x_target = torch.rand(B, P, T) * 5

    yhat, params = model(inps, patches_latlon, x_target)

    assert yhat.shape == (B, P, T)
    assert torch.isfinite(yhat).all()
    assert (yhat >= 0).all(), "enforce_nonneg=True by default should clip negatives"


def test_spatiotemporal_qm_forward_with_harmonics_and_explicit_t_idx():
    torch.manual_seed(1)
    B, P, T, F_in, degree, f_model = 1, 2, 10, 3, 3, 8

    model = SpatioTemporalQM(
        f_in=F_in, f_model=f_model, heads=2, t_blocks=1, st_layers=1,
        degree=degree, transform_type="monotone", temp_enc="Conv1d",
        n_harmonics=2,
    )
    inps = torch.randn(B, P, T, F_in)
    patches_latlon = torch.rand(B, P, 2) * 10 + 30
    x_target = torch.rand(B, P, T) * 5
    # NOTE: despite the docstring saying t_idx is (T,), the collate_fn in
    # data/loader.py always builds it as (B, T) (same time labels broadcast
    # over the batch), and _fourier_basis relies on that by indexing t[0] to
    # get the (T,) vector -- so we match that real calling convention here.
    t_idx = (torch.arange(1, T + 1, dtype=torch.float32) / T).unsqueeze(0).expand(B, T)

    yhat, params = model(inps, patches_latlon, x_target, t_idx=t_idx)

    assert yhat.shape == (B, P, T)
    assert params.shape == (B, P, T, model.ny)
