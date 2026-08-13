"""Tests for the pure scoring/diagnostic functions in run_model_selector.py."""
import math
import pytest

from run_model_selector import _ema, _compute_J, _good_tail_toward_zero


def test_ema_first_value_is_unsmoothed():
    series = [10.0, 20.0, 30.0]
    out = _ema(series, alpha=0.5)
    assert out[0] == 10.0


def test_ema_converges_toward_series_with_high_alpha():
    series = [10.0, 0.0, 0.0, 0.0]
    out = _ema(series, alpha=1.0)  # alpha=1 -> no smoothing, tracks series exactly
    assert out == series


def test_compute_J_all_zero_metrics_is_zero():
    metrics = {k: 0.0 for k in [
        "Rx1day", "Rx5day", "SDII (Monthly)", "CDD (Yearly)", "CWD (Yearly)",
        "R10mm", "R20mm", "R95pTOT", "R99pTOT",
    ]}
    assert _compute_J(metrics) == pytest.approx(0.0)


def test_compute_J_scales_with_metric_magnitude():
    small = {"Rx1day": 1.0}
    large = {"Rx1day": 10.0}
    assert _compute_J(large) > _compute_J(small)


def test_compute_J_empty_metrics_defaults_to_zero():
    # `parts` is built from METRIC_WEIGHTS keys with `get(...) or 0.0` fallbacks,
    # so an empty metrics dict yields all-zero parts (and wsum > 0 from the
    # weights themselves) -- the `else 1.0` fallback in _compute_J is only hit
    # if wsum is 0, which can't happen while METRIC_WEIGHTS is non-empty.
    assert _compute_J({}) == pytest.approx(0.0)


def test_good_tail_toward_zero_flags_stable_improving_curve():
    # Monotonically shrinking toward zero -> stable, no degradation
    curve = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5]
    good, diag = _good_tail_toward_zero(curve)
    assert good is True
    assert diag["i_best"] == len(curve) - 1


def test_good_tail_toward_zero_flags_degrading_curve():
    # Bottoms out early, then degrades sharply toward the end
    curve = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]
    good, diag = _good_tail_toward_zero(curve)
    assert good is False


def test_good_tail_toward_zero_handles_nan():
    curve = [1.0, float("nan"), 2.0]
    good, diag = _good_tail_toward_zero(curve)
    assert good is False
    assert diag["reason"] == "nan"


def test_good_tail_toward_zero_short_curve_is_trivially_good():
    good, diag = _good_tail_toward_zero([1.0, 2.0])
    assert good is True
    assert diag["reason"] == "short"
