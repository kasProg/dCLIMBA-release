"""Tests for the ETCCDI-style climate index computations in eval/metrics.py,
against small hand-constructed daily series with known answers.
"""
import numpy as np
import pandas as pd
import pytest

from eval.metrics import (
    compute_cdd,
    compute_cwd,
    compute_rx1day,
    compute_rx5day,
    compute_r10mm,
    compute_r20mm,
    compute_mean_bias,
    compute_mean_bias_percentage,
    get_season,
)


def _daily_index(n_days, start="2000-01-01"):
    return pd.date_range(start=start, periods=n_days, freq="D")


def test_compute_cdd_counts_longest_dry_streak():
    # 10 days: wet, then 5 dry days (<1mm), then wet, then 3 dry days
    time = _daily_index(10)
    y = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    result = compute_cdd(time, y)
    assert result[0] == 5  # longest run of consecutive dry days within the year


def test_compute_cwd_counts_longest_wet_streak():
    time = _daily_index(10)
    y = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 5.0, 5.0, 0.0, 5.0, 5.0])
    result = compute_cwd(time, y)
    assert result[0] == 3  # longest run of consecutive wet days (>=1mm)


def test_compute_rx1day_is_monthly_max():
    time = _daily_index(31)  # January 2000
    y = np.zeros(31)
    y[14] = 42.0  # single spike mid-month
    result = compute_rx1day(time, y)
    assert result[0, 0] == pytest.approx(42.0)


def test_compute_rx5day_captures_best_5day_window():
    time = _daily_index(10)
    y = np.array([0, 0, 5, 5, 5, 5, 5, 0, 0, 0], dtype=float)
    result = compute_rx5day(time, y)
    # best 5-consecutive-day window sums to 25
    assert result[0, 0] == pytest.approx(25.0)


def test_compute_r10mm_counts_days_above_threshold():
    time = _daily_index(5)
    y = np.array([15.0, 5.0, 10.0, 9.99, 20.0])
    result = compute_r10mm(time, y)
    assert result[0, 0] == 3  # 15, 10, 20 are >= 10


def test_compute_r20mm_counts_days_above_threshold():
    time = _daily_index(5)
    y = np.array([15.0, 25.0, 10.0, 20.0, 30.0])
    result = compute_r20mm(time, y)
    assert result[0, 0] == 3  # 25, 20, 30 are >= 20


def test_compute_mean_bias_no_mask():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[1.0, 1.0], [1.0, 1.0]])
    xt = np.array([[1.0, 1.0], [1.0, 1.0]])
    bias_x, bias_xt = compute_mean_bias(None, x, y, xt)
    assert np.allclose(bias_x, [1.0, 2.0])  # mean(x - y) per column
    assert np.allclose(bias_xt, [0.0, 0.0])  # xt matches y exactly


def test_compute_mean_bias_percentage_matches_manual():
    x = np.array([[110.0], [90.0]])
    y = np.array([[100.0], [100.0]])
    xt = np.array([[100.0], [100.0]])
    pct_x, pct_xt = compute_mean_bias_percentage(None, x, y, xt, threshold=1.0)
    assert pct_x[0] == pytest.approx(0.0)  # mean of (+10%, -10%) is 0%
    assert pct_xt[0] == pytest.approx(0.0)


@pytest.mark.parametrize("month,expected", [
    (1, "Winter"), (2, "Winter"), (12, "Winter"),
    (3, "Spring"), (4, "Spring"), (5, "Spring"),
    (6, "Summer"), (7, "Summer"), (8, "Summer"),
    (9, "Autumn"), (10, "Autumn"), (11, "Autumn"),
])
def test_get_season_mapping(month, expected):
    # Note: eval.metrics.get_season returns "Spring"/"Summer"/"Autumn"/"Winter",
    # distinct from data.helper.get_season which returns "DJF"/"MAM"/"JJA"/"SON"
    # (see test_helper.py) -- two independent, differently-labeled helpers.
    assert get_season(month) == expected
