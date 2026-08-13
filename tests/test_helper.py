"""Tests for the small utility functions in data/helper.py."""
import pytest

from data.helper import generate_run_id, extract_time_labels, get_season, load_trial_config


def test_generate_run_id_is_deterministic():
    cfg = {"clim": "access_cm2", "degree": 8, "emph_quantile": 0.5}
    assert generate_run_id(cfg) == generate_run_id(cfg)


def test_generate_run_id_is_key_order_independent():
    cfg_a = {"clim": "access_cm2", "degree": 8}
    cfg_b = {"degree": 8, "clim": "access_cm2"}
    assert generate_run_id(cfg_a) == generate_run_id(cfg_b)


def test_generate_run_id_differs_for_different_configs():
    cfg_a = {"clim": "access_cm2", "degree": 8}
    cfg_b = {"clim": "access_cm2", "degree": 10}
    assert generate_run_id(cfg_a) != generate_run_id(cfg_b)


def test_generate_run_id_is_short_hash():
    run_id = generate_run_id({"a": 1})
    assert len(run_id) == 8


def test_get_season_mapping():
    assert get_season(1) == "DJF"
    assert get_season(4) == "MAM"
    assert get_season(7) == "JJA"
    assert get_season(10) == "SON"


def test_extract_time_labels_month():
    import pandas as pd
    times = pd.date_range("2000-01-01", periods=3, freq="MS")  # Jan, Feb, Mar
    labels = extract_time_labels(times, label_type="month")
    assert labels == ["01", "02", "03"]


def test_extract_time_labels_season():
    import pandas as pd
    times = pd.to_datetime(["2000-01-15", "2000-07-15"])
    labels = extract_time_labels(times, label_type="season")
    assert labels == ["DJF", "JJA"]


def test_load_trial_config_finds_and_parses_yaml(tmp_path):
    run_id = "abc12345"
    trial_dir = tmp_path / f"{run_id}_1979_2000"
    trial_dir.mkdir()
    (trial_dir / "train_config.yaml").write_text("clim: access_cm2\ndegree: 8\n")

    run_path, config = load_trial_config(run_id, base_dir=str(tmp_path))

    assert run_path == str(trial_dir)
    assert config == {"clim": "access_cm2", "degree": 8}
