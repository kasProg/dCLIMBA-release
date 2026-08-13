"""Tests for the shared model construction / checkpoint-loading helpers in
model/build.py.
"""
import torch
import pytest

from model.build import build_model, load_checkpoint
from model.model import SpatioTemporalQM


def _base_config(**overrides):
    cfg = dict(hidden_size=8, layers=1, degree=3, transform_type='monotone', temp_enc='Conv1d')
    cfg.update(overrides)
    return cfg


def test_build_model_defaults_n_harmonics_to_zero_when_absent():
    # Reproduces the run_val.py bug this helper fixes: a config saved before
    # n_harmonics existed (or that simply omits it) must build a model with
    # n_harmonics=0, matching the current config.yaml default -- not the
    # SpatioTemporalQM constructor's own default of 2 -- so checkpoints
    # trained under the current default still load.
    cfg = _base_config()  # no 'n_harmonics' key
    model = build_model(cfg, nx=5, device='cpu')
    assert model.n_harmonics == 0
    assert isinstance(model, SpatioTemporalQM)


def test_build_model_respects_explicit_n_harmonics():
    cfg = _base_config(n_harmonics=2)
    model = build_model(cfg, nx=5, device='cpu')
    assert model.n_harmonics == 2


def test_load_checkpoint_roundtrip(tmp_path):
    cfg = _base_config()
    model = build_model(cfg, nx=5, device='cpu')
    ckpt_path = tmp_path / "model_10.pth"
    torch.save({"epoch": 10, "model_state": model.state_dict(),
                "optimizer_state": {}}, ckpt_path)

    fresh_model = build_model(cfg, nx=5, device='cpu')
    epoch = load_checkpoint(fresh_model, str(ckpt_path), device='cpu')

    assert epoch == 10
    for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p1, p2)


def test_load_checkpoint_accepts_bare_state_dict(tmp_path):
    cfg = _base_config()
    model = build_model(cfg, nx=5, device='cpu')
    ckpt_path = tmp_path / "bare_state_dict.pth"
    torch.save(model.state_dict(), ckpt_path)

    fresh_model = build_model(cfg, nx=5, device='cpu')
    epoch = load_checkpoint(fresh_model, str(ckpt_path), device='cpu')

    assert epoch is None
    for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p1, p2)


def test_load_checkpoint_remaps_legacy_to_params_key(tmp_path):
    cfg = _base_config()
    model = build_model(cfg, nx=5, device='cpu')
    state_dict = model.state_dict()
    # Simulate a legacy checkpoint saved before the to_params -> to_coeffs rename
    state_dict['to_params.weight'] = state_dict.pop('to_coeffs.weight')
    state_dict['to_params.bias'] = state_dict.pop('to_coeffs.bias')
    ckpt_path = tmp_path / "legacy.pth"
    torch.save({"model_state": state_dict}, ckpt_path)

    fresh_model = build_model(cfg, nx=5, device='cpu')
    load_checkpoint(fresh_model, str(ckpt_path), device='cpu')  # should not raise

    assert torch.equal(fresh_model.to_coeffs.weight, model.to_coeffs.weight)
