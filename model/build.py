"""Shared model construction/checkpoint-loading helpers.

Centralizes the SpatioTemporalQM(...) constructor call and the checkpoint
state_dict loading logic (including the to_params -> to_coeffs rename
compatibility shim) that used to be duplicated across run_exp.py,
eval_exp.py, and run_val.py.
"""
import torch

from model.model import SpatioTemporalQM


def build_model(config, nx, device, heads=2, st_layers=1, dropout=0.1):
    """
    Constructs a SpatioTemporalQM consistent with how a trial's
    train_config.yaml (or a live Hydra cfg converted to a dict) describes it.

    Args:
        config: dict-like (train_config.yaml contents or OmegaConf-derived
            dict) with at least 'hidden_size', 'layers', 'degree',
            'transform_type', 'temp_enc', and optionally 'n_harmonics'
            (defaults to 0 if absent, matching the current config.yaml
            default and older trials saved before n_harmonics existed).
        nx: number of input features (f_in).
        device: torch device to move the model to.

    Returns:
        SpatioTemporalQM instance on `device`.
    """
    n_harmonics = config.get('n_harmonics', 0) if hasattr(config, 'get') else config['n_harmonics']
    return SpatioTemporalQM(
        f_in=nx,
        f_model=config['hidden_size'],
        heads=heads,
        t_blocks=config['layers'],
        st_layers=st_layers,
        degree=config['degree'],
        dropout=dropout,
        transform_type=config['transform_type'],
        temp_enc=config['temp_enc'],
        n_harmonics=n_harmonics,
    ).to(device)


def load_checkpoint(model, path, device, optimizer=None, strict=False):
    """
    Loads a checkpoint saved by run_exp.py into `model` (and `optimizer`, if
    given). Handles both the current {"epoch", "model_state",
    "optimizer_state"} checkpoint format and a bare state_dict, and remaps
    the legacy 'to_params' parameter name to the current 'to_coeffs' name so
    old checkpoints keep loading.

    Returns:
        epoch: int or None -- the checkpoint's recorded epoch, if present.
    """
    ckpt = torch.load(path, map_location=device)

    try:
        state_dict = ckpt['model_state']
    except (KeyError, TypeError):
        state_dict = ckpt

    if 'to_params.weight' in state_dict and 'to_coeffs.weight' not in state_dict:
        state_dict['to_coeffs.weight'] = state_dict.pop('to_params.weight')
        state_dict['to_coeffs.bias'] = state_dict.pop('to_params.bias')
        print("Remapped 'to_params' -> 'to_coeffs' for compatibility")

    model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and isinstance(ckpt, dict) and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])

    return ckpt.get('epoch') if isinstance(ckpt, dict) else None
