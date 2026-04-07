"""Diagnostic utilities: PDE residual maps and sanity checks."""

from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn


def compute_residual_grid(
    model: nn.Module,
    operator: Callable,
    x1_range: np.ndarray,
    x2_range: np.ndarray,
    t_val: float,
    device: torch.device,
) -> np.ndarray:
    """Compute PDE residual on a 2-D grid for visualisation.

    Parameters
    ----------
    model : nn.Module
    operator : callable
        Partially-applied PDE operator ``(t, x) -> residual``.
    x1_range, x2_range : np.ndarray
        1-D arrays for the grid axes.
    t_val : float
        Fixed time value.
    device : torch.device

    Returns
    -------
    np.ndarray
        Absolute residual values, shape ``(len(x1_range), len(x2_range))``.
    """
    X1, X2 = np.meshgrid(x1_range, x2_range, indexing="ij")
    n = X1.size
    x = torch.tensor(
        np.stack([X1.ravel(), X2.ravel()], axis=-1),
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)
    t = torch.full((n, 1), t_val, dtype=torch.float32, device=device).requires_grad_(True)

    residual = operator(t, x)
    res_np = residual.detach().cpu().numpy().reshape(X1.shape)
    return np.abs(res_np)


def check_nan_outputs(model: nn.Module, d: int, device: torch.device) -> bool:
    """Check that model produces no NaN outputs on random inputs.

    Parameters
    ----------
    model : nn.Module
    d : int
        Number of asset dimensions.
    device : torch.device

    Returns
    -------
    bool
        True if the model outputs are all finite.
    """
    t = torch.rand(100, 1, device=device) * 0.9
    x = torch.randn(100, d, device=device) * 0.3
    with torch.no_grad():
        u = model(t, x)
    return bool(torch.isfinite(u).all())
