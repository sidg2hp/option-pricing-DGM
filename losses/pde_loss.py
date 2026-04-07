"""PDE residual loss: mean squared Black-Scholes operator residual.

Computes :math:`\\mathbb{E}[|\\mathcal{L}[u_\\theta]|^2]` at interior
collocation points using automatic differentiation through the network.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn


def compute_pde_loss(
    model: nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    operator: Callable,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compute the PDE residual loss.

    .. math::

        \mathcal{J}_{\mathrm{PDE}} = \frac{1}{N}\sum_{k=1}^{N}
            w_k\,|\mathcal{L}[u_\theta](t_k, x_k)|^2

    Parameters
    ----------
    model : nn.Module
        The DGM or MLP network.
    t : torch.Tensor
        Interior time points, shape ``(N, 1)``.
    x : torch.Tensor
        Interior log-price points, shape ``(N, d)``.
    operator : callable
        The PDE operator, signature
        ``(u_fn, t, x, sigma, rho, r) -> residual``.
        **Here** the operator is assumed to be already partially applied
        with market parameters, so the signature is
        ``(t, x) -> residual`` of shape ``(N, 1)``.
    weights : torch.Tensor or None
        Importance-sampling weights, shape ``(N, 1)``.

    Returns
    -------
    torch.Tensor
        Scalar PDE loss.
    """
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)
    residual = operator(t, x)
    sq_residual = residual**2
    if weights is not None:
        sq_residual = sq_residual * weights
    return sq_residual.mean()
