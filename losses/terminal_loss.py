"""Terminal condition loss: squared error between network and payoff at t=T.

Only used when the hard terminal constraint is disabled.
"""

from typing import Callable

import torch
import torch.nn as nn


def compute_terminal_loss(
    model: nn.Module,
    x_terminal: torch.Tensor,
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    T: float,
) -> torch.Tensor:
    r"""Compute the terminal condition loss.

    .. math::

        \mathcal{J}_{\mathcal{T}} = \frac{1}{N}\sum_{k=1}^{N}
            |u_\theta(T, x_k) - \Phi(x_k)|^2

    Parameters
    ----------
    model : nn.Module
        The neural network.
    x_terminal : torch.Tensor
        Log-price points at maturity, shape ``(N, d)``.
    payoff_fn : callable
        Payoff function in log-price coordinates.
    T : float
        Maturity time.

    Returns
    -------
    torch.Tensor
        Scalar terminal loss.
    """
    N = x_terminal.shape[0]
    t_T = torch.full((N, 1), T, dtype=x_terminal.dtype, device=x_terminal.device)
    u_pred = model(t_T, x_terminal)
    phi = payoff_fn(x_terminal)
    return ((u_pred - phi) ** 2).mean()
