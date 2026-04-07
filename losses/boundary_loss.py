"""Boundary condition loss: enforces u -> 0 at the lower domain boundary.

For call-type payoffs, as any asset price goes to zero the option
becomes worthless.
"""

import torch
import torch.nn as nn


def compute_boundary_loss(
    model: nn.Module,
    t_boundary: torch.Tensor,
    x_boundary: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the boundary condition loss.

    .. math::

        \mathcal{J}_{\mathcal{B}} = \frac{1}{N}\sum_{k=1}^{N}
            |u_\theta(t_k, x_k)|^2

    at boundary points where at least one asset coordinate is at its
    lower truncation limit.

    Parameters
    ----------
    model : nn.Module
        The neural network.
    t_boundary : torch.Tensor
        Time points, shape ``(N, 1)``.
    x_boundary : torch.Tensor
        Log-price points at boundary, shape ``(N, d)``.

    Returns
    -------
    torch.Tensor
        Scalar boundary loss.
    """
    u_pred = model(t_boundary, x_boundary)
    return (u_pred**2).mean()
