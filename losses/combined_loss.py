"""Combined loss: weighted sum of PDE, terminal, and boundary losses.

Returns a dictionary of individual and total loss components for
separate logging.
"""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from losses.boundary_loss import compute_boundary_loss
from losses.pde_loss import compute_pde_loss
from losses.terminal_loss import compute_terminal_loss


def compute_combined_loss(
    model: nn.Module,
    t_interior: torch.Tensor,
    x_interior: torch.Tensor,
    x_terminal: torch.Tensor,
    t_boundary: torch.Tensor,
    x_boundary: torch.Tensor,
    operator: Callable,
    payoff_fn: Callable,
    T: float,
    lambda_pde: float = 1.0,
    lambda_terminal: float = 10.0,
    lambda_boundary: float = 1.0,
    use_hard_terminal: bool = True,
    weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    r"""Compute the full DGM loss functional.

    .. math::

        \mathcal{J}(\theta) = \lambda_{\mathcal{L}}\,\mathcal{J}_{\text{PDE}}
            + \lambda_{\mathcal{T}}\,\mathcal{J}_{\text{terminal}}
            + \lambda_{\mathcal{B}}\,\mathcal{J}_{\text{boundary}}

    When using the hard terminal constraint, the terminal loss term is
    omitted (:math:`\lambda_{\mathcal{T}} = 0`).

    Parameters
    ----------
    model : nn.Module
    t_interior, x_interior : torch.Tensor
        Interior collocation points.
    x_terminal : torch.Tensor
        Terminal collocation points.
    t_boundary, x_boundary : torch.Tensor
        Boundary collocation points.
    operator : callable
        Partially-applied PDE operator ``(t, x) -> residual``.
    payoff_fn : callable
        Payoff function ``x -> Phi(x)``.
    T : float
        Maturity.
    lambda_pde, lambda_terminal, lambda_boundary : float
        Loss component weights.
    use_hard_terminal : bool
        If True, skip the terminal loss (it is enforced exactly).
    weights : torch.Tensor or None
        Importance weights for the PDE loss.

    Returns
    -------
    dict
        ``{"loss_pde": ..., "loss_terminal": ..., "loss_boundary": ...,
        "loss_total": ...}``
    """
    loss_pde = compute_pde_loss(model, t_interior, x_interior, operator, weights)
    loss_boundary = compute_boundary_loss(model, t_boundary, x_boundary)

    if use_hard_terminal:
        loss_terminal = torch.tensor(0.0, device=loss_pde.device)
        loss_total = lambda_pde * loss_pde + lambda_boundary * loss_boundary
    else:
        loss_terminal = compute_terminal_loss(model, x_terminal, payoff_fn, T)
        loss_total = (
            lambda_pde * loss_pde
            + lambda_terminal * loss_terminal
            + lambda_boundary * loss_boundary
        )

    return {
        "loss_pde": loss_pde,
        "loss_terminal": loss_terminal,
        "loss_boundary": loss_boundary,
        "loss_total": loss_total,
    }
