"""Boundary and terminal condition utilities for the truncated PDE domain.

Provides helpers for computing the truncated computational domain and
evaluating boundary conditions in log-price coordinates.
"""

from typing import Tuple

import numpy as np
import torch


def compute_domain_bounds(
    sigma: np.ndarray,
    T: float,
    k: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute the truncated log-price domain bounds.

    The domain is centred at the at-the-money point :math:`x = 0`
    (since :math:`x = \log(S/K)` and ATM means :math:`S = K`).
    Each dimension is truncated at

    .. math::

        x_i^{\min} = -k\,\sigma_i\,\sqrt{T},
        \quad
        x_i^{\max} = +k\,\sigma_i\,\sqrt{T}

    Parameters
    ----------
    sigma : np.ndarray
        Volatilities, shape ``(d,)``.
    T : float
        Maturity.
    k : float
        Number of standard deviations for truncation (default 4).

    Returns
    -------
    x_min, x_max : np.ndarray
        Lower and upper bounds, each of shape ``(d,)``.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    half_width = k * sigma * np.sqrt(T)
    return -half_width, half_width


def evaluate_boundary_condition_lower(
    u_theta: torch.Tensor,
) -> torch.Tensor:
    r"""Boundary condition loss for :math:`x_i \to -\infty`.

    For call-type payoffs the option is worthless when any asset goes to
    zero, so the boundary condition is :math:`u \to 0`.

    Parameters
    ----------
    u_theta : torch.Tensor
        Network predictions at boundary points, shape ``(N, 1)``.

    Returns
    -------
    torch.Tensor
        Pointwise squared boundary residual, shape ``(N, 1)``.
    """
    return u_theta**2


def evaluate_linearity_condition_upper(
    d2u_dx2: torch.Tensor,
) -> torch.Tensor:
    r"""Boundary condition for :math:`x_i \to +\infty`.

    For deep-in-the-money options the price becomes approximately linear
    in the asset price, so :math:`\partial^2 u / \partial x_i^2 \to 0`.

    Parameters
    ----------
    d2u_dx2 : torch.Tensor
        Second derivative of the network output w.r.t. the boundary
        coordinate, shape ``(N, 1)``.

    Returns
    -------
    torch.Tensor
        Pointwise squared linearity residual, shape ``(N, 1)``.
    """
    return d2u_dx2**2
