"""Multi-asset Black-Scholes PDE operator in log-price coordinates.

Computes the differential operator

.. math::

    \\mathcal{L}[u] = \\frac{\\partial u}{\\partial t}
        + \\frac{1}{2}\\sum_{i,j} A_{ij}
            \\frac{\\partial^2 u}{\\partial x_i \\partial x_j}
        + \\sum_i \\mu_i \\frac{\\partial u}{\\partial x_i}
        - r\\,u

where :math:`A_{ij} = \\rho_{ij}\\sigma_i\\sigma_j` is the diffusion tensor
and :math:`\\mu_i = r - \\sigma_i^2/2` is the drift coefficient in log-price
coordinates.

For :math:`d \\le 5` the full Hessian is computed exactly via double
back-propagation.  For :math:`d > 5` Hutchinson's stochastic trace
estimator is used to avoid the :math:`O(d)` backward passes.
"""

from typing import Callable, Optional

import torch


def black_scholes_operator(
    u_theta: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    x: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
    r: float,
    n_hutchinson: int = 20,
) -> torch.Tensor:
    r"""Evaluate the Black-Scholes PDE residual :math:`\mathcal{L}[u_\theta]`.

    The Jacobian from raw price to log-price coordinates has been absorbed
    into the PDE:

    .. math::

        \mathcal{L}[u] = \frac{\partial u}{\partial t}
            + \frac{1}{2}\sum_{i,j}\rho_{ij}\sigma_i\sigma_j
                \frac{\partial^2 u}{\partial x_i\partial x_j}
            + \sum_i\!\left(r - \tfrac{\sigma_i^2}{2}\right)
                \frac{\partial u}{\partial x_i}
            - r\,u

    Parameters
    ----------
    u_theta : callable
        Network mapping ``(t, x) -> u``.  ``t`` has shape ``(N, 1)``
        and ``x`` has shape ``(N, d)``, both with ``requires_grad=True``.
        Output shape is ``(N, 1)``.
    t : torch.Tensor
        Time points, shape ``(N, 1)``, with ``requires_grad=True``.
    x : torch.Tensor
        Log-price points, shape ``(N, d)``, with ``requires_grad=True``.
    sigma : torch.Tensor
        Volatility vector, shape ``(d,)``.
    rho : torch.Tensor
        Correlation matrix, shape ``(d, d)``.
    r : float
        Risk-free rate.
    n_hutchinson : int
        Number of Rademacher vectors for Hutchinson's trace estimator
        (used only when ``d > 5``).

    Returns
    -------
    torch.Tensor
        PDE residual :math:`\mathcal{L}[u_\theta]`, shape ``(N, 1)``.
    """
    d = x.shape[-1]

    u_val = u_theta(t, x)

    du_dt = torch.autograd.grad(
        u_val, t, grad_outputs=torch.ones_like(u_val), create_graph=True
    )[0]

    du_dx = torch.autograd.grad(
        u_val, x, grad_outputs=torch.ones_like(u_val), create_graph=True
    )[0]

    drift_coeff = r - 0.5 * sigma**2
    first_order = (du_dx * drift_coeff.unsqueeze(0)).sum(dim=-1, keepdim=True)

    A = rho * torch.outer(sigma, sigma)

    if d <= 5:
        trace_term = _exact_diffusion_trace(du_dx, x, A, d)
    else:
        trace_term = _hutchinson_diffusion_trace(
            u_val, x, A, d, n_hutchinson
        )

    L_u = du_dt + 0.5 * trace_term + first_order - r * u_val
    return L_u


def _exact_diffusion_trace(
    du_dx: torch.Tensor,
    x: torch.Tensor,
    A: torch.Tensor,
    d: int,
) -> torch.Tensor:
    """Compute the diffusion trace exactly via d backward passes.

    Computes :math:`\\sum_{i,j} A_{ij} H_{ij}` where
    :math:`H_{ij} = \\partial^2 u / \\partial x_i \\partial x_j`.
    """
    N = x.shape[0]
    trace_term = torch.zeros(N, 1, dtype=x.dtype, device=x.device)

    for i in range(d):
        grad2_i = torch.autograd.grad(
            du_dx[:, i].sum(), x, create_graph=True
        )[0]
        for j in range(d):
            trace_term = trace_term + A[i, j] * grad2_i[:, j:j+1]

    return trace_term


def _hutchinson_diffusion_trace(
    u_val: torch.Tensor,
    x: torch.Tensor,
    A: torch.Tensor,
    d: int,
    n_hutchinson: int,
) -> torch.Tensor:
    r"""Estimate the diffusion trace via Hutchinson's method.

    Uses the identity

    .. math::

        \text{tr}(A \cdot H) = \mathbb{E}_v\bigl[v^\top A\, H\, v\bigr]

    where :math:`v \sim \text{Rademacher}(\pm 1)^d`.  The matrix-vector
    product :math:`H v` is computed with a single forward-over-backward
    pass.  Variance is :math:`O(1/m)` with :math:`m` samples.
    """
    N = x.shape[0]
    trace_est = torch.zeros(N, 1, dtype=x.dtype, device=x.device)

    du_dx = torch.autograd.grad(
        u_val, x, grad_outputs=torch.ones_like(u_val), create_graph=True
    )[0]

    for _ in range(n_hutchinson):
        v = torch.randint(0, 2, (N, d), device=x.device, dtype=x.dtype) * 2 - 1

        Hv = torch.autograd.grad(
            (du_dx * v).sum(), x, create_graph=True
        )[0]

        Av = (A.unsqueeze(0) * v.unsqueeze(1)).sum(dim=-1)
        trace_est = trace_est + (Av * Hv).sum(dim=-1, keepdim=True)

    return trace_est / n_hutchinson
