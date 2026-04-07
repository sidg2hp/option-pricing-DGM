"""Payoff functions for multi-asset European options in log-price coordinates.

All functions operate in log-price coordinates :math:`x_i = \\log(S_i / K)`,
so that :math:`S_i = K \\exp(x_i)`.  The smoothed payoff wrapper provides
a differentiable approximation to the ``max(f, 0)`` kink for training
stability.
"""

from typing import Callable, Optional

import torch


def basket_call_payoff(
    x: torch.Tensor,
    K: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Arithmetic basket call payoff in log-price coordinates.

    .. math::

        \Phi(\mathbf{x}) = K \left(\sum_i w_i e^{x_i} - 1\right)^+

    Parameters
    ----------
    x : torch.Tensor
        Log-price coordinates, shape ``(N, d)``.
    K : float
        Strike price.
    weights : torch.Tensor or None
        Asset weights summing to 1, shape ``(d,)``.
        Defaults to equal weights ``1/d``.

    Returns
    -------
    torch.Tensor
        Payoff values, shape ``(N, 1)``.
    """
    d = x.shape[-1]
    if weights is None:
        weights = torch.full((d,), 1.0 / d, dtype=x.dtype, device=x.device)
    weighted_sum = (weights.unsqueeze(0) * torch.exp(x)).sum(dim=-1, keepdim=True)
    return K * torch.clamp(weighted_sum - 1.0, min=0.0)


def basket_call_payoff_unclipped(
    x: torch.Tensor,
    K: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unclipped arithmetic basket payoff (before max(., 0))."""
    d = x.shape[-1]
    if weights is None:
        weights = torch.full((d,), 1.0 / d, dtype=x.dtype, device=x.device)
    weighted_sum = (weights.unsqueeze(0) * torch.exp(x)).sum(dim=-1, keepdim=True)
    return K * (weighted_sum - 1.0)


def geometric_basket_payoff(x: torch.Tensor, K: float) -> torch.Tensor:
    r"""Geometric basket call payoff in log-price coordinates.

    .. math::

        \Phi(\mathbf{x}) = K \left(e^{\bar{x}} - 1\right)^+

    where :math:`\bar{x} = \frac{1}{d}\sum_i x_i`.
    Has a closed-form Black-Scholes price for validation.

    Parameters
    ----------
    x : torch.Tensor
        Log-price coordinates, shape ``(N, d)``.
    K : float
        Strike price.

    Returns
    -------
    torch.Tensor
        Payoff values, shape ``(N, 1)``.
    """
    x_bar = x.mean(dim=-1, keepdim=True)
    return K * torch.clamp(torch.exp(x_bar) - 1.0, min=0.0)


def geometric_basket_payoff_unclipped(x: torch.Tensor, K: float) -> torch.Tensor:
    """Unclipped geometric basket payoff (before max(., 0))."""
    x_bar = x.mean(dim=-1, keepdim=True)
    return K * (torch.exp(x_bar) - 1.0)


def max_call_payoff(x: torch.Tensor, K: float) -> torch.Tensor:
    r"""Best-of (max-call) payoff in log-price coordinates.

    .. math::

        \Phi(\mathbf{x}) = K \left(\max_i e^{x_i} - 1\right)^+

    Parameters
    ----------
    x : torch.Tensor
        Log-price coordinates, shape ``(N, d)``.
    K : float
        Strike price.

    Returns
    -------
    torch.Tensor
        Payoff values, shape ``(N, 1)``.
    """
    max_exp = torch.exp(x).max(dim=-1, keepdim=True).values
    return K * torch.clamp(max_exp - 1.0, min=0.0)


def max_call_payoff_unclipped(x: torch.Tensor, K: float) -> torch.Tensor:
    """Unclipped max-call payoff."""
    max_exp = torch.exp(x).max(dim=-1, keepdim=True).values
    return K * (max_exp - 1.0)


def spread_option_payoff(x: torch.Tensor, K: float) -> torch.Tensor:
    r"""Spread option payoff for :math:`d=2` in log-price coordinates.

    .. math::

        \Phi(\mathbf{x}) = \left(K e^{x_1} - K e^{x_2} - K\right)^+

    Parameters
    ----------
    x : torch.Tensor
        Log-price coordinates, shape ``(N, 2)``.
    K : float
        Strike price.

    Returns
    -------
    torch.Tensor
        Payoff values, shape ``(N, 1)``.

    Raises
    ------
    ValueError
        If ``x`` does not have exactly 2 asset dimensions.
    """
    if x.shape[-1] != 2:
        raise ValueError(f"Spread option requires d=2, got d={x.shape[-1]}")
    f = K * (torch.exp(x[:, 0:1]) - torch.exp(x[:, 1:2]) - 1.0)
    return torch.clamp(f, min=0.0)


def spread_option_payoff_unclipped(x: torch.Tensor, K: float) -> torch.Tensor:
    """Unclipped spread option payoff."""
    if x.shape[-1] != 2:
        raise ValueError(f"Spread option requires d=2, got d={x.shape[-1]}")
    return K * (torch.exp(x[:, 0:1]) - torch.exp(x[:, 1:2]) - 1.0)


def smoothed_payoff(
    payoff_unclipped_fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    eps: float,
    **kwargs,
) -> torch.Tensor:
    r"""Smooth approximation of :math:`\max(f(x), 0)`.

    Uses the softplus-like smoothing:

    .. math::

        \Phi_\epsilon(\mathbf{x}) =
        \frac{1}{2}\bigl(f(\mathbf{x})
            + \sqrt{f(\mathbf{x})^2 + \epsilon^2}\bigr)

    This converges to :math:`\max(f, 0)` as :math:`\epsilon \to 0`.

    Parameters
    ----------
    payoff_unclipped_fn : callable
        Function returning the unclipped payoff :math:`f(\mathbf{x})`
        (before applying ``max(., 0)``).
    x : torch.Tensor
        Log-price coordinates, shape ``(N, d)``.
    eps : float
        Smoothing parameter.  Annealed from a positive value to 0
        during training.
    **kwargs
        Extra keyword arguments forwarded to *payoff_unclipped_fn*.

    Returns
    -------
    torch.Tensor
        Smoothed payoff values, shape ``(N, 1)``.
    """
    f = payoff_unclipped_fn(x, **kwargs)
    if eps <= 0.0:
        return torch.clamp(f, min=0.0)
    return 0.5 * (f + torch.sqrt(f**2 + eps**2))


PAYOFF_REGISTRY = {
    "basket_call": (basket_call_payoff, basket_call_payoff_unclipped),
    "geometric_basket": (geometric_basket_payoff, geometric_basket_payoff_unclipped),
    "max_call": (max_call_payoff, max_call_payoff_unclipped),
    "spread": (spread_option_payoff, spread_option_payoff_unclipped),
}


def get_payoff_fn(payoff_type: str, K: float):
    """Return ``(payoff_fn, payoff_unclipped_fn)`` for the given payoff type.

    Parameters
    ----------
    payoff_type : str
        One of ``basket_call``, ``geometric_basket``, ``max_call``, ``spread``.
    K : float
        Strike price (curried into the returned callables).

    Returns
    -------
    tuple of callable
        ``(payoff_fn(x), unclipped_fn(x))`` both expecting a tensor ``x``
        of shape ``(N, d)``.
    """
    if payoff_type not in PAYOFF_REGISTRY:
        raise ValueError(
            f"Unknown payoff type '{payoff_type}'. "
            f"Choose from {list(PAYOFF_REGISTRY.keys())}"
        )
    clipped, unclipped = PAYOFF_REGISTRY[payoff_type]
    return (
        lambda x, _K=K, _fn=clipped: _fn(x, _K),
        lambda x, _K=K, _fn=unclipped: _fn(x, _K),
    )
