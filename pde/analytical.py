"""Closed-form analytical solutions for European options under Black-Scholes.

Implements the exact Black-Scholes call price for a single asset and
the closed-form geometric basket call price that reduces to a 1-D
Black-Scholes problem.  These serve as ground-truth benchmarks for
validating the DGM solver.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_call_1d(
    S: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    T: float,
    t: np.ndarray,
) -> np.ndarray:
    r"""Exact Black-Scholes European call price for a single asset.

    .. math::

        V(t, S) = S\,\Phi(d_1) - K\,e^{-r(T-t)}\,\Phi(d_2)

    where

    .. math::

        d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}},
        \quad d_2 = d_1 - \sigma\sqrt{T-t}

    Parameters
    ----------
    S : np.ndarray
        Spot price(s), any shape.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Maturity.
    t : np.ndarray
        Current time(s), same shape as *S*.

    Returns
    -------
    np.ndarray
        Call price(s), same shape as *S*.
    """
    S = np.asarray(S, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    tau = T - t

    at_maturity = tau <= 1e-14
    tau_safe = np.where(at_maturity, 1.0, tau)

    sqrt_tau = np.sqrt(tau_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    price = S * norm.cdf(d1) - K * np.exp(-r * tau_safe) * norm.cdf(d2)
    payoff = np.maximum(S - K, 0.0)
    return np.where(at_maturity, payoff, price)


def geometric_basket_call(
    S0: np.ndarray,
    K: float,
    r: float,
    sigma: np.ndarray,
    rho: np.ndarray,
    T: float,
    t: float = 0.0,
) -> float:
    r"""Exact price of a geometric basket call via 1-D Black-Scholes reduction.

    The geometric basket has payoff
    :math:`\bigl(\prod_i S_i(T)\bigr)^{1/d} - K)^+`.
    Its dynamics reduce to a single GBM with effective parameters:

    .. math::

        \sigma_{\text{geo}} = \frac{1}{d}\sqrt{\sum_{i,j}
            \rho_{ij}\,\sigma_i\,\sigma_j},
        \quad
        r_{\text{geo}} = r - \frac{1}{2d}\sum_i \sigma_i^2
            + \frac{\sigma_{\text{geo}}^2}{2}

    and initial price :math:`S_{\text{geo}} = \bigl(\prod_i S_i(0)\bigr)^{1/d}`.

    Parameters
    ----------
    S0 : np.ndarray
        Initial asset prices, shape ``(d,)``.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : np.ndarray
        Volatilities, shape ``(d,)``.
    rho : np.ndarray
        Correlation matrix, shape ``(d, d)``.
    T : float
        Maturity.
    t : float
        Valuation time (default 0).

    Returns
    -------
    float
        Geometric basket call price.
    """
    S0 = np.asarray(S0, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    d = len(S0)

    sigma_geo = (1.0 / d) * np.sqrt(np.sum(rho * np.outer(sigma, sigma)))
    r_geo = r - (1.0 / (2.0 * d)) * np.sum(sigma**2) + 0.5 * sigma_geo**2
    S_geo = np.prod(S0) ** (1.0 / d)

    # The geometric basket is NOT a self-financing portfolio, so the
    # risk-neutral drift rate (r_geo) differs from the discount rate (r).
    # Use BSM with continuous dividend yield q = r - r_geo.
    tau = T - t
    if tau <= 1e-14:
        return float(max(S_geo - K, 0.0))
    q = r - r_geo
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S_geo / K) + (r - q + 0.5 * sigma_geo**2) * tau) / (
        sigma_geo * sqrt_tau
    )
    d2 = d1 - sigma_geo * sqrt_tau
    price = (
        S_geo * np.exp(-q * tau) * norm.cdf(d1)
        - K * np.exp(-r * tau) * norm.cdf(d2)
    )
    return float(price)


def black_scholes_delta_1d(
    S: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    T: float,
    t: np.ndarray,
) -> np.ndarray:
    r"""Analytical delta of a European call: :math:`\Delta = \Phi(d_1)`.

    Parameters
    ----------
    S : np.ndarray
        Spot price(s).
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Maturity.
    t : np.ndarray
        Current time(s).

    Returns
    -------
    np.ndarray
        Delta value(s).
    """
    S = np.asarray(S, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    tau = T - t
    at_maturity = tau <= 1e-14
    tau_safe = np.where(at_maturity, 1.0, tau)

    sqrt_tau = np.sqrt(tau_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * sqrt_tau)
    delta = norm.cdf(d1)
    return np.where(at_maturity, np.where(S > K, 1.0, 0.0), delta)


def black_scholes_gamma_1d(
    S: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    T: float,
    t: np.ndarray,
) -> np.ndarray:
    r"""Analytical gamma of a European call.

    .. math::

        \Gamma = \frac{\phi(d_1)}{S\,\sigma\,\sqrt{T-t}}

    where :math:`\phi` is the standard normal PDF.

    Parameters
    ----------
    S : np.ndarray
        Spot price(s).
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Maturity.
    t : np.ndarray
        Current time(s).

    Returns
    -------
    np.ndarray
        Gamma value(s).
    """
    S = np.asarray(S, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    tau = T - t
    at_maturity = tau <= 1e-14
    tau_safe = np.where(at_maturity, 1.0, tau)

    sqrt_tau = np.sqrt(tau_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * sqrt_tau)
    gamma = norm.pdf(d1) / (S * sigma * sqrt_tau)
    return np.where(at_maturity, 0.0, gamma)
