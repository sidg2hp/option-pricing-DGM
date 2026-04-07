"""Monte Carlo pricer for multi-asset European options under Black-Scholes.

Uses exact simulation of geometric Brownian motion (no Euler
discretisation error) with Cholesky-correlated Brownian increments.
Supports antithetic variates for variance reduction.
"""

from typing import Callable, Dict, List, Optional

import numpy as np

from utils.math_utils import validate_correlation_matrix


class MonteCarloPricer:
    r"""Monte Carlo engine for multi-asset European option pricing.

    Under the risk-neutral measure, the exact terminal asset prices are:

    .. math::

        S_i(T) = S_i(0)\,\exp\!\bigl(
            (r - \tfrac{\sigma_i^2}{2})\,T
            + \sigma_i\,\sqrt{T}\,Z_i\bigr)

    where :math:`\mathbf{Z} \sim \mathcal{N}(0, \boldsymbol{\rho})`,
    generated via the Cholesky factor
    :math:`\mathbf{Z} = L\,\boldsymbol{\varepsilon}`,
    :math:`\boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)`.
    """

    def price(
        self,
        S0: np.ndarray,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        r: float,
        sigma: np.ndarray,
        rho: np.ndarray,
        T: float,
        n_paths: int = 1_000_000,
        use_antithetic: bool = True,
        seed: int = 0,
    ) -> Dict[str, float]:
        r"""Compute the discounted expected payoff via Monte Carlo.

        Parameters
        ----------
        S0 : np.ndarray
            Initial asset prices, shape ``(d,)``.
        payoff_fn : callable
            Payoff function mapping terminal prices ``S_T`` of shape
            ``(n_paths, d)`` to payoffs of shape ``(n_paths,)``.
        r : float
            Risk-free rate.
        sigma : np.ndarray
            Volatilities, shape ``(d,)``.
        rho : np.ndarray
            Correlation matrix, shape ``(d, d)``.
        T : float
            Maturity.
        n_paths : int
            Number of Monte Carlo paths.
        use_antithetic : bool
            Use antithetic variates for variance reduction.
        seed : int
            Random seed.

        Returns
        -------
        dict
            ``{"price", "std_error", "ci_lower", "ci_upper", "n_paths"}``.
        """
        S0 = np.asarray(S0, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        d = len(S0)

        L = validate_correlation_matrix(rho)
        rng = np.random.RandomState(seed)

        if use_antithetic:
            half = n_paths // 2
            eps = rng.randn(half, d)
            eps = np.concatenate([eps, -eps], axis=0)
        else:
            eps = rng.randn(n_paths, d)

        Z = eps @ L.T

        drift = (r - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T) * Z
        S_T = S0 * np.exp(drift + diffusion)

        payoffs = payoff_fn(S_T)
        discounted = np.exp(-r * T) * payoffs

        price = float(np.mean(discounted))
        std_error = float(np.std(discounted) / np.sqrt(len(discounted)))
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return {
            "price": price,
            "std_error": std_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_paths": len(discounted),
        }

    def price_surface(
        self,
        S0_grid: np.ndarray,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        r: float,
        sigma: np.ndarray,
        rho: np.ndarray,
        T: float,
        n_paths: int = 100_000,
        seed: int = 0,
    ) -> np.ndarray:
        """Compute option prices at multiple initial conditions.

        Parameters
        ----------
        S0_grid : np.ndarray
            Grid of initial price vectors, shape ``(M, d)``.
        payoff_fn : callable
            Payoff function on terminal prices.
        r, sigma, rho, T : market parameters
        n_paths : int
            Paths per initial condition.
        seed : int

        Returns
        -------
        np.ndarray
            Prices, shape ``(M,)``.
        """
        M = S0_grid.shape[0]
        prices = np.zeros(M)
        for m in range(M):
            result = self.price(
                S0_grid[m], payoff_fn, r, sigma, rho, T,
                n_paths=n_paths, seed=seed + m,
            )
            prices[m] = result["price"]
        return prices
