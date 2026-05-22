"""Monte Carlo pricer for multi-asset European options under Black-Scholes.

Uses exact simulation of geometric Brownian motion (no Euler
discretisation error) with Cholesky-correlated Brownian increments.
Supports antithetic variates for variance reduction.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

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


class HybridMCControlVariate(MonteCarloPricer):
    """Monte Carlo engine enhanced with a Neural PDE Solver Control Variate.
    
    Uses a pre-trained Deep Galerkin Method (DGM) model to compute a control 
    variate for variance reduction.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device
        
    def price_with_cv(
        self,
        S0: np.ndarray,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        r: float,
        sigma: np.ndarray,
        rho: np.ndarray,
        T: float,
        K: float,
        n_paths: int = 1_000_000,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Compute the discounted expected payoff using DGM as a control variate.
        
        The estimator replaces the empirical sample mean with the analytical 
        PDE solution u(0, S_0) at t=0, via Feynman-Kac.
        """
        S0_np = np.asarray(S0, dtype=np.float64)
        sigma_np = np.asarray(sigma, dtype=np.float64)
        rho_np = np.asarray(rho, dtype=np.float64)
        d = len(S0_np)

        L = validate_correlation_matrix(rho_np)
        rng = np.random.RandomState(seed)

        eps = rng.randn(n_paths, d)
        Z = eps @ L.T

        drift = (r - 0.5 * sigma_np**2) * T
        diffusion = sigma_np * np.sqrt(T) * Z
        S_T = S0_np * np.exp(drift + diffusion)

        payoffs = payoff_fn(S_T)
        discounted = np.exp(-r * T) * payoffs
        
        # DGM predictions at terminal paths
        x_T = np.log(S_T / K)
        x_t = torch.tensor(x_T, dtype=torch.float32, device=self.device)
        t_t = torch.full((n_paths, 1), T * 0.999, dtype=torch.float32, device=self.device)
        
        # The true mathematical expected value of the DGM representation is its deterministic value at t=0
        x0 = np.log(S0_np / K).reshape(1, d)
        x0_t = torch.tensor(x0, dtype=torch.float32, device=self.device)
        t0_t = torch.zeros((1, 1), dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            dgm_at_paths = self.model(t_t, x_t).cpu().numpy().ravel()
            dgm_mean = self.model(t0_t, x0_t).item()

        # Control variate computation
        cov_XY = np.cov(discounted, dgm_at_paths)[0, 1]
        var_Y = np.var(dgm_at_paths)
        c_star = -cov_XY / var_Y if var_Y > 1e-15 else 0.0

        cv_estimate = discounted + c_star * (dgm_at_paths - dgm_mean)

        vanilla_price = float(np.mean(discounted))
        vanilla_se = float(np.std(discounted) / np.sqrt(n_paths))
        cv_price = float(np.mean(cv_estimate))
        cv_se = float(np.std(cv_estimate) / np.sqrt(n_paths))
        variance_reduction = 1 - (cv_se / vanilla_se)**2 if vanilla_se > 0 else 0

        return {
            "vanilla_price": vanilla_price,
            "vanilla_se": vanilla_se,
            "cv_price": cv_price,
            "cv_se": cv_se,
            "variance_reduction": float(variance_reduction),
            "c_star": float(c_star),
            "n_paths": n_paths,
        }

