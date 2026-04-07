"""Tests for the Monte Carlo pricer.

Verifies MC convergence against the analytical geometric basket call
price and the 1D Black-Scholes formula.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from evaluation.monte_carlo import MonteCarloPricer
from pde.analytical import black_scholes_call_1d, geometric_basket_call


def test_mc_vs_analytical_1d():
    """MC price of 1D call should match Black-Scholes formula."""
    S0 = np.array([1.0])
    K, r, sigma_val, T = 1.0, 0.05, 0.2, 1.0

    def payoff_fn(S_T):
        return np.maximum(S_T[:, 0] - K, 0.0)

    pricer = MonteCarloPricer()
    result = pricer.price(
        S0, payoff_fn, r,
        np.array([sigma_val]),
        np.array([[1.0]]),
        T, n_paths=1_000_000, seed=42,
    )

    analytical = black_scholes_call_1d(
        np.array([1.0]), K, r, sigma_val, T, np.array([0.0])
    )[0]

    err = abs(result["price"] - analytical) / analytical
    print(f"[MC 1D] MC={result['price']:.6f}, Analytical={analytical:.6f}, "
          f"RelErr={err:.4%}, StdErr={result['std_error']:.6f}")
    assert err < 0.01, f"MC 1D relative error too large: {err:.4%}"


def test_mc_vs_analytical_geometric_basket():
    """MC price of geometric basket should match closed-form."""
    S0 = np.array([1.0, 1.0])
    K, r, T = 1.0, 0.05, 1.0
    sigma = np.array([0.2, 0.2])
    rho = np.array([[1.0, 0.3], [0.3, 1.0]])

    def payoff_fn(S_T):
        geo_avg = np.prod(S_T, axis=1) ** (1.0 / S_T.shape[1])
        return np.maximum(geo_avg - K, 0.0)

    pricer = MonteCarloPricer()
    result = pricer.price(
        S0, payoff_fn, r, sigma, rho, T,
        n_paths=1_000_000, seed=42,
    )

    analytical = geometric_basket_call(S0, K, r, sigma, rho, T)

    err = abs(result["price"] - analytical) / analytical
    print(f"[MC Geo 2D] MC={result['price']:.6f}, Analytical={analytical:.6f}, "
          f"RelErr={err:.4%}, StdErr={result['std_error']:.6f}")
    assert err < 0.01, f"MC geometric basket relative error too large: {err:.4%}"


def test_mc_confidence_interval():
    """Analytical price should fall within 95% MC confidence interval."""
    S0 = np.array([1.0])
    K, r, sigma_val, T = 1.0, 0.05, 0.2, 1.0

    def payoff_fn(S_T):
        return np.maximum(S_T[:, 0] - K, 0.0)

    pricer = MonteCarloPricer()
    result = pricer.price(
        S0, payoff_fn, r,
        np.array([sigma_val]),
        np.array([[1.0]]),
        T, n_paths=1_000_000, seed=42,
    )

    analytical = black_scholes_call_1d(
        np.array([1.0]), K, r, sigma_val, T, np.array([0.0])
    )[0]

    print(f"[MC CI] Price={result['price']:.6f}, "
          f"CI=[{result['ci_lower']:.6f}, {result['ci_upper']:.6f}], "
          f"Analytical={analytical:.6f}")
    assert result["ci_lower"] <= analytical <= result["ci_upper"], \
        "Analytical price outside MC 95% CI"


if __name__ == "__main__":
    test_mc_vs_analytical_1d()
    test_mc_vs_analytical_geometric_basket()
    test_mc_confidence_interval()
    print("\nAll Monte Carlo tests PASSED.")
