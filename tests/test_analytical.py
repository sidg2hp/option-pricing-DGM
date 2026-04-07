"""Tests for analytical pricing formulas against known table values."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pde.analytical import (
    black_scholes_call_1d,
    geometric_basket_call,
    black_scholes_delta_1d,
    black_scholes_gamma_1d,
)


def test_bs_1d_known_value():
    """Verify BS call against a known textbook value.

    S=100, K=100, r=0.05, sigma=0.2, T=1 -> V ≈ 10.4506
    """
    V = black_scholes_call_1d(
        np.array([100.0]), 100.0, 0.05, 0.2, 1.0, np.array([0.0])
    )[0]
    assert abs(V - 10.4506) < 0.01, f"BS call = {V:.4f}, expected ~10.4506"


def test_bs_1d_at_maturity():
    """At t=T, price equals intrinsic value."""
    V_itm = black_scholes_call_1d(
        np.array([110.0]), 100.0, 0.05, 0.2, 1.0, np.array([1.0])
    )[0]
    assert abs(V_itm - 10.0) < 1e-6

    V_otm = black_scholes_call_1d(
        np.array([90.0]), 100.0, 0.05, 0.2, 1.0, np.array([1.0])
    )[0]
    assert abs(V_otm) < 1e-6


def test_bs_delta_bounds():
    """Delta should be between 0 and 1 for calls."""
    S = np.linspace(50, 150, 100)
    delta = black_scholes_delta_1d(S, 100.0, 0.05, 0.2, 1.0, np.zeros(100))
    assert np.all(delta >= -1e-6) and np.all(delta <= 1.0 + 1e-6)


def test_bs_gamma_positive():
    """Gamma should be positive for calls."""
    S = np.linspace(60, 140, 100)
    gamma = black_scholes_gamma_1d(S, 100.0, 0.05, 0.2, 1.0, np.zeros(100))
    assert np.all(gamma >= -1e-6)


def test_geometric_basket_reduces_to_1d():
    """For d=1, geometric basket should equal the standard BS call."""
    S0 = np.array([100.0])
    bs = black_scholes_call_1d(S0, 100.0, 0.05, 0.2, 1.0, np.array([0.0]))[0]
    geo = geometric_basket_call(S0, 100.0, 0.05, np.array([0.2]), np.array([[1.0]]), 1.0)
    assert abs(bs - geo) < 1e-6, f"1D: BS={bs:.6f}, Geo={geo:.6f}"


def test_geometric_basket_2d():
    """Geometric basket 2D should be positive and less than max single-asset call."""
    geo = geometric_basket_call(
        np.array([100.0, 100.0]), 100.0, 0.05,
        np.array([0.2, 0.2]), np.array([[1.0, 0.3], [0.3, 1.0]]), 1.0,
    )
    bs = black_scholes_call_1d(np.array([100.0]), 100.0, 0.05, 0.2, 1.0, np.array([0.0]))[0]
    assert 0 < geo < bs, f"Geometric basket {geo:.4f} should be in (0, {bs:.4f})"


if __name__ == "__main__":
    test_bs_1d_known_value()
    test_bs_1d_at_maturity()
    test_bs_delta_bounds()
    test_bs_gamma_positive()
    test_geometric_basket_reduces_to_1d()
    test_geometric_basket_2d()
    print("All analytical tests PASSED.")
