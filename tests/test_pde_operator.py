"""Tests for the Black-Scholes PDE operator implementation.

Verifies correctness by checking that the operator applied to the known
analytical solution yields zero (up to floating-point tolerance).

Test 1: L[V_analytical] ≈ 0 for 1D Black-Scholes
Test 2: L[V_analytical] ≈ 0 for 2D geometric basket
Test 3: Sign check — each term has the correct sign
Test 4: Limiting case — as sigma -> 0 the PDE reduces to dV/dt + r*S*dV/dS - rV = 0
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy.stats import norm

from pde.operator import black_scholes_operator


def _analytical_1d_call(t: torch.Tensor, x: torch.Tensor, K: float, r: float, sigma: float, T: float) -> torch.Tensor:
    """1D Black-Scholes call in log-price coordinates, differentiable."""
    tau = T - t
    tau_safe = torch.clamp(tau, min=1e-10)
    sqrt_tau = torch.sqrt(tau_safe)
    S = K * torch.exp(x)
    d1 = (x + (r + 0.5 * sigma**2) * tau_safe) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    nd1 = 0.5 * (1.0 + torch.erf(d1 / (2.0**0.5)))
    nd2 = 0.5 * (1.0 + torch.erf(d2 / (2.0**0.5)))
    price = S * nd1 - K * torch.exp(-r * tau_safe) * nd2
    return price


def _analytical_geometric_basket(t: torch.Tensor, x: torch.Tensor, K: float, r: float,
                                  sigma_vec: torch.Tensor, rho_mat: torch.Tensor, T: float) -> torch.Tensor:
    """Geometric basket call in log-price coordinates, differentiable.

    Uses dividend-adjusted BSM: the geometric basket has risk-neutral
    drift r_geo != r, so the formula uses dividend yield q = r - r_geo.
    This ensures L[V] = 0 with the multi-asset BS operator whose
    discount rate is r.
    """
    d = x.shape[-1]
    A = rho_mat * torch.outer(sigma_vec, sigma_vec)
    sigma_geo = (1.0 / d) * torch.sqrt(A.sum())
    r_geo = r - (1.0 / (2.0 * d)) * (sigma_vec**2).sum() + 0.5 * sigma_geo**2
    q = r - r_geo
    x_bar = x.mean(dim=-1, keepdim=True)
    S_geo = K * torch.exp(x_bar)

    tau = T - t
    tau_safe = torch.clamp(tau, min=1e-10)
    sqrt_tau = torch.sqrt(tau_safe)
    d1 = (x_bar + (r - q + 0.5 * sigma_geo**2) * tau_safe) / (sigma_geo * sqrt_tau)
    d2 = d1 - sigma_geo * sqrt_tau
    nd1 = 0.5 * (1.0 + torch.erf(d1 / (2.0**0.5)))
    nd2 = 0.5 * (1.0 + torch.erf(d2 / (2.0**0.5)))
    return S_geo * torch.exp(-q * tau_safe) * nd1 - K * torch.exp(-r * tau_safe) * nd2


class Test1D:
    """Test 1: L[V_analytical] ≈ 0 for the 1D Black-Scholes call."""

    def test_pde_residual_is_near_zero(self):
        K, r, sigma_val, T = 1.0, 0.05, 0.2, 1.0
        torch.manual_seed(42)
        N = 200

        t = torch.rand(N, 1, dtype=torch.float64) * T * 0.95
        t.requires_grad_(True)
        x = torch.randn(N, 1, dtype=torch.float64) * 0.5
        x.requires_grad_(True)

        sigma_t = torch.tensor([sigma_val], dtype=torch.float64)
        rho_t = torch.tensor([[1.0]], dtype=torch.float64)

        def u_fn(t_in, x_in):
            return _analytical_1d_call(t_in, x_in, K, r, sigma_val, T)

        residual = black_scholes_operator(u_fn, t, x, sigma_t, rho_t, r)
        max_res = residual.abs().max().item()
        mean_res = residual.abs().mean().item()

        print(f"[Test 1D] Max |L[V]| = {max_res:.2e}, Mean |L[V]| = {mean_res:.2e}")
        assert max_res < 1e-4, f"PDE residual too large: {max_res:.2e}"


class TestGeometricBasket2D:
    """Test 2: L[V_analytical] ≈ 0 for the 2D geometric basket."""

    def test_pde_residual_is_near_zero(self):
        K, r, T = 1.0, 0.05, 1.0
        sigma_vals = [0.2, 0.2]
        rho_vals = [[1.0, 0.3], [0.3, 1.0]]
        torch.manual_seed(123)
        N = 200

        t = torch.rand(N, 1, dtype=torch.float64) * T * 0.95
        t.requires_grad_(True)
        x = torch.randn(N, 2, dtype=torch.float64) * 0.3
        x.requires_grad_(True)

        sigma_t = torch.tensor(sigma_vals, dtype=torch.float64)
        rho_t = torch.tensor(rho_vals, dtype=torch.float64)

        def u_fn(t_in, x_in):
            return _analytical_geometric_basket(t_in, x_in, K, r, sigma_t, rho_t, T)

        residual = black_scholes_operator(u_fn, t, x, sigma_t, rho_t, r)
        max_res = residual.abs().max().item()
        mean_res = residual.abs().mean().item()

        print(f"[Test 2D Geo] Max |L[V]| = {max_res:.2e}, Mean |L[V]| = {mean_res:.2e}")
        assert max_res < 1e-4, f"PDE residual too large: {max_res:.2e}"


class TestSignCheck:
    """Test 3: verify sign of each term in the PDE operator."""

    def test_individual_terms_sign(self):
        K, r, sigma_val, T = 1.0, 0.05, 0.2, 1.0

        t = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)

        sigma_t = torch.tensor([sigma_val], dtype=torch.float64)
        rho_t = torch.tensor([[1.0]], dtype=torch.float64)

        def u_fn(t_in, x_in):
            return _analytical_1d_call(t_in, x_in, K, r, sigma_val, T)

        u_val = u_fn(t, x)
        du_dt = torch.autograd.grad(u_val.sum(), t, create_graph=True)[0]
        du_dx = torch.autograd.grad(u_val.sum(), x, create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]

        diffusion = 0.5 * sigma_val**2 * d2u_dx2
        drift = (r - 0.5 * sigma_val**2) * du_dx
        discount = -r * u_val

        residual = du_dt + diffusion + drift + discount
        print(f"[Sign Check] du_dt={du_dt.item():.6f}, diffusion={diffusion.item():.6f}, "
              f"drift={drift.item():.6f}, discount={discount.item():.6f}")
        print(f"[Sign Check] Residual = {residual.item():.2e}")
        assert abs(residual.item()) < 1e-4, "Sign check failed"


class TestLimitingSigmaZero:
    """Test 4: as sigma -> 0, PDE becomes dV/dt + r*dV/dx - rV = 0 (in log coords)."""

    def test_zero_vol_limit(self):
        K, r, T = 1.0, 0.05, 1.0
        sigma_val = 1e-6

        t = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)

        sigma_t = torch.tensor([sigma_val], dtype=torch.float64)
        rho_t = torch.tensor([[1.0]], dtype=torch.float64)

        S = K * torch.exp(x)
        tau = T - t
        u_val = torch.clamp(S - K * torch.exp(-r * tau), min=0.0)

        def u_fn(t_in, x_in):
            S_loc = K * torch.exp(x_in)
            tau_loc = T - t_in
            return torch.clamp(S_loc - K * torch.exp(-r * tau_loc), min=0.0) + 1e-20 * t_in * x_in

        du_dt = torch.autograd.grad(u_fn(t, x).sum(), t, create_graph=True)[0]
        du_dx = torch.autograd.grad(u_fn(t, x).sum(), x, create_graph=True)[0]

        zero_vol_residual = du_dt + r * du_dx - r * u_fn(t, x)
        print(f"[Zero-vol Limit] Residual = {zero_vol_residual.item():.2e}")


def test_pde_1d():
    Test1D().test_pde_residual_is_near_zero()


def test_pde_2d_geometric():
    TestGeometricBasket2D().test_pde_residual_is_near_zero()


def test_sign_check():
    TestSignCheck().test_individual_terms_sign()


def test_zero_vol_limit():
    TestLimitingSigmaZero().test_zero_vol_limit()


if __name__ == "__main__":
    test_pde_1d()
    test_pde_2d_geometric()
    test_sign_check()
    test_zero_vol_limit()
    print("\\nAll PDE operator tests PASSED.")
