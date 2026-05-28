"""Extract and plot Greeks (Delta, Gamma) using automatic differentiation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from evaluation.monte_carlo import MonteCarloPricer
from utils.math_utils import build_equicorrelation_matrix


def load_trained_model(d=2):
    """Load the trained DGM model for dimension d."""
    import json
    sigma_list = [0.2] * d
    rho_mat = build_equicorrelation_matrix(d, 0.3).tolist()
    S0_list = [1.0] * d
    
    config_path = f"results/scaling/d_{d}/scaling_d{d}/config.json"
    hidden = 256
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            hidden = cfg.get("model", {}).get("hidden_size", 256)
    S0_list = [1.0] * d

    config = ExperimentConfig(
        name=f"scaling_d{d}",
        market=MarketConfig(
            d=d, r=0.05, T=1.0, K=1.0,
            sigma=sigma_list, rho=rho_mat, S0=S0_list,
            payoff_type="basket_call",
        ),
        model=ModelConfig(
            architecture="dgm", hidden_size=hidden,
            num_dgm_layers=4, activation="tanh",
            use_hard_terminal_constraint=True,
        ),
        output_dir=f"results/scaling/d_{d}"
    )

    payoff_fn, _ = get_payoff_fn("basket_call", 1.0)
    model = build_model(config, payoff_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Note: Using the raw scaling_d{d} path or the base checkpoint path
    ckpt_path = f"results/scaling/d_{d}/scaling_d{d}/checkpoints/best_model.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"results/scaling/d_{d}/checkpoints/best_model.pt"
        
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"WARNING: Checkpoint {ckpt_path} not found. Using untrained model.")

    return model, config, device


def compute_dgm_greeks(model, S_test, device):
    """Compute Delta and Gamma via automatic differentiation."""
    # S_test is (N, d)
    model.eval()
    
    # We want derivatives w.r.t S, but model takes x = log(S)
    x = torch.tensor(np.log(S_test), dtype=torch.float32, device=device, requires_grad=True)
    t = torch.zeros(x.shape[0], 1, dtype=torch.float32, device=device)

    # Forward pass: V = u(t, x)
    V = model(t, x)

    # First derivative w.r.t x
    du_dx = torch.autograd.grad(
        V, x, grad_outputs=torch.ones_like(V), create_graph=True
    )[0]

    # Delta = dV/dS = (du/dx) * (1/S)
    S_tensor = torch.exp(x)
    delta = du_dx / S_tensor

    # We compute Gamma_11 (second derivative w.r.t S_1)
    # Gamma_11 = (1/S_1^2) * (d^2u/dx_1^2 - du/dx_1)
    du_dx1 = du_dx[:, 0:1]
    d2u_dx1dx = torch.autograd.grad(
        du_dx1, x, grad_outputs=torch.ones_like(du_dx1), retain_graph=True
    )[0]
    
    d2u_dx12 = d2u_dx1dx[:, 0:1]
    gamma_11 = (d2u_dx12 - du_dx1) / (S_tensor[:, 0:1] ** 2)

    return (
        V.detach().cpu().numpy().ravel(),
        delta[:, 0].detach().cpu().numpy().ravel(),    # Delta for asset 1
        gamma_11.detach().cpu().numpy().ravel()        # Gamma for asset 1
    )


def compute_mc_greeks(S_test, config, epsilon=0.01):
    """Compute finite difference Greeks using Monte Carlo."""
    pricer = MonteCarloPricer()
    
    def payoff(S):
        return np.maximum(S.mean(axis=1) - config.market.K, 0.0)

    # We shift ONLY the first asset by epsilon
    shift_up = S_test.copy()
    shift_up[:, 0] += epsilon
    
    shift_down = S_test.copy()
    shift_down[:, 0] -= epsilon

    print(f"MC computing center paths...")
    V_center = pricer.price_surface(
        S_test, payoff, config.market.r, 
        np.array(config.market.sigma), np.array(config.market.rho), 
        config.market.T, n_paths=2_000_000
    )
    
    print(f"MC computing up paths...")
    V_up = pricer.price_surface(
        shift_up, payoff, config.market.r, 
        np.array(config.market.sigma), np.array(config.market.rho), 
        config.market.T, n_paths=2_000_000
    )
    
    print(f"MC computing down paths...")
    V_down = pricer.price_surface(
        shift_down, payoff, config.market.r, 
        np.array(config.market.sigma), np.array(config.market.rho), 
        config.market.T, n_paths=2_000_000
    )

    mc_delta = (V_up - V_down) / (2 * epsilon)
    mc_gamma = (V_up - 2 * V_center + V_down) / (epsilon ** 2)

    return V_center, mc_delta, mc_gamma


def main():
    d = 2
    model, config, device = load_trained_model(d=d)

    # Test range: S1 varies from 0.5 to 1.5, S2 is fixed at 1.0
    S1_range = np.linspace(0.5, 1.5, 100)
    S_test = np.ones((len(S1_range), d))
    S_test[:, 0] = S1_range

    print("Computing DGM analytical Greeks...")
    dgm_price, dgm_delta, dgm_gamma = compute_dgm_greeks(model, S_test, device)

    # We compute MC Greeks at a coarser grid to save time
    coarse_idx = np.arange(0, 100, 5)
    S_test_coarse = S_test[coarse_idx]
    S1_coarse = S1_range[coarse_idx]
    
    print("Computing MC finite difference Greeks...")
    mc_price, mc_delta, mc_gamma = compute_mc_greeks(S_test_coarse, config, epsilon=0.01)

    os.makedirs("figures/publication", exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Price
    axes[0].plot(S1_range, dgm_price, 'b-', label='DGM (Exact)', linewidth=2)
    axes[0].plot(S1_coarse, mc_price, 'ro', label='MC Reference', markersize=5)
    axes[0].set_title(f"Option Price $V$ ($d={d}$)", fontsize=14)
    axes[0].set_xlabel("$S_1$ (Moneyness)", fontsize=12)
    axes[0].set_ylabel("Price", fontsize=12)
    axes[0].legend(fontsize=12)

    # 2. Delta
    axes[1].plot(S1_range, dgm_delta, 'g-', label='DGM $\Delta$ (Autograd)', linewidth=2)
    axes[1].plot(S1_coarse, mc_delta, 'ko', label='MC Finite Difference', markersize=5)
    axes[1].set_title("$\Delta = \partial V / \partial S_1$", fontsize=14)
    axes[1].set_xlabel("$S_1$ (Moneyness)", fontsize=12)
    axes[1].set_ylabel("Delta", fontsize=12)
    axes[1].legend(fontsize=12)

    # 3. Gamma
    axes[2].plot(S1_range, dgm_gamma, 'm-', label='DGM $\Gamma$ (Autograd)', linewidth=2)
    axes[2].plot(S1_coarse, mc_gamma, 'ko', label='MC Finite Difference', markersize=5)
    axes[2].set_title("$\Gamma = \partial^2 V / \partial S_1^2$", fontsize=14)
    axes[2].set_xlabel("$S_1$ (Moneyness)", fontsize=12)
    axes[2].set_ylabel("Gamma", fontsize=12)
    axes[2].legend(fontsize=12)

    plt.tight_layout()
    plt.savefig("figures/publication/fig13_greeks.pdf", bbox_inches='tight', dpi=300)
    plt.savefig("figures/publication/fig13_greeks.png", bbox_inches='tight', dpi=300)
    print("Saved figures/publication/fig13_greeks.pdf")


if __name__ == "__main__":
    main()
