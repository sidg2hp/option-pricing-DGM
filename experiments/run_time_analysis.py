"""Time-to-Solution Analysis: DGM Inference vs Monte Carlo simulation."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from evaluation.monte_carlo import MonteCarloPricer
from utils.math_utils import build_equicorrelation_matrix


def load_trained_model(d=5, hidden=512):
    """Load the trained DGM model for dimension d."""
    sigma_list = [0.2] * d
    rho_mat = build_equicorrelation_matrix(d, 0.3).tolist()
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


def main():
    d = 5
    model, config, device = load_trained_model(d=d)
    model.eval()

    n_options = 10000
    print(f"\nBenchmarking pricing of {n_options} options simultaneously (d={d})...")

    # Generate random initial states
    rng = np.random.RandomState(42)
    S_test = np.exp(rng.randn(n_options, d) * 0.3)
    
    # ---------------------------------------------------------
    # 1. DGM Inference Time
    # ---------------------------------------------------------
    x_test = np.log(S_test)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(n_options, 1, dtype=torch.float32, device=device)

    # Warmup
    with torch.no_grad():
        _ = model(t_t, x_t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        dgm_prices = model(t_t, x_t).cpu().numpy().ravel()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    dgm_time = time.time() - start_time
    print(f"DGM Inference Time: {dgm_time:.4f} seconds")

    # ---------------------------------------------------------
    # 2. Monte Carlo Time (100k paths per option)
    # ---------------------------------------------------------
    pricer = MonteCarloPricer()
    def payoff(S):
        return np.maximum(S.mean(axis=1) - config.market.K, 0.0)

    # We time the MC pricing (we don't need 10,000 to know it takes forever, 
    # but let's time a subset and extrapolate to be fair)
    subset_size = 100
    S_subset = S_test[:subset_size]

    start_time = time.time()
    _ = pricer.price_surface(
        S_subset, payoff, config.market.r, 
        np.array(config.market.sigma), np.array(config.market.rho), 
        config.market.T, n_paths=100_000
    )
    mc_subset_time = time.time() - start_time
    mc_extrapolated_time = mc_subset_time * (n_options / subset_size)
    print(f"Monte Carlo Time (Extrapolated for {n_options}): {mc_extrapolated_time:.2f} seconds")

    speedup = mc_extrapolated_time / dgm_time
    print(f"\nSpeedup Factor: {speedup:,.0f}x")

    # ---------------------------------------------------------
    # 3. Plotting
    # ---------------------------------------------------------
    os.makedirs("figures/publication", exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['DGM (Neural PDE)', 'Monte Carlo (100k paths)']
    times = [dgm_time, mc_extrapolated_time]
    
    bars = ax.bar(methods, times, color=['#2ca02c', '#1f77b4'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_yscale('log')
    ax.set_ylabel("Inference Time (seconds) - Log Scale", fontsize=14)
    ax.set_title(f"Time-to-Solution: Pricing {n_options:,} Options ($d={d}$)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add text labels on top of bars
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                f"{t:.3f}s" if t < 1 else f"{t:,.0f}s",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add speedup text
    ax.text(0.5, 0.8, f"Massive {speedup:,.0f}x Speedup!", 
            transform=ax.transAxes, ha='center', va='center', 
            fontsize=20, fontweight='bold', color='#d62728',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#d62728', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig("figures/publication/fig14_time_solution.pdf", bbox_inches='tight', dpi=300)
    plt.savefig("figures/publication/fig14_time_solution.png", bbox_inches='tight', dpi=300)
    print("Saved figures/publication/fig14_time_solution.pdf")


if __name__ == "__main__":
    main()
