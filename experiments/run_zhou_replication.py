"""Exact replication of Zhou et al. (2021).

Trains both the Zhou et al. MC+NN method and the DGM method on the
same test cases, then produces a side-by-side comparison table.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from baselines.zhou_et_al import ZhouEtAlPricer
from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig, TrainingConfig
from evaluation.monte_carlo import MonteCarloPricer
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from plotting.zhou_figures import (
    plot_zhou_figure_1_price_surface_2d,
    plot_zhou_figure_2_training_convergence,
    plot_zhou_figure_4_price_comparison_table,
    plot_zhou_figure_5_scatter_dgm_vs_mc,
    plot_zhou_figure_7_delta_surface,
)
from training.trainer import DGMTrainer
from utils.io_utils import save_json
from utils.math_utils import build_equicorrelation_matrix
from utils.random import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Zhou et al. (2021) replication")
    parser.add_argument("--d", type=int, default=2, help="Number of assets")
    parser.add_argument("--n_steps", type=int, default=200_000)
    args = parser.parse_args()

    d = args.d
    K = 100.0
    r = 0.05
    T = 1.0
    sigma_list = [0.2] * d
    rho_mat = build_equicorrelation_matrix(d, 0.3).tolist()
    S0_list = [K] * d

    out_dir = f"results/zhou_replication/d_{d}"

    config = ExperimentConfig(
        name=f"zhou_d{d}",
        market=MarketConfig(
            d=d, r=r, T=T, K=K,
            sigma=sigma_list, rho=rho_mat, S0=S0_list,
            payoff_type="basket_call",
        ),
        model=ModelConfig(
            architecture="dgm", hidden_size=256, num_dgm_layers=4,
        ),
        training=TrainingConfig(n_steps=args.n_steps, lbfgs_finetune=True, lbfgs_steps=500),
        output_dir=out_dir,
    )

    payoff_fn, payoff_unclipped_fn = get_payoff_fn("basket_call", K)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Computing MC reference prices...")
    mc_pricer = MonteCarloPricer()

    S0_test_values = [80, 90, 100, 110, 120]
    S0_test = np.array([[s] * d for s in S0_test_values])
    S0_labels = [str(s) for s in S0_test_values]

    def mc_payoff(S_T):
        return np.maximum(S_T.mean(axis=1) - K, 0.0)

    mc_results = []
    for i, s0 in enumerate(S0_test):
        res = mc_pricer.price(
            s0, mc_payoff, r, np.array(sigma_list), np.array(rho_mat), T,
            n_paths=1_000_000, seed=i,
        )
        mc_results.append(res)

    mc_prices = np.array([r["price"] for r in mc_results])
    mc_errors = np.array([r["std_error"] for r in mc_results])

    mc_reference = {"S0_test": S0_test, "mc_prices": mc_prices}

    print("\n--- Training DGM ---")
    model = build_model(config, payoff_fn)
    trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, mc_reference)
    dgm_results = trainer.train()

    x_test = np.log(S0_test / K)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(len(S0_test), 1, dtype=torch.float32, device=device)
    with torch.no_grad():
        dgm_prices = model(t_t, x_t).cpu().numpy().ravel()

    print("\n--- Training Zhou et al. NN ---")

    def zhou_payoff(S_T):
        return np.maximum(S_T.mean(axis=1) - K, 0.0)

    zhou = ZhouEtAlPricer(config.market, zhou_payoff, seed=42)
    zhou_history = zhou.train(n_samples=100_000, n_epochs=200)
    zhou_prices = zhou.predict(S0_test)

    print("\n--- Price Comparison ---")
    print(f"{'S0':>8s} {'DGM':>10s} {'Zhou':>10s} {'MC':>10s} {'MC SE':>10s}")
    for i in range(len(S0_test_values)):
        print(f"{S0_test_values[i]:8d} {dgm_prices[i]:10.4f} {zhou_prices[i]:10.4f} "
              f"{mc_prices[i]:10.4f} {mc_errors[i]:10.6f}")

    plot_zhou_figure_4_price_comparison_table(
        dgm_prices, mc_prices, mc_errors, zhou_prices, S0_labels, out_dir,
    )
    plot_zhou_figure_2_training_convergence(dgm_results["loss_history"], out_dir)
    plot_zhou_figure_5_scatter_dgm_vs_mc(dgm_prices, mc_prices, out_dir)

    if d == 2:
        plot_zhou_figure_1_price_surface_2d(model, config.market, device, out_dir)
        plot_zhou_figure_7_delta_surface(model, K, device, out_dir)

    summary = {
        "d": d,
        "dgm_prices": dgm_prices.tolist(),
        "zhou_prices": zhou_prices.tolist(),
        "mc_prices": mc_prices.tolist(),
        "mc_errors": mc_errors.tolist(),
        "dgm_train_time": dgm_results["train_time_seconds"],
    }
    save_json(summary, f"{out_dir}/replication_summary.json")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
