"""Greeks accuracy comparison: DGM delta and gamma vs analytical values.

Runs on the 1D case where exact analytical formulas are available.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig, TrainingConfig
from evaluation.metrics import relative_l2_error
from models.model_factory import build_model
from pde.analytical import black_scholes_call_1d, black_scholes_delta_1d, black_scholes_gamma_1d
from pde.payoffs import get_payoff_fn
from plotting.greeks_plots import plot_delta_1d, plot_gamma_1d
from training.trainer import DGMTrainer
from utils.io_utils import save_json
from utils.random import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=50_000)
    args = parser.parse_args()

    K, r, sigma, T = 1.0, 0.05, 0.2, 1.0
    out_dir = "results/greeks"

    config = ExperimentConfig(
        name="greeks_1d",
        market=MarketConfig(d=1, sigma=[sigma], rho=[[1.0]], S0=[1.0]),
        training=TrainingConfig(
            n_steps=args.n_steps, lbfgs_finetune=True, lbfgs_steps=300,
        ),
        output_dir=out_dir,
    )

    payoff_fn, payoff_unclipped_fn = get_payoff_fn("basket_call", K)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S_test = np.linspace(0.5 * K, 1.5 * K, 50)
    analytical_prices = black_scholes_call_1d(S_test, K, r, sigma, T, np.zeros_like(S_test))
    mc_reference = {"S0_test": S_test.reshape(-1, 1), "mc_prices": analytical_prices}

    model = build_model(config, payoff_fn)
    trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, mc_reference)
    trainer.train()

    x_test = np.log(S_test / K).reshape(-1, 1)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(len(x_test), 1, dtype=torch.float32, device=device)

    dgm_delta = model.get_delta(t_t, x_t).cpu().numpy().ravel()
    dgm_gamma = model.get_gamma(t_t, x_t).cpu().numpy().ravel()

    anal_delta = black_scholes_delta_1d(S_test, K, r, sigma, T, np.zeros_like(S_test))
    anal_gamma = black_scholes_gamma_1d(S_test, K, r, sigma, T, np.zeros_like(S_test))

    delta_err = relative_l2_error(dgm_delta, anal_delta)
    gamma_err = relative_l2_error(dgm_gamma, anal_gamma)

    print(f"Delta relative L2 error: {delta_err:.4%}")
    print(f"Gamma relative L2 error: {gamma_err:.4%}")

    plot_delta_1d(
        model, K, T, device,
        lambda S: black_scholes_delta_1d(S, K, r, sigma, T, np.zeros_like(S)),
        out_dir,
    )
    plot_gamma_1d(
        model, K, T, device,
        lambda S: black_scholes_gamma_1d(S, K, r, sigma, T, np.zeros_like(S)),
        out_dir,
    )

    save_json({
        "delta_rel_l2_error": delta_err,
        "gamma_rel_l2_error": gamma_err,
    }, f"{out_dir}/greeks_results.json")


if __name__ == "__main__":
    main()
