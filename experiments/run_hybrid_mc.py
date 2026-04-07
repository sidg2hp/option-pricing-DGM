"""Hybrid DGM + MC control variate experiment.

Uses the DGM solution as a control variate to reduce MC variance.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from configs.base_config import ExperimentConfig, MarketConfig, TrainingConfig
from evaluation.monte_carlo import MonteCarloPricer
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from training.trainer import DGMTrainer
from utils.io_utils import save_json
from utils.math_utils import validate_correlation_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=50_000)
    parser.add_argument("--n_mc", type=int, default=100_000)
    args = parser.parse_args()

    K, r, T, d = 1.0, 0.05, 1.0, 2
    sigma = [0.2, 0.2]
    rho = [[1.0, 0.3], [0.3, 1.0]]
    out_dir = "results/hybrid_mc"

    config = ExperimentConfig(
        name="hybrid_mc",
        market=MarketConfig(d=d, sigma=sigma, rho=rho, S0=[1.0, 1.0]),
        training=TrainingConfig(n_steps=args.n_steps, lbfgs_finetune=True, lbfgs_steps=200),
        output_dir=out_dir,
    )

    payoff_fn, payoff_unclipped_fn = get_payoff_fn("basket_call", K)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, payoff_fn)
    trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, None)
    trainer.train()

    sigma_np = np.array(sigma)
    rho_np = np.array(rho)
    L = validate_correlation_matrix(rho_np)
    S0 = np.array([1.0, 1.0])

    rng = np.random.RandomState(42)
    n_paths = args.n_mc
    eps = rng.randn(n_paths, d)
    Z = eps @ L.T
    drift = (r - 0.5 * sigma_np**2) * T
    S_T = S0 * np.exp(drift + sigma_np * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T.mean(axis=1) - K, 0.0)
    discounted = np.exp(-r * T) * payoffs

    x_T = np.log(S_T / K)
    x_t = torch.tensor(x_T, dtype=torch.float32, device=device)
    t_t = torch.full((n_paths, 1), T * 0.999, dtype=torch.float32, device=device)
    with torch.no_grad():
        dgm_at_paths = model(t_t, x_t).cpu().numpy().ravel()

    dgm_mean = dgm_at_paths.mean()
    cov_XY = np.cov(discounted, dgm_at_paths)[0, 1]
    var_Y = np.var(dgm_at_paths)
    c_star = -cov_XY / var_Y if var_Y > 1e-15 else 0.0

    cv_estimate = discounted + c_star * (dgm_at_paths - dgm_mean)

    vanilla_price = np.mean(discounted)
    vanilla_se = np.std(discounted) / np.sqrt(n_paths)
    cv_price = np.mean(cv_estimate)
    cv_se = np.std(cv_estimate) / np.sqrt(n_paths)

    variance_reduction = 1 - (cv_se / vanilla_se)**2 if vanilla_se > 0 else 0

    print(f"Vanilla MC:  price={vanilla_price:.6f}, SE={vanilla_se:.6f}")
    print(f"CV MC:       price={cv_price:.6f}, SE={cv_se:.6f}")
    print(f"Variance reduction: {variance_reduction:.2%}")

    save_json({
        "vanilla_price": vanilla_price,
        "vanilla_se": vanilla_se,
        "cv_price": cv_price,
        "cv_se": cv_se,
        "variance_reduction": variance_reduction,
        "c_star": c_star,
    }, f"{out_dir}/hybrid_mc_results.json")


if __name__ == "__main__":
    main()
