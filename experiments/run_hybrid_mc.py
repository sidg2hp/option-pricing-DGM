"""Hybrid DGM + MC control variate experiment across multiple dimensions.

Uses pre-trained DGM models as control variates to reduce MC variance.
Can load models from a prior scaling study, or train fresh ones.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import numpy as np
import torch

from configs.base_config import ExperimentConfig, MarketConfig, TrainingConfig
from evaluation.monte_carlo import MonteCarloPricer
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from training.trainer import DGMTrainer
from utils.io_utils import save_json
from utils.math_utils import build_equicorrelation_matrix, validate_correlation_matrix
from utils.random import seed_everything


def run_hybrid_mc_for_d(
    d: int,
    n_mc: int = 100_000,
    model_dir: str | None = None,
    n_steps: int = 50_000,
) -> dict:
    """Run hybrid MC for a single dimension.

    Parameters
    ----------
    d : int
        Number of assets.
    n_mc : int
        Number of MC paths.
    model_dir : str or None
        Path to directory containing a pre-trained model checkpoint.
        If None, trains a new model.
    n_steps : int
        Training steps if training from scratch.

    Returns
    -------
    dict
        Results including vanilla/CV prices, standard errors, and variance reduction.
    """
    K, r, T = 1.0, 0.05, 1.0
    sigma_list = [0.2] * d
    rho_mat = build_equicorrelation_matrix(d, 0.3).tolist()
    out_dir = f"results/hybrid_mc/d_{d}"
    os.makedirs(out_dir, exist_ok=True)

    config = ExperimentConfig(
        name=f"hybrid_mc_d{d}",
        market=MarketConfig(
            d=d, r=r, T=T, K=K,
            sigma=sigma_list, rho=rho_mat, S0=[1.0] * d,
            payoff_type="basket_call",
        ),
        training=TrainingConfig(n_steps=n_steps, lbfgs_finetune=True, lbfgs_steps=200),
        output_dir=out_dir,
    )

    payoff_fn, payoff_unclipped_fn = get_payoff_fn("basket_call", K)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, payoff_fn)

    # Try to load pre-trained model
    loaded = False
    if model_dir:
        ckpt_path = os.path.join(model_dir, "checkpoints", "best_model.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(model_dir, "checkpoints", "latest_model.pt")
        if os.path.exists(ckpt_path):
            print(f"  Loading pre-trained model from {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            loaded = True

    if not loaded:
        print(f"  Training new model for d={d} ({n_steps} steps)...")
        trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, None)
        trainer.train()

    model.eval()

    # Monte Carlo simulation
    sigma_np = np.array(sigma_list)
    rho_np = np.array(rho_mat)
    L = validate_correlation_matrix(rho_np)
    S0 = np.array([1.0] * d)

    rng = np.random.RandomState(42)
    eps = rng.randn(n_mc, d)
    Z = eps @ L.T
    drift = (r - 0.5 * sigma_np**2) * T
    S_T = S0 * np.exp(drift + sigma_np * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T.mean(axis=1) - K, 0.0)
    discounted = np.exp(-r * T) * payoffs

    # DGM predictions at terminal paths
    x_T = np.log(S_T / K)
    x_t = torch.tensor(x_T, dtype=torch.float32, device=device)
    t_t = torch.full((n_mc, 1), T * 0.999, dtype=torch.float32, device=device)
    with torch.no_grad():
        dgm_at_paths = model(t_t, x_t).cpu().numpy().ravel()

    # Control variate computation
    dgm_mean = dgm_at_paths.mean()
    cov_XY = np.cov(discounted, dgm_at_paths)[0, 1]
    var_Y = np.var(dgm_at_paths)
    c_star = -cov_XY / var_Y if var_Y > 1e-15 else 0.0

    cv_estimate = discounted + c_star * (dgm_at_paths - dgm_mean)

    vanilla_price = np.mean(discounted)
    vanilla_se = np.std(discounted) / np.sqrt(n_mc)
    cv_price = np.mean(cv_estimate)
    cv_se = np.std(cv_estimate) / np.sqrt(n_mc)

    variance_reduction = 1 - (cv_se / vanilla_se)**2 if vanilla_se > 0 else 0

    print(f"  d={d}: Vanilla SE={vanilla_se:.6f}, CV SE={cv_se:.6f}, "
          f"VR={variance_reduction:.2%}, c*={c_star:.4f}")

    result = {
        "d": d,
        "vanilla_price": float(vanilla_price),
        "vanilla_se": float(vanilla_se),
        "cv_price": float(cv_price),
        "cv_se": float(cv_se),
        "variance_reduction": float(variance_reduction),
        "c_star": float(c_star),
        "n_mc": n_mc,
    }
    save_json(result, f"{out_dir}/hybrid_mc_results.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--n_steps", type=int, default=50_000)
    parser.add_argument("--n_mc", type=int, default=100_000)
    parser.add_argument("--model_base", type=str, default="results/scaling",
                        help="Base dir for pre-trained models (expects d_*/scaling_d*/)")
    args = parser.parse_args()

    seed_everything(42)
    all_results = {}

    for d in args.dims:
        print(f"\n{'='*50}")
        print(f"Hybrid MC: d = {d}")
        print(f"{'='*50}")

        # Try to find pre-trained model from scaling study
        model_dir = None
        scaling_dir = os.path.join(args.model_base, f"d_{d}", f"scaling_d{d}")
        if os.path.isdir(scaling_dir):
            model_dir = scaling_dir

        all_results[str(d)] = run_hybrid_mc_for_d(d, args.n_mc, model_dir, args.n_steps)

    save_json(all_results, "results/hybrid_mc/hybrid_mc_summary.json")

    print("\n" + "=" * 50)
    print("Hybrid MC Summary")
    print("=" * 50)
    print(f"{'d':>4s} {'Vanilla SE':>12s} {'CV SE':>12s} {'VR':>10s} {'c*':>8s}")
    for d_str in sorted(all_results.keys(), key=int):
        r = all_results[d_str]
        print(f"{r['d']:4d} {r['vanilla_se']:12.6f} {r['cv_se']:12.8f} "
              f"{r['variance_reduction']:10.2%} {r['c_star']:8.4f}")


if __name__ == "__main__":
    main()
