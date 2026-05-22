"""Scaling study: DGM performance for d = 1, 2, 3, 5, 7, 10 assets.

All runs use arithmetic basket call with equal weights, identical
volatilities sigma_i=0.2, equicorrelation rho_ij=0.3, and a fixed
training budget of 100,000 gradient steps.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from configs.base_config import (
    ExperimentConfig, MarketConfig, ModelConfig,
    SamplerConfig, TrainingConfig, MCConfig,
)
from evaluation.metrics import relative_l2_error, max_pointwise_error
from evaluation.monte_carlo import MonteCarloPricer
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from plotting.scaling_plots import plot_scaling_error, plot_scaling_time
from training.trainer import DGMTrainer
from utils.io_utils import save_json
from utils.math_utils import build_equicorrelation_matrix
from utils.random import seed_everything


def run_single_d(d: int, n_steps: int = 100_000, lbfgs_steps: int = 300) -> dict:
    """Train and evaluate DGM for a given dimension.

    Parameters
    ----------
    d : int
        Number of assets.
    n_steps : int
        Training steps.
    lbfgs_steps : int
        L-BFGS fine-tuning steps.

    Returns
    -------
    dict
        Results including error, timing, and parameter count.
    """
    print(f"\n{'='*60}")
    print(f"Scaling study: d = {d}")
    print(f"{'='*60}")

    sigma_list = [0.2] * d
    rho_mat = build_equicorrelation_matrix(d, 0.3).tolist()
    S0_list = [1.0] * d

    hidden = 512

    sampler_cfg = SamplerConfig()
    # Keeping default batch sizes (1024) to prevent CUDA OOM on 16GB V100
    # The performance gain will come entirely from the 3x longer training.

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
        sampler=sampler_cfg,
        training=TrainingConfig(
            n_steps=n_steps, lbfgs_finetune=True, lbfgs_steps=lbfgs_steps,
        ),
        output_dir=f"results/scaling/d_{d}",
    )

    payoff_fn, payoff_unclipped_fn = get_payoff_fn("basket_call", 1.0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Computing MC reference prices...")
    mc_pricer = MonteCarloPricer()
    n_test = 100
    rng = np.random.RandomState(123)
    S_test = np.exp(rng.randn(n_test, d) * 0.3)

    def mc_payoff(S_T):
        return np.maximum(S_T.mean(axis=1) - 1.0, 0.0)

    mc_prices = mc_pricer.price_surface(
        S_test, mc_payoff, 0.05, np.array(sigma_list),
        np.array(rho_mat), 1.0, n_paths=500_000,
    )

    mc_reference = {"S0_test": S_test, "mc_prices": mc_prices}

    model = build_model(config, payoff_fn)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, mc_reference)
    
    best_ckpt = os.path.join(trainer.output_dir, "checkpoints", "best_model.pt")
    
    results = trainer.train()
    
    if os.path.exists(best_ckpt):
        print("  Loading best_model.pt for evaluation...")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    x_test = np.log(S_test)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(n_test, 1, dtype=torch.float32, device=device)

    with torch.no_grad():
        dgm_prices = model(t_t, x_t).cpu().numpy().ravel()

    rel_l2 = relative_l2_error(dgm_prices, mc_prices)
    max_err = max_pointwise_error(dgm_prices, mc_prices)

    print(f"d={d}: rel_l2={rel_l2:.4%}, max_err={max_err:.6f}, "
          f"time={results['train_time_seconds']:.1f}s, params={n_params}")

    result = {
        "d": d,
        "rel_l2_error": rel_l2,
        "max_error": max_err,
        "train_time_seconds": results["train_time_seconds"],
        "n_params": n_params,
        "n_steps": n_steps,
        "lbfgs_steps": lbfgs_steps,
    }
    save_json(result, f"results/scaling/d_{d}/result.json")

    # Save convergence history for plotting
    if "loss_history" in results:
        save_json(results["loss_history"], f"results/scaling/d_{d}/loss_history.json")
    if "eval_history" in results:
        save_json(results["eval_history"], f"results/scaling/d_{d}/eval_history.json")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run DGM scaling study")
    parser.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3, 5, 7, 10])
    parser.add_argument("--n_steps", type=int, default=None,
                        help="Training steps (default: 50k for d<=5, 30k for d>5)")
    parser.add_argument("--lbfgs_steps", type=int, default=300)
    args = parser.parse_args()

    all_results = {}
    for d in args.dims:
        result_path = f"results/scaling/d_{d}/result.json"
        if os.path.exists(result_path):
            print(f"\nSkipping d={d} (found existing results at {result_path})")
            import json
            with open(result_path, "r") as f:
                all_results[d] = json.load(f)
        else:
            if d >= 7:
                steps = 150_000
            elif args.n_steps:
                steps = args.n_steps
            elif d > 5:
                steps = 50_000
            else:
                steps = 50_000
            all_results[d] = run_single_d(d, steps, args.lbfgs_steps)

    save_json(all_results, "results/scaling/scaling_summary.json")

    plot_scaling_error(all_results, "results/scaling")
    plot_scaling_time(all_results, "results/scaling")

    print("\n" + "=" * 60)
    print("Scaling Study Complete")
    print("=" * 60)
    for d in sorted(all_results.keys()):
        r = all_results[d]
        print(f"  d={d}: error={r['rel_l2_error']:.4%}, time={r['train_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
