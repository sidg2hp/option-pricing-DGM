"""Validation experiments: verify DGM against analytical solutions.

1. 1D Black-Scholes: DGM vs analytical formula.
2. 2D Geometric Basket: DGM vs analytical formula.

PASS criteria:
  - 1D relative L2 error < 1%, delta error < 2%
  - 2D geometric basket relative L2 error < 2%
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig, SamplerConfig, TrainingConfig, MCConfig, load_experiment_config
from evaluation.metrics import relative_l2_error, max_pointwise_error
from evaluation.monte_carlo import MonteCarloPricer
from models.model_factory import build_model
from pde.analytical import black_scholes_call_1d, geometric_basket_call, black_scholes_delta_1d
from pde.payoffs import get_payoff_fn
from plotting.convergence import plot_training_convergence
from plotting.greeks_plots import plot_delta_1d
from plotting.price_surface import plot_price_surface_2d
from training.trainer import DGMTrainer
from utils.io_utils import save_json
from utils.random import seed_everything


class ValidationError(Exception):
    pass


def run_1d_validation(config: ExperimentConfig) -> dict:
    """Run 1D Black-Scholes validation.

    Parameters
    ----------
    config : ExperimentConfig

    Returns
    -------
    dict
        Validation results including errors.
    """
    print("=" * 60)
    print("1D Black-Scholes Validation")
    print("=" * 60)
    m = config.market

    payoff_fn, payoff_unclipped_fn = get_payoff_fn(m.payoff_type, m.K)

    device = torch.device(
        config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    S_test = np.linspace(0.5 * m.K, 1.5 * m.K, 50)
    x_test = np.log(S_test / m.K).reshape(-1, 1)
    t_test = np.zeros_like(x_test)

    analytical_prices = black_scholes_call_1d(S_test, m.K, m.r, m.sigma[0], m.T, t_test.ravel())

    mc_reference = {
        "S0_test": S_test.reshape(-1, 1),
        "mc_prices": analytical_prices,
    }

    model = build_model(config, payoff_fn)
    trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, mc_reference)
    results = trainer.train()

    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(len(x_test), 1, dtype=torch.float32, device=device)

    with torch.no_grad():
        dgm_prices = model(t_t, x_t).cpu().numpy().ravel()

    rel_l2 = relative_l2_error(dgm_prices, analytical_prices)
    max_err = max_pointwise_error(dgm_prices, analytical_prices)

    dgm_delta = model.get_delta(t_t, x_t).cpu().numpy().ravel()
    analytical_delta = black_scholes_delta_1d(S_test, m.K, m.r, m.sigma[0], m.T, t_test.ravel())
    delta_err = relative_l2_error(dgm_delta, analytical_delta)

    print(f"\n1D Results:")
    print(f"  Relative L2 error: {rel_l2:.4%}")
    print(f"  Max pointwise error: {max_err:.6f}")
    print(f"  Delta relative L2 error: {delta_err:.4%}")

    out_dir = str(config.output_dir) + "/" + config.name
    plot_training_convergence(results["loss_history"], out_dir)

    result = {
        "rel_l2_error": rel_l2,
        "max_error": max_err,
        "delta_error": delta_err,
        "train_time": results["train_time_seconds"],
        "passed": rel_l2 < 0.01 and delta_err < 0.02,
    }
    save_json(result, f"{out_dir}/validation_1d.json")

    if rel_l2 >= 0.01:
        raise ValidationError(f"1D validation FAILED: rel_l2={rel_l2:.4%} >= 1%")
    if delta_err >= 0.02:
        raise ValidationError(f"1D delta validation FAILED: delta_err={delta_err:.4%} >= 2%")

    print("  STATUS: PASSED")
    return result


def run_2d_geometric_validation(config: ExperimentConfig) -> dict:
    """Run 2D geometric basket validation.

    Parameters
    ----------
    config : ExperimentConfig

    Returns
    -------
    dict
        Validation results.
    """
    print("=" * 60)
    print("2D Geometric Basket Validation")
    print("=" * 60)
    m = config.market

    payoff_fn, payoff_unclipped_fn = get_payoff_fn(m.payoff_type, m.K)

    device = torch.device(
        config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    n_test = 200
    rng = np.random.RandomState(999)
    S_test = m.K * np.exp(rng.randn(n_test, m.d) * 0.3)

    analytical_prices = np.array([
        geometric_basket_call(S_test[i], m.K, m.r, np.array(m.sigma), np.array(m.rho), m.T)
        for i in range(n_test)
    ])

    mc_reference = {
        "S0_test": S_test,
        "mc_prices": analytical_prices,
    }

    model = build_model(config, payoff_fn)
    trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, mc_reference)
    results = trainer.train()

    x_test = np.log(S_test / m.K)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(n_test, 1, dtype=torch.float32, device=device)

    with torch.no_grad():
        dgm_prices = model(t_t, x_t).cpu().numpy().ravel()

    rel_l2 = relative_l2_error(dgm_prices, analytical_prices)
    max_err = max_pointwise_error(dgm_prices, analytical_prices)

    print(f"\n2D Geometric Basket Results:")
    print(f"  Relative L2 error: {rel_l2:.4%}")
    print(f"  Max pointwise error: {max_err:.6f}")

    out_dir = str(config.output_dir) + "/" + config.name
    plot_training_convergence(results["loss_history"], out_dir)

    if m.d == 2:
        plot_price_surface_2d(model, m.K, m.T, device, out_dir)

    result = {
        "rel_l2_error": rel_l2,
        "max_error": max_err,
        "train_time": results["train_time_seconds"],
        "passed": rel_l2 < 0.02,
    }
    save_json(result, f"{out_dir}/validation_2d_geometric.json")

    if rel_l2 >= 0.02:
        raise ValidationError(f"2D geometric validation FAILED: rel_l2={rel_l2:.4%} >= 2%")

    print("  STATUS: PASSED")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run DGM validation experiments")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--test", type=str, default="all",
                        choices=["1d", "2d_geometric", "all"])
    parser.add_argument("--n_steps", type=int, default=None, help="Override training steps")
    args = parser.parse_args()

    results = {}

    if args.test in ("1d", "all"):
        if args.config and "1d" in args.config:
            cfg = load_experiment_config(args.config)
        else:
            cfg = ExperimentConfig(
                name="validate_1d",
                market=MarketConfig(d=1, sigma=[0.2], rho=[[1.0]], S0=[1.0]),
                training=TrainingConfig(
                    n_steps=args.n_steps or 50000, eval_every=2500,
                    lbfgs_finetune=True, lbfgs_steps=300,
                ),
                output_dir="results/validation",
            )
        results["1d"] = run_1d_validation(cfg)

    if args.test in ("2d_geometric", "all"):
        if args.config and "2d" in args.config:
            cfg = load_experiment_config(args.config)
        else:
            cfg = ExperimentConfig(
                name="validate_2d_geometric",
                market=MarketConfig(
                    d=2, payoff_type="geometric_basket",
                    sigma=[0.2, 0.2], rho=[[1.0, 0.3], [0.3, 1.0]],
                    S0=[1.0, 1.0],
                ),
                training=TrainingConfig(
                    n_steps=args.n_steps or 50000, eval_every=2500,
                    lbfgs_finetune=True, lbfgs_steps=300,
                ),
                output_dir="results/validation",
            )
        results["2d_geometric"] = run_2d_geometric_validation(cfg)

    save_json(results, "results/validation/summary.json")
    print("\nAll validations PASSED.")
    return results


if __name__ == "__main__":
    main()
