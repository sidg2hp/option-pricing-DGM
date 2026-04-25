"""Ablation experiments: architecture, activation, constraint, and sampling.

Runs DGM with different configurations and compares errors.
Expanded from 4 to 7 ablations for paper-ready results.
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig, SamplerConfig, TrainingConfig
from evaluation.metrics import relative_l2_error
from evaluation.monte_carlo import MonteCarloPricer
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from training.trainer import DGMTrainer
from utils.io_utils import save_json
from utils.random import seed_everything


def _build_mc_reference(d: int, K: float):
    """Build MC reference for ablation experiments."""
    mc_pricer = MonteCarloPricer()
    rng = np.random.RandomState(123)
    S_test = K * np.exp(rng.randn(100, d) * 0.3)

    def mc_payoff(S_T):
        return np.maximum(S_T.mean(axis=1) - K, 0.0)

    mc_prices = mc_pricer.price_surface(
        S_test, mc_payoff, 0.05,
        np.array([0.2] * d),
        np.array([[0.3 if i != j else 1.0 for j in range(d)] for i in range(d)]),
        1.0, n_paths=500_000,
    )
    return S_test, mc_prices


def run_ablation(
    name: str,
    model_config: ModelConfig,
    sampler_config: SamplerConfig = SamplerConfig(),
    training_config: TrainingConfig | None = None,
    n_steps: int = 50_000,
) -> dict:
    """Run a single ablation experiment.

    Parameters
    ----------
    name : str
        Ablation identifier.
    model_config : ModelConfig
    sampler_config : SamplerConfig
    training_config : TrainingConfig or None
        If None, uses default (hard constraint, no L-BFGS).
    n_steps : int

    Returns
    -------
    dict
    """
    d = 2
    K = 1.0

    if training_config is None:
        training_config = TrainingConfig(n_steps=n_steps, lbfgs_finetune=False)

    config = ExperimentConfig(
        name=f"ablation_{name}",
        market=MarketConfig(
            d=d, sigma=[0.2, 0.2], rho=[[1.0, 0.3], [0.3, 1.0]],
            S0=[1.0, 1.0], payoff_type="basket_call",
        ),
        model=model_config,
        sampler=sampler_config,
        training=training_config,
        output_dir=f"results/ablations/{name}",
    )

    payoff_fn, payoff_unclipped_fn = get_payoff_fn("basket_call", K)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S_test, mc_prices = _build_mc_reference(d, K)
    mc_reference = {"S0_test": S_test, "mc_prices": mc_prices}

    model = build_model(config, payoff_fn)
    trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, mc_reference)
    results = trainer.train()

    x_test = np.log(S_test / K)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(len(S_test), 1, dtype=torch.float32, device=device)
    with torch.no_grad():
        dgm_prices = model(t_t, x_t).cpu().numpy().ravel()

    rel_l2 = relative_l2_error(dgm_prices, mc_prices)
    final_loss = results.get("final_loss", None)

    return {
        "name": name,
        "rel_l2_error": rel_l2,
        "final_loss": float(final_loss) if final_loss is not None else None,
        "train_time": results["train_time_seconds"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=50_000)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip ablations that already have results")
    args = parser.parse_args()

    seed_everything(42)

    # Define all ablation configurations
    ablations = [
        # 1. Baseline: DGM + tanh (hard constraint, risk-neutral sampling)
        ("dgm_tanh", {
            "model": ModelConfig(architecture="dgm", activation="tanh"),
        }),
        # 2. Architecture: MLP + tanh
        ("mlp_tanh", {
            "model": ModelConfig(architecture="mlp", activation="tanh"),
        }),
        # 3. Activation: DGM + softplus
        ("dgm_softplus", {
            "model": ModelConfig(architecture="dgm", activation="softplus"),
        }),
        # 4. Constraint: DGM + tanh + soft terminal constraint
        ("soft_constraint", {
            "model": ModelConfig(
                architecture="dgm", activation="tanh",
                use_hard_terminal_constraint=False,
            ),
            "training": TrainingConfig(
                n_steps=args.n_steps, lbfgs_finetune=False,
                lambda_terminal=10.0,
            ),
        }),
        # 5. Sampling: uniform
        ("uniform_sampling", {
            "model": ModelConfig(architecture="dgm", activation="tanh"),
            "sampler": SamplerConfig(sampler_type="uniform"),
        }),
        # 6. Sampling: risk-neutral (same as baseline, for completeness)
        ("risk_neutral_sampling", {
            "model": ModelConfig(architecture="dgm", activation="tanh"),
            "sampler": SamplerConfig(sampler_type="risk_neutral"),
        }),
        # 7. Sampling: adaptive
        ("adaptive_sampling", {
            "model": ModelConfig(architecture="dgm", activation="tanh"),
            "sampler": SamplerConfig(sampler_type="adaptive"),
        }),
    ]

    all_results = {}

    for name, cfg in ablations:
        result_path = f"results/ablations/{name}/ablation_result.json"
        legacy_path = f"results/ablations/{name}/ablation_{name}/results.json"
        if args.skip_existing and (os.path.exists(result_path) or os.path.exists(legacy_path)):
            print(f"\nSkipping {name} (found existing results)")
            existing = result_path if os.path.exists(result_path) else legacy_path
            with open(existing) as f:
                all_results[name] = json.load(f)
            continue

        print(f"\n{'='*50}")
        print(f"Ablation: {name}")
        print(f"{'='*50}")

        result = run_ablation(
            name,
            model_config=cfg["model"],
            sampler_config=cfg.get("sampler", SamplerConfig()),
            training_config=cfg.get("training", None),
            n_steps=args.n_steps,
        )
        all_results[name] = result
        # Save individual result
        save_json(result, result_path)

    save_json(all_results, "results/ablations/ablation_summary.json")

    print("\n" + "=" * 50)
    print("Ablation Summary")
    print("=" * 50)
    print(f"{'Name':<25s} {'Rel L2 Error':>14s} {'Final Loss':>14s} {'Time (s)':>10s}")
    for name, r in all_results.items():
        err_str = f"{r['rel_l2_error']:.4%}" if r.get('rel_l2_error') is not None else "NaN"
        loss_str = f"{r['final_loss']:.2e}" if r.get('final_loss') is not None else "N/A"
        t = r.get('train_time', r.get('train_time_seconds', 0))
        print(f"{name:<25s} {err_str:>14s} {loss_str:>14s} {t:10.1f}")


if __name__ == "__main__":
    main()
