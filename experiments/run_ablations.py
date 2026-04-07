"""Ablation experiments: architecture, activation, sampling, and loss weights.

Runs DGM with different configurations and compares errors.
"""

import argparse
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
from plotting.zhou_figures import plot_dgm_ablation_architecture, plot_dgm_ablation_activation
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
    n_steps: int = 50_000,
) -> dict:
    """Run a single ablation experiment.

    Parameters
    ----------
    name : str
        Ablation identifier.
    model_config : ModelConfig
    sampler_config : SamplerConfig
    n_steps : int

    Returns
    -------
    dict
    """
    d = 2
    K = 1.0
    config = ExperimentConfig(
        name=f"ablation_{name}",
        market=MarketConfig(
            d=d, sigma=[0.2, 0.2], rho=[[1.0, 0.3], [0.3, 1.0]],
            S0=[1.0, 1.0], payoff_type="basket_call",
        ),
        model=model_config,
        sampler=sampler_config,
        training=TrainingConfig(n_steps=n_steps, lbfgs_finetune=False),
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
    return {"name": name, "rel_l2_error": rel_l2, "train_time": results["train_time_seconds"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=50_000)
    args = parser.parse_args()

    all_results = {}

    print("Ablation: DGM (tanh)")
    all_results["dgm_tanh"] = run_ablation(
        "dgm_tanh",
        ModelConfig(architecture="dgm", activation="tanh"),
        n_steps=args.n_steps,
    )

    print("Ablation: DGM (softplus)")
    all_results["dgm_softplus"] = run_ablation(
        "dgm_softplus",
        ModelConfig(architecture="dgm", activation="softplus"),
        n_steps=args.n_steps,
    )

    print("Ablation: MLP (tanh)")
    all_results["mlp_tanh"] = run_ablation(
        "mlp_tanh",
        ModelConfig(architecture="mlp", activation="tanh"),
        n_steps=args.n_steps,
    )

    print("Ablation: Adaptive sampling")
    all_results["adaptive"] = run_ablation(
        "adaptive",
        ModelConfig(architecture="dgm", activation="tanh"),
        sampler_config=SamplerConfig(sampler_type="adaptive"),
        n_steps=args.n_steps,
    )

    save_json(all_results, "results/ablations/ablation_summary.json")

    arch_results = {
        "DGM": all_results["dgm_tanh"]["rel_l2_error"],
        "MLP": all_results["mlp_tanh"]["rel_l2_error"],
    }
    plot_dgm_ablation_architecture(arch_results, "results/ablations")

    act_results = {
        "tanh": all_results["dgm_tanh"]["rel_l2_error"],
        "softplus": all_results["dgm_softplus"]["rel_l2_error"],
    }
    plot_dgm_ablation_activation(act_results, "results/ablations")

    print("\nAblation Summary:")
    for name, r in all_results.items():
        print(f"  {name}: error={r['rel_l2_error']:.4%}, time={r['train_time']:.1f}s")


if __name__ == "__main__":
    main()
