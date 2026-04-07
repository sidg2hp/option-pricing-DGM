"""End-to-end test: DGM on 1D Black-Scholes, error < 1%.

Uses a short training run to verify the full pipeline works.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig, SamplerConfig, TrainingConfig
from evaluation.metrics import relative_l2_error
from models.model_factory import build_model
from pde.analytical import black_scholes_call_1d
from pde.payoffs import get_payoff_fn
from training.trainer import DGMTrainer
from utils.random import seed_everything


def test_dgm_1d_e2e():
    """Full pipeline test: DGM 1D BS with a modest training budget."""
    config = ExperimentConfig(
        name="test_1d",
        market=MarketConfig(d=1, sigma=[0.2], rho=[[1.0]], S0=[1.0]),
        model=ModelConfig(
            architecture="dgm", hidden_size=64, num_dgm_layers=2,
        ),
        sampler=SamplerConfig(n_interior=512, n_terminal=128, n_boundary=64),
        training=TrainingConfig(
            n_steps=3_000, lbfgs_finetune=True, lbfgs_steps=100,
            log_every=500, eval_every=1500,
        ),
        output_dir="results/test_dgm_1d",
    )
    seed_everything(42)

    payoff_fn, payoff_unclipped_fn = get_payoff_fn("basket_call", 1.0)
    device = torch.device("cpu")

    S_test = np.linspace(0.6, 1.4, 30)
    anal = black_scholes_call_1d(S_test, 1.0, 0.05, 0.2, 1.0, np.zeros_like(S_test))
    mc_ref = {"S0_test": S_test.reshape(-1, 1), "mc_prices": anal}

    model = build_model(config, payoff_fn)
    trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, mc_ref)
    results = trainer.train()

    x_test = np.log(S_test).reshape(-1, 1)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(len(x_test), 1, dtype=torch.float32, device=device)
    with torch.no_grad():
        dgm_prices = model(t_t, x_t).cpu().numpy().ravel()

    rel_l2 = relative_l2_error(dgm_prices, anal)
    print(f"[E2E Test] 1D DGM rel L2 error: {rel_l2:.4%}")

    assert rel_l2 < 0.20, f"E2E test: error {rel_l2:.4%} too large (threshold 20% for short run)"
    print("[E2E Test] PASSED")


if __name__ == "__main__":
    test_dgm_1d_e2e()
