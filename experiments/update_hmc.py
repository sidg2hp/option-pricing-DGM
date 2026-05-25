import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import numpy as np
import torch
from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig
from evaluation.monte_carlo import HybridMCControlVariate
from models.model_factory import build_model
from utils.math_utils import build_equicorrelation_matrix
from pde.payoffs import get_payoff_fn

summary_path = "results/paper/comparison_summary.json"
with open(summary_path, "r") as f:
    summary = json.load(f)

device = torch.device("cpu")
K = 1.0
r = 0.05
T = 1.0

def mc_payoff(S_T):
    return np.maximum(S_T.mean(axis=1) - K, 0.0)

for d_str in ["1", "2", "3", "5", "7", "10"]:
    d = int(d_str)
    print(f"Recomputing Hybrid MC for d={d}...")
    sigma_list = [0.2] * d
    rho_mat = build_equicorrelation_matrix(d, 0.3).tolist()
    
    config = ExperimentConfig(
        name=f"comparison_dgm_d{d}",
        market=MarketConfig(d=d, r=r, T=T, K=K, sigma=sigma_list, rho=rho_mat, S0=[1.0]*d, payoff_type="basket_call"),
        model=ModelConfig(architecture="dgm", hidden_size=512, num_dgm_layers=4, use_hard_terminal_constraint=True),
        output_dir=f"results/scaling/d_{d}"
    )
    
    payoff_fn, _ = get_payoff_fn("basket_call", K)
    model = build_model(config, payoff_fn)
    ckpt_path = f"results/scaling/d_{d}/scaling_d{d}/checkpoints/best_model.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "initial_layer.0.weight" in ckpt["state_dict"]:
            actual_hidden = ckpt["state_dict"]["initial_layer.0.weight"].shape[0]
            config.model.hidden_size = actual_hidden
            model = build_model(config, payoff_fn)
        model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pricer = HybridMCControlVariate(model, device)
    
    S0_test_values = [0.8, 0.9, 1.0, 1.1, 1.2]
    cv_prices = []
    cv_errors = []
    
    for i, s0_val in enumerate(S0_test_values):
        s0 = np.array([s0_val] * d)
        res = pricer.price_with_delta_cv(
            S0=s0, payoff_fn=mc_payoff, r=r, sigma=np.array(sigma_list), rho=np.array(rho_mat), T=T, K=K,
            n_paths=50000, n_steps=30, seed=i
        )
        cv_prices.append(res["cv_price"])
        cv_errors.append(res["cv_se"])
        
    summary[d_str]["cv_prices"] = cv_prices
    summary[d_str]["cv_errors"] = cv_errors
    
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("Updated comparison_summary.json with correct Hybrid MC prices.")
