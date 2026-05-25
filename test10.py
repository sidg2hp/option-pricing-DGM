
import json
import torch
import numpy as np
import os
from evaluation.monte_carlo import HybridMCControlVariate
from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from utils.math_utils import build_equicorrelation_matrix

device = torch.device('cpu')
K, r, T = 1.0, 0.05, 1.0
def mc_payoff(S_T):
    return np.maximum(S_T.mean(axis=1) - K, 0.0)

for d_str in ['7', '10']:
    d = int(d_str)
    print(f'd={d}')
    sigma_list = [0.2] * d
    rho_mat = build_equicorrelation_matrix(d, 0.3).tolist()
    config = ExperimentConfig(name=f'd{d}', market=MarketConfig(d=d, r=r, T=T, K=K, sigma=sigma_list, rho=rho_mat, S0=[1.0]*d, payoff_type='basket_call'), model=ModelConfig(architecture='dgm', hidden_size=512, num_dgm_layers=4, use_hard_terminal_constraint=True), output_dir=f'results/scaling/d_{d}')
    payoff_fn, _ = get_payoff_fn('basket_call', K)
    model = build_model(config, payoff_fn)
    ckpt = torch.load(f'results/scaling/d_{d}/scaling_d{d}/checkpoints/best_model.pt', map_location=device, weights_only=False)
    if 'initial_layer.0.weight' in ckpt['state_dict']:
        config.model.hidden_size = ckpt['state_dict']['initial_layer.0.weight'].shape[0]
        model = build_model(config, payoff_fn)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    cv_pricer = HybridMCControlVariate(model, device)
    cv_prices, cv_errors = [], []
    for i, s0_val in enumerate([0.8, 0.9, 1.0, 1.1, 1.2]):
        res = cv_pricer.price_with_delta_cv(np.array([s0_val]*d), mc_payoff, r, np.array(sigma_list), np.array(rho_mat), T, K, n_paths=50000, n_steps=30, seed=i)
        cv_prices.append(res['cv_price'])
        cv_errors.append(res['cv_se'])
    
    print(cv_prices)

