import torch
import numpy as np
from evaluation.monte_carlo import HybridMCControlVariate
from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn

d = 3
device = torch.device('cpu')
payoff_fn, _ = get_payoff_fn('basket_call', 1.0)
config = ExperimentConfig(name='d3', market=MarketConfig(d=d, r=0.05, T=1.0, K=1.0, sigma=[0.2]*d, rho=[[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]], S0=[1.0]*d, payoff_type='basket_call'), model=ModelConfig(architecture='dgm', hidden_size=256, num_dgm_layers=4, use_hard_terminal_constraint=True), training=None)
model = build_model(config, payoff_fn)
ckpt = torch.load('results/scaling/d_3/scaling_d3/checkpoints/best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()
cv = HybridMCControlVariate(model, device)
def mc_payoff(S_T):
    return np.maximum(S_T.mean(axis=1) - 1.0, 0.0)
res = cv.price_with_delta_cv(np.array([1.2]*3), mc_payoff, 0.05, np.array([0.2]*3), np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]]), 1.0, 1.0, n_paths=50000, n_steps=30, seed=4)
print(res)
