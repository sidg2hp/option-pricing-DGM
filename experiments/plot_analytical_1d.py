import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import load_experiment_config
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from plotting.style import apply_style
from pde.analytical import black_scholes_call_1d

CB_BLUE    = "#0072B2"
CB_ORANGE  = "#E69F00"
CB_RED     = "#D55E00"

from configs.base_config import ExperimentConfig, MarketConfig, TrainingConfig, ModelConfig

def main():
    apply_style()
    
    # Check if validation model exists
    chkpt_dir = "results/validation/validate_1d/checkpoints"
    if not os.path.exists(chkpt_dir):
        print("Validation 1D model not found. Run: python experiments/run_validation.py --test 1d")
        return
        
    cfg = ExperimentConfig(
        name="validate_1d",
        model=ModelConfig(hidden_size=256),
        market=MarketConfig(d=1, sigma=[0.2], rho=[[1.0]], S0=[1.0]),
        training=TrainingConfig(
            n_steps=50000, eval_every=2500,
            lbfgs_finetune=True, lbfgs_steps=300,
        ),
        output_dir="results/validation",
    )
    m = cfg.market
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    payoff_fn, _ = get_payoff_fn(m.payoff_type, m.K)
    model = build_model(cfg, payoff_fn)
    
    checkpoint = torch.load(os.path.join(chkpt_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    # Generate points
    S_test = np.linspace(0.5 * m.K, 1.5 * m.K, 200)
    x_test = np.log(S_test / m.K).reshape(-1, 1)
    t_test = np.zeros_like(x_test)
    
    # Analytical
    analytical_prices = black_scholes_call_1d(S_test, m.K, m.r, m.sigma[0], m.T, t_test.ravel())
    
    # DGM
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(len(x_test), 1, dtype=torch.float32, device=device)
    with torch.no_grad():
        dgm_prices = model(t_t, x_t).cpu().numpy().ravel()
        
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.plot(S_test, analytical_prices, 'k-', label="Analytical (Black-Scholes)", linewidth=2.5, alpha=0.7)
    ax.plot(S_test, dgm_prices, '--', color=CB_BLUE, label="DGM", linewidth=2.0)
    
    # Plot payoff
    payoff = np.maximum(S_test - m.K, 0)
    ax.plot(S_test, payoff, ':', color='gray', label="Payoff", alpha=0.8)
    
    ax.set_xlabel(r"Spot Price $S_0$")
    ax.set_ylabel(r"Option Price $V(0, S_0)$")
    ax.set_title("1D Black-Scholes: DGM vs Analytical Solution")
    ax.legend(frameon=True)
    
    os.makedirs("figures/publication", exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/publication/fig15_analytical_vs_dgm_1d.png", dpi=300)
    plt.savefig("figures/publication/fig15_analytical_vs_dgm_1d.pdf")
    
    print("Plot saved to figures/publication/fig15_analytical_vs_dgm_1d.png")

if __name__ == "__main__":
    main()
