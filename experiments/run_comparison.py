"""Three-way comparison: DGM vs Zhou et al. vs MC across multiple dimensions.

Loads pre-trained DGM models from the scaling study, trains Zhou NN baselines,
and computes MC reference prices for a side-by-side comparison table.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import numpy as np
import torch

from baselines.zhou_et_al import ZhouEtAlPricer
from baselines.fbsde_pricer import DeepBSDEPricer
from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig, TrainingConfig
from evaluation.monte_carlo import MonteCarloPricer, HybridMCControlVariate
from models.model_factory import build_model
from pde.payoffs import get_payoff_fn
from training.trainer import DGMTrainer
from utils.io_utils import save_json
from utils.math_utils import build_equicorrelation_matrix
from utils.random import seed_everything


def run_comparison_for_d(
    d: int,
    model_base: str = "results/scaling",
    n_dgm_steps: int = 50_000,
    run_fbsde: bool = False,
) -> dict:
    """Run three-way comparison for a single dimension.

    Parameters
    ----------
    d : int
        Number of assets.
    model_base : str
        Base directory for pre-trained DGM models.
    n_dgm_steps : int
        Training steps if no pre-trained model exists.

    Returns
    -------
    dict
        Comparison results.
    """
    K = 1.0
    r = 0.05
    T = 1.0
    sigma_list = [0.2] * d
    rho_mat = build_equicorrelation_matrix(d, 0.3).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payoff_fn, payoff_unclipped_fn = get_payoff_fn("basket_call", K)

    out_dir = f"results/publication/comparison/d_{d}"
    os.makedirs(out_dir, exist_ok=True)

    # --- Test points ---
    S0_test_values = [0.8, 0.9, 1.0, 1.1, 1.2]  # normalised to K=1
    S0_test = np.array([[s] * d for s in S0_test_values])

    # --- 1. Monte Carlo reference ---
    print(f"  Computing MC reference ({1_000_000} paths)...")
    mc_pricer = MonteCarloPricer()

    def mc_payoff(S_T):
        return np.maximum(S_T.mean(axis=1) - K, 0.0)

    mc_prices = []
    mc_errors = []
    for i, s0 in enumerate(S0_test):
        res = mc_pricer.price(
            s0, mc_payoff, r, np.array(sigma_list), np.array(rho_mat), T,
            n_paths=1_000_000, seed=i,
        )
        mc_prices.append(res["price"])
        mc_errors.append(res["std_error"])
    mc_prices = np.array(mc_prices)
    mc_errors = np.array(mc_errors)

    # --- 2. DGM ---
    print(f"  Loading/training DGM model...")
    # Build default config
    config = ExperimentConfig(
        name=f"comparison_dgm_d{d}",
        market=MarketConfig(
            d=d, r=r, T=T, K=K,
            sigma=sigma_list, rho=rho_mat,
            S0=[1.0]*d,
            payoff_type="basket_call",
        ),
        model=ModelConfig(
            architecture="dgm",
            hidden_size=512,
            num_dgm_layers=4,
            use_hard_terminal_constraint=True,
        ),
        training=TrainingConfig(n_steps=n_dgm_steps),
        output_dir=out_dir,
    )

    model = None
    loaded = False

    scaling_dir = f"results/scaling/d_{d}/scaling_d{d}"
    
    # Try to load the original config to ensure model architecture matches exactly
    config_path = os.path.join(scaling_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            saved_cfg = json.load(f)
        if "model" in saved_cfg:
            if "hidden_size" in saved_cfg["model"]:
                config.model.hidden_size = saved_cfg["model"]["hidden_size"]
            if "architecture" in saved_cfg["model"]:
                config.model.architecture = saved_cfg["model"]["architecture"]

    model = build_model(config, payoff_fn)
    model.to(device)

    # Try to load pre-trained DGM weights
    ckpt_names = ["best_model.pt", "latest_model.pt"]
    for ckpt_name in ckpt_names:
        ckpt_path = os.path.join(scaling_dir, "checkpoints", ckpt_name)
        if os.path.exists(ckpt_path):
            print(f"  Found pre-trained model: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            
            # Auto-detect hidden_size from weights to prevent shape mismatch
            if "initial_layer.0.weight" in sd:
                actual_hidden = sd["initial_layer.0.weight"].shape[0]
                if actual_hidden != config.model.hidden_size:
                    print(f"  Auto-adjusting hidden_size from {config.model.hidden_size} to {actual_hidden} to match checkpoint.")
                    config.model.hidden_size = actual_hidden
                    model = build_model(config, payoff_fn)
                    model.to(device)
            
            model.load_state_dict(sd)
            loaded = True
            break

    if not loaded:
        print(f"  No pre-trained model found. Training from scratch ({n_dgm_steps} steps)...")
        mc_ref = {"S0_test": S0_test, "mc_prices": mc_prices}
        trainer = DGMTrainer(model, config, payoff_fn, payoff_unclipped_fn, mc_ref)
        trainer.train()

    model.eval()

    x_test = np.log(S0_test / K)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    t_t = torch.zeros(len(S0_test), 1, dtype=torch.float32, device=device)
    with torch.no_grad():
        dgm_prices = model(t_t, x_t).cpu().numpy().ravel()

    # --- 3. Zhou et al. NN ---
    print(f"  Training Zhou NN (100k samples, 200 epochs)...")

    def zhou_payoff(S_T):
        return np.maximum(S_T.mean(axis=1) - K, 0.0)

    zhou = ZhouEtAlPricer(config.market, zhou_payoff, seed=42)
    zhou.train(n_samples=100_000, n_epochs=200)
    zhou_prices = zhou.predict(S0_test)
    
    # --- 4. Hybrid MC (Control Variates) ---
    print(f"  Computing Hybrid MC reference (50k paths, 30 time steps)...")
    cv_pricer = HybridMCControlVariate(model, device)
    cv_prices = []
    cv_errors = []
    for i, s0 in enumerate(S0_test):
        res = cv_pricer.price_with_delta_cv(
            S0=s0, payoff_fn=mc_payoff, r=r, sigma=np.array(sigma_list), rho=np.array(rho_mat), T=T, K=K,
            n_paths=50_000, n_steps=30, seed=i
        )
        cv_prices.append(res["cv_price"])
        cv_errors.append(res["cv_se"])
    cv_prices = np.array(cv_prices)
    cv_errors = np.array(cv_errors)

    # --- Compute errors ---
    dgm_rel_errors = np.abs(dgm_prices - mc_prices) / mc_prices
    zhou_rel_errors = np.abs(zhou_prices - mc_prices) / mc_prices

    # --- 5. Deep BSDE Baseline ---
    fbsde_prices = np.full(len(S0_test_values), np.nan)
    fbsde_atm_rel_error = None
    if run_fbsde:
        print(f"  Training Deep BSDE (ATM only, 20k steps)...")
        fbsde = DeepBSDEPricer(config.market, n_steps=20_000, device=device)
        fbsde.train()
        fbsde_prices = fbsde.price(S0_test)
        
        atm_idx = S0_test_values.index(1.0)
        fbsde_atm_price = fbsde_prices[atm_idx]
        mc_atm_price = mc_prices[atm_idx]
        fbsde_atm_rel_error = float(abs(fbsde_atm_price - mc_atm_price) / max(mc_atm_price, 1e-6))
        print(f"  FBSDE ATM Price: {fbsde_atm_price:.4f} (Rel Error: {fbsde_atm_rel_error:.2%})")

    result = {
        "d": d,
        "S0_test": S0_test_values,
        "dgm_prices": dgm_prices.tolist(),
        "zhou_prices": zhou_prices.tolist(),
        "mc_prices": mc_prices.tolist(),
        "mc_errors": mc_errors.tolist(),
        "cv_prices": cv_prices.tolist(),
        "cv_errors": cv_errors.tolist(),
        "dgm_mean_rel_error": float(np.mean(dgm_rel_errors)),
        "zhou_mean_rel_error": float(np.mean(zhou_rel_errors)),
        "fbsde_prices": fbsde_prices.tolist(),
        "fbsde_atm_rel_error": fbsde_atm_rel_error,
    }

    save_json(result, f"{out_dir}/comparison_results.json")

    # Print comparison
    print(f"\n  {'S0':>6s} {'DGM':>10s} {'Zhou':>10s} {'MC':>10s} {'MC SE':>10s} {'Hybrid MC':>10s} {'HMC SE':>10s}")
    for i in range(len(S0_test_values)):
        print(f"  {S0_test_values[i]:6.2f} {dgm_prices[i]:10.4f} "
              f"{zhou_prices[i]:10.4f} {mc_prices[i]:10.4f} {mc_errors[i]:10.6f} {cv_prices[i]:10.4f} {cv_errors[i]:10.6f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run 5-way baseline comparison")
    parser.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3, 5, 7, 10, 25, 50, 100])
    parser.add_argument("--n_dgm_steps", type=int, default=50_000)
    parser.add_argument("--model_base", type=str, default="results/scaling", help="Directory where pre-trained DGM models are located")
    parser.add_argument("--run_fbsde", action="store_true", help="Run Deep BSDE baseline (ATM only)")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if results exist")
    args = parser.parse_args()

    seed_everything(42)
    all_results = {}

    for d in args.dims:
        print(f"\n{'='*60}")
        print(f"5-Way Comparison: d = {d}")
        print(f"{'='*60}")

        result_path = f"results/publication/comparison/d_{d}/comparison_results.json"
        
        # If the file already exists, we might only need to run FBSDE
        if os.path.exists(result_path) and not args.force:
            with open(result_path) as f:
                d_result = json.load(f)
            
            # If the user requested FBSDE and it's missing from the loaded results, just run FBSDE!
            if args.run_fbsde and "fbsde_atm_rel_error" not in d_result:
                print(f"  Existing results found for d={d}, but FBSDE is missing. Running FBSDE only...")
                from configs.base_config import ExperimentConfig, MarketConfig
                
                sigma_list = [0.2] * d
                from utils.math_utils import build_equicorrelation_matrix
                rho_mat = build_equicorrelation_matrix(d, 0.3).tolist()
                S0_list = [1.0] * d
                market_cfg = MarketConfig(d=d, r=0.05, T=1.0, K=1.0, sigma=sigma_list, rho=rho_mat, S0=S0_list, payoff_type="basket_call")
                
                fbsde = DeepBSDEPricer(market_cfg, n_steps=20_000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                fbsde.train()
                
                # Re-evaluate exactly like in the main function
                S0_test_values = [0.8, 0.9, 1.0, 1.1, 1.2]
                fbsde_prices = fbsde.price(np.array([S0_val * np.ones(d) for S0_val in S0_test_values]))
                
                atm_idx = S0_test_values.index(1.0)
                fbsde_atm_price = fbsde_prices[atm_idx]
                mc_atm_price = d_result["mc_prices"][atm_idx]
                fbsde_atm_rel_error = float(abs(fbsde_atm_price - mc_atm_price) / max(mc_atm_price, 1e-6))
                
                d_result["fbsde_prices"] = fbsde_prices.tolist()
                d_result["fbsde_atm_rel_error"] = fbsde_atm_rel_error
                print(f"  FBSDE ATM Price: {fbsde_atm_price:.4f} (Rel Error: {fbsde_atm_rel_error:.2%})")
                
                # Re-save the patched result
                save_json(d_result, result_path)
            else:
                print(f"  Skipping d={d} (found existing complete results)")
                
            all_results[str(d)] = d_result
        else:
            all_results[str(d)] = run_comparison_for_d(d, args.model_base, args.n_dgm_steps, args.run_fbsde)

    os.makedirs("results/publication", exist_ok=True)
    save_json(all_results, "results/publication/comparison_summary.json")

    # Print summary table
    print("\n" + "=" * 95)
    print("Summary: Mean relative error vs MC & Hybrid MC SE")
    print("=" * 95)
    print(f"{'d':>4s} {'DGM (Err)':>12s} {'Zhou (Err)':>12s} {'MC SE':>12s} {'HMC SE':>12s} {'FBSDE ATM Err':>15s}")
    for d_str in sorted(all_results.keys(), key=int):
        r = all_results[d_str]
        mean_mc_se = np.mean(r['mc_errors'])
        mean_cv_se = np.mean(r['cv_errors'])
        fbsde_err = r.get("fbsde_atm_rel_error")
        fbsde_str = f"{fbsde_err:15.4%}" if fbsde_err is not None else f"{'N/A':>15s}"
        print(f"{r['d']:4d} {r['dgm_mean_rel_error']:12.4%} "
              f"{r['zhou_mean_rel_error']:12.4%} {mean_mc_se:12.6f} {mean_cv_se:12.6f} {fbsde_str}")


if __name__ == "__main__":
    main()
