# Deep Galerkin Method for Multi-Asset European Option Pricing

A publication-quality research codebase implementing the **Deep Galerkin Method (DGM)** for solving the multi-asset Black-Scholes PDE, with a replication of Zhou et al. (2021).

## Overview

This project:

1. **Solves the multi-asset Black-Scholes PDE directly** using a physics-informed neural network (DGM), operating in log-price coordinates.
2. **Replicates** the key results of *"Deep Learning Artificial Neural Network for Pricing Multi-Asset European Options"* by Zhou et al. (2021), which uses Monte Carlo + Neural Network regression.
3. **Extends** the comparison with scaling studies, ablations, Greeks computation, and residual diagnostics unique to the PDE-solving approach.

### Supported option types

| Payoff | Formula (log-price coords) |
|---|---|
| Arithmetic basket call | `K * max(mean(exp(x_i)) - 1, 0)` |
| Geometric basket call | `K * max(exp(mean(x_i)) - 1, 0)` |
| Best-of (max) call | `K * max(max(exp(x_i)) - 1, 0)` |
| Spread option (d=2) | `max(K*exp(x_1) - K*exp(x_2) - K, 0)` |

---

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA GPU recommended (CPU works but is significantly slower)

## Installation

```bash
# Clone or navigate to the project
cd dgm_option_pricing

# Install dependencies
pip install -r requirements.txt

# Or install as a package (editable mode)
pip install -e ".[dev]"
```

### Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Run all unit tests (no training, completes in ~30 seconds)

```bash
cd dgm_option_pricing

python tests/test_pde_operator.py
python tests/test_payoffs.py
python tests/test_analytical.py
python tests/test_monte_carlo.py
```

### Run the 1D validation experiment

```bash
python experiments/run_validation.py --test 1d --n_steps 50000
```

### Run the 2D geometric basket validation

```bash
python experiments/run_validation.py --test 2d_geometric --n_steps 50000
```

### Run both validations

```bash
python experiments/run_validation.py --test all --n_steps 50000
```

---

## Experiments

All experiment scripts are in the `experiments/` directory. Each accepts `--n_steps` to control the training budget. Reduce this value for faster (but less accurate) runs.

### 1. Validation (Step 22 in the pipeline)

Validates DGM against exact analytical solutions.

```bash
# Using default parameters
python experiments/run_validation.py

# Using a YAML config
python experiments/run_validation.py --config configs/experiments/validate_1d.yaml

# Quick test with fewer steps
python experiments/run_validation.py --test 1d --n_steps 10000
```

**Pass criteria:** 1D relative L2 error < 1%, 2D geometric basket < 2%.

**Output:** `results/validation/`

### 2. Scaling Study (d = 1 to 10 assets)

Measures how DGM error and training time scale with dimension.

```bash
# Full scaling study
python experiments/run_scaling.py --dims 1 2 3 5 7 10

# Quick subset
python experiments/run_scaling.py --dims 1 2 3 --n_steps 20000
```

**Output:** `results/scaling/` with per-dimension subdirectories and scaling plots.

### 3. Zhou et al. (2021) Replication

Trains both DGM and the Zhou et al. MC+NN baseline, produces a side-by-side comparison.

```bash
# 2-asset replication (default)
python experiments/run_zhou_replication.py --d 2

# 5-asset replication
python experiments/run_zhou_replication.py --d 5 --n_steps 100000
```

**Output:** `results/zhou_replication/` with price comparison tables (CSV + PDF) and figures.

### 4. Ablation Studies

Compares DGM vs MLP, tanh vs softplus, uniform vs adaptive sampling.

```bash
python experiments/run_ablations.py --n_steps 50000
```

**Output:** `results/ablations/` with bar charts and summary JSON.

### 5. Greeks Accuracy

Compares DGM-computed Delta and Gamma against analytical values (1D).

```bash
python experiments/run_greeks.py --n_steps 50000
```

**Output:** `results/greeks/` with delta and gamma plots.

### 6. Hybrid DGM + Monte Carlo (Control Variate)

Uses the trained DGM solution as a control variate for variance reduction in MC pricing.

```bash
python experiments/run_hybrid_mc.py --n_steps 50000 --n_mc 100000
```

**Output:** `results/hybrid_mc/` with variance reduction statistics.

---

## Configuration

All parameters are controlled via Python dataclasses in `configs/base_config.py`. Pre-built YAML configs are in `configs/experiments/`.

### Using YAML configs

```bash
python experiments/run_validation.py --config configs/experiments/validate_1d.yaml
```

### Key parameters

| Parameter | Location | Default | Description |
|---|---|---|---|
| `market.d` | MarketConfig | 2 | Number of assets |
| `market.r` | MarketConfig | 0.05 | Risk-free rate |
| `market.T` | MarketConfig | 1.0 | Maturity (years) |
| `market.K` | MarketConfig | 1.0 | Strike price |
| `market.sigma` | MarketConfig | [0.2, 0.2] | Volatilities |
| `market.rho` | MarketConfig | [[1,0.3],[0.3,1]] | Correlation matrix |
| `market.payoff_type` | MarketConfig | basket_call | Payoff type |
| `model.architecture` | ModelConfig | dgm | `dgm` or `mlp` |
| `model.hidden_size` | ModelConfig | 256 | Hidden layer width |
| `model.num_dgm_layers` | ModelConfig | 4 | Number of DGM layers |
| `model.activation` | ModelConfig | tanh | `tanh` or `softplus` |
| `model.use_hard_terminal_constraint` | ModelConfig | true | Hard constraint ansatz |
| `training.n_steps` | TrainingConfig | 100000 | Adam gradient steps |
| `training.lr_init` | TrainingConfig | 1e-3 | Initial learning rate |
| `training.lbfgs_finetune` | TrainingConfig | true | L-BFGS fine-tuning |
| `training.seed` | TrainingConfig | 42 | Global random seed |
| `sampler.sampler_type` | SamplerConfig | risk_neutral | `uniform`, `risk_neutral`, `adaptive` |
| `sampler.n_interior` | SamplerConfig | 4096 | Interior batch size |

---

## Project Structure

```
dgm_option_pricing/
├── configs/                 # Configuration dataclasses and YAML files
│   ├── base_config.py       # Master config with all parameters
│   └── experiments/         # Pre-built experiment configs
├── pde/                     # PDE mathematics
│   ├── operator.py          # Black-Scholes operator L[u] via autograd
│   ├── payoffs.py           # Payoff functions in log-price coordinates
│   ├── boundary.py          # Domain truncation and boundary conditions
│   └── analytical.py        # Closed-form solutions (1D BS, geometric basket)
├── models/                  # Neural network architectures
│   ├── dgm_layer.py         # DGM gating layer (Z, G, R, H mechanism)
│   ├── dgm_network.py       # Full DGM network with hard constraint
│   ├── mlp_network.py       # MLP baseline with residual blocks
│   └── model_factory.py     # build_model(config) factory
├── samplers/                # Collocation point sampling
│   ├── uniform_sampler.py   # Uniform in truncated domain
│   ├── risk_neutral_sampler.py  # Gaussian matching risk-neutral measure
│   └── adaptive_sampler.py  # Residual-based importance resampling
├── losses/                  # Loss functions
│   ├── pde_loss.py          # Mean squared PDE residual
│   ├── terminal_loss.py     # Terminal condition loss (soft constraint)
│   ├── boundary_loss.py     # Boundary condition loss
│   └── combined_loss.py     # Weighted combination
├── training/                # Training loop
│   ├── trainer.py           # DGMTrainer (Adam + L-BFGS)
│   ├── callbacks.py         # Early stopping, checkpointing
│   └── scheduler.py         # Cosine annealing with warm restarts
├── evaluation/              # Evaluation tools
│   ├── monte_carlo.py       # MC pricer with antithetic variates
│   ├── metrics.py           # L2 error, relative error
│   └── diagnostics.py       # Residual maps, NaN checks
├── baselines/               # Baseline methods
│   └── zhou_et_al.py        # Zhou et al. (2021) MC+NN replication
├── experiments/             # Runnable experiment scripts
│   ├── run_validation.py    # 1D and geometric basket validation
│   ├── run_scaling.py       # d = 1..10 scaling study
│   ├── run_zhou_replication.py  # Zhou et al. comparison
│   ├── run_ablations.py     # Architecture/activation/sampling ablations
│   ├── run_greeks.py        # Delta and Gamma accuracy
│   └── run_hybrid_mc.py     # DGM as MC control variate
├── plotting/                # Publication-quality figures
│   ├── style.py             # Global matplotlib style
│   ├── zhou_figures.py      # All Zhou et al. + DGM extension figures
│   ├── price_surface.py     # 3D price surface plots
│   ├── convergence.py       # Training loss curves
│   ├── error_maps.py        # Pointwise error heatmaps
│   ├── scaling_plots.py     # Error/time vs dimension
│   └── greeks_plots.py      # Delta and Gamma plots
├── utils/                   # Utilities
│   ├── random.py            # seed_everything()
│   ├── math_utils.py        # Correlation matrix validation, Cholesky
│   ├── logging.py           # Structured logging + optional W&B
│   └── io_utils.py          # Config/result I/O
├── tests/                   # Test suite
│   ├── test_pde_operator.py # L[V_analytical] ≈ 0 verification
│   ├── test_payoffs.py      # Payoff values at known points
│   ├── test_analytical.py   # Analytical formulas vs known tables
│   ├── test_monte_carlo.py  # MC convergence to analytical
│   └── test_dgm_1d.py       # End-to-end pipeline test
└── results/                 # Auto-created experiment outputs
```

---

## Running Tests

```bash
# Run all tests with pytest
cd dgm_option_pricing
pytest tests/ -v

# Run individual test files
python tests/test_pde_operator.py    # ~8 sec, verifies PDE operator correctness
python tests/test_payoffs.py         # ~1 sec, payoff function checks
python tests/test_analytical.py      # ~1 sec, analytical formula checks
python tests/test_monte_carlo.py     # ~7 sec, MC vs analytical convergence
python tests/test_dgm_1d.py          # ~13 min on CPU, end-to-end pipeline
```

---

## Recommended Execution Order

For a full reproduction, run the following in sequence:

```bash
# 1. Verify correctness (fast, no GPU needed)
python tests/test_pde_operator.py
python tests/test_payoffs.py
python tests/test_analytical.py
python tests/test_monte_carlo.py

# 2. Validate DGM against analytical solutions
python experiments/run_validation.py --test all --n_steps 50000

# 3. Scaling study
python experiments/run_scaling.py --dims 1 2 3 5 --n_steps 100000

# 4. Zhou et al. replication
python experiments/run_zhou_replication.py --d 2 --n_steps 100000

# 5. Ablation studies
python experiments/run_ablations.py --n_steps 50000

# 6. Greeks comparison
python experiments/run_greeks.py --n_steps 50000
```

### Estimated runtimes

| Experiment | GPU (A100) | CPU |
|---|---|---|
| Unit tests (no training) | ~30 sec | ~30 sec |
| 1D validation (50k steps) | ~5 min | ~2 hr |
| 2D validation (50k steps) | ~8 min | ~3 hr |
| Scaling d=1..5 (100k steps each) | ~1 hr | ~15 hr |
| Zhou replication d=2 | ~15 min | ~4 hr |
| All ablations (4 runs, 50k each) | ~30 min | ~8 hr |

---

## Outputs

All experiments save results to the `results/` directory:

- **JSON files**: Metrics, configs, and structured results
- **PDF figures**: Publication-ready (300 DPI, serif fonts)
- **PNG figures**: Preview versions
- **CSV files**: Price comparison tables
- **Model checkpoints**: `best_model.pt` and `latest_model.pt`
- **Metrics logs**: `metrics.jsonl` (JSON-lines format)

---

## Key Design Decisions

- **Log-price coordinates**: The PDE is solved in `x = log(S/K)` space, avoiding numerical issues with raw prices.
- **Hard terminal constraint**: The ansatz `u(t,x) = Phi(x)*exp(-r*tau) + tau*u_hat(t,x)` guarantees exact payoff at maturity.
- **No ReLU**: Only `tanh` and `softplus` activations are used. ReLU has zero second derivative almost everywhere, which silently zeros the PDE residual.
- **Exact GBM simulation**: Monte Carlo uses the exact log-normal solution (no Euler discretization error).
- **Hutchinson estimator**: For d > 5, the Hessian trace is estimated stochastically to avoid O(d) backward passes.

---

## Citation

If you use this code, please cite:

```bibtex
@article{zhou2021deep,
  title={Deep Learning Artificial Neural Network for Pricing Multi-Asset
         European Options with Monte Carlo Samples},
  author={Zhou, Zhiqiang and others},
  year={2021}
}

@article{sirignano2018dgm,
  title={DGM: A deep learning algorithm for solving partial differential equations},
  author={Sirignano, Justin and Spiliopoulos, Konstantinos},
  journal={Journal of Computational Physics},
  year={2018}
}
```
