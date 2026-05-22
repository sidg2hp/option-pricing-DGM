"""Master configuration dataclasses for all experiment parameters.

Uses Python dataclasses with OmegaConf for YAML loading and merging.
Every tunable parameter in the codebase is represented here; no magic
numbers should appear outside of these definitions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class MarketConfig:
    """Parameters describing the multi-asset Black-Scholes market.

    Attributes
    ----------
    d : int
        Number of underlying assets.
    r : float
        Continuously-compounded risk-free interest rate.
    T : float
        Option maturity in years.
    K : float
        Common strike price.
    sigma : list of float
        Per-asset volatilities, length ``d``.
    rho : list of list of float
        Correlation matrix, shape ``(d, d)``.  Must be symmetric positive
        definite with unit diagonal.
    S0 : list of float
        Initial asset prices, length ``d``.
    payoff_type : str
        One of ``basket_call``, ``geometric_basket``, ``max_call``, ``spread``.
    """

    d: int = 2
    r: float = 0.05
    T: float = 1.0
    K: float = 1.0
    sigma: List[float] = field(default_factory=lambda: [0.2, 0.2])
    rho: List[List[float]] = field(
        default_factory=lambda: [[1.0, 0.3], [0.3, 1.0]]
    )
    S0: List[float] = field(default_factory=lambda: [1.0, 1.0])
    payoff_type: str = "basket_call"


@dataclass
class ModelConfig:
    """Neural network architecture parameters.

    Attributes
    ----------
    architecture : str
        ``"dgm"`` for the Deep Galerkin architecture,
        ``"mlp"`` for a standard MLP baseline.
    hidden_size : int
        Width of hidden layers.
    num_dgm_layers : int
        Number of DGM gating layers (ignored for MLP).
    activation : str
        ``"tanh"`` (default, required for DGM) or ``"softplus"`` (ablation).
    use_hard_terminal_constraint : bool
        If True, enforce the terminal condition exactly via the ansatz
        u(t,x) = Phi(x)*exp(-r*tau) + tau*u_hat(t,x).
    payoff_smoothing_eps_init : float
        Initial smoothing parameter for the softplus-like payoff
        approximation.
    payoff_smoothing_anneal_fraction : float
        Fraction of total training steps over which epsilon is annealed
        from ``eps_init`` to 0.
    """

    architecture: str = "dgm"
    hidden_size: int = 512
    num_dgm_layers: int = 4
    activation: str = "tanh"
    use_hard_terminal_constraint: bool = True
    hard_constraint_smoothing_eps: float = 0.01
    payoff_smoothing_eps_init: float = 0.1
    payoff_smoothing_anneal_fraction: float = 0.3


@dataclass
class SamplerConfig:
    """Collocation-point sampling parameters.

    Attributes
    ----------
    sampler_type : str
        ``"uniform"``, ``"risk_neutral"``, or ``"adaptive"``.
    n_interior : int
        Number of interior collocation points per batch.
    n_terminal : int
        Number of terminal-condition sample points per batch.
    n_boundary : int
        Number of boundary-condition sample points per batch.
    domain_std_multiplier : float
        The domain is truncated at +/- this many standard deviations
        from the at-the-money log-price.
    """

    sampler_type: str = "risk_neutral"
    n_interior: int = 1024
    n_terminal: int = 256
    n_boundary: int = 128
    domain_std_multiplier: float = 4.0


@dataclass
class TrainingConfig:
    """Optimiser and training-loop parameters.

    Attributes
    ----------
    n_steps : int
        Total number of Adam gradient steps.
    optimizer : str
        Optimizer name (only ``"adam"`` is currently supported).
    lr_init : float
        Initial learning rate.
    lr_min : float
        Minimum learning rate for the cosine schedule.
    scheduler : str
        LR scheduler type.
    T_0 : int
        Steps per cosine cycle (for warm restarts).
    T_mult : int
        Cycle length multiplier after each restart.
    grad_clip_norm : float
        Max gradient norm for clipping.
    lambda_pde : float
        Weight for the PDE residual loss.
    lambda_terminal : float
        Weight for the terminal-condition loss (soft constraint only).
    lambda_boundary : float
        Weight for the boundary-condition loss.
    use_ntk_balancing : bool
        Experimental: use NTK-based adaptive loss balancing.
    lbfgs_finetune : bool
        Whether to run L-BFGS fine-tuning after Adam training.
    lbfgs_steps : int
        Number of L-BFGS iterations.
    log_every : int
        Steps between metric logging.
    eval_every : int
        Steps between evaluation against reference prices.
    seed : int
        Global random seed.
    """

    n_steps: int = 100_000
    optimizer: str = "adam"
    lr_init: float = 1e-3
    lr_min: float = 1e-5
    scheduler: str = "cosine_warm_restarts"
    T_0: int = 10_000
    T_mult: int = 2
    grad_clip_norm: float = 1.0
    lambda_pde: float = 1.0
    lambda_terminal: float = 10.0
    lambda_boundary: float = 1.0
    use_ntk_balancing: bool = False
    lbfgs_finetune: bool = True
    lbfgs_steps: int = 500
    log_every: int = 500
    eval_every: int = 5000
    seed: int = 42


@dataclass
class MCConfig:
    """Monte Carlo reference-price computation parameters.

    Attributes
    ----------
    n_paths : int
        Number of simulation paths.
    use_antithetic : bool
        Whether to use antithetic-variate variance reduction.
    seed : int
        RNG seed for Monte Carlo simulation.
    """

    n_paths: int = 1_000_000
    use_antithetic: bool = True
    seed: int = 0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration aggregating all sub-configs.

    Attributes
    ----------
    name : str
        Experiment identifier (used in directory naming and logging).
    market : MarketConfig
    model : ModelConfig
    sampler : SamplerConfig
    training : TrainingConfig
    mc : MCConfig
    output_dir : str
        Root directory for experiment outputs.
    device : str
        ``"cuda"`` or ``"cpu"``.  Falls back to CPU automatically.
    """

    name: str = "default"
    market: MarketConfig = field(default_factory=MarketConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    output_dir: str = "results"
    device: str = "cuda"


def load_experiment_config(yaml_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Load an experiment configuration from YAML with optional CLI overrides.

    Parameters
    ----------
    yaml_path : str or None
        Path to a YAML config file.  If None, uses all defaults.
    overrides : dict or None
        Dot-separated key-value pairs to override, e.g.
        ``{"market.d": 5, "training.n_steps": 50000}``.

    Returns
    -------
    ExperimentConfig
        Fully resolved configuration.
    """
    base = OmegaConf.structured(ExperimentConfig)

    if yaml_path is not None:
        file_cfg = OmegaConf.load(yaml_path)
        base = OmegaConf.merge(base, file_cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(
            [f"{k}={v}" for k, v in overrides.items()]
        )
        base = OmegaConf.merge(base, override_cfg)

    schema = OmegaConf.to_object(base)
    return ExperimentConfig(**{
        k: (
            globals()[type(getattr(ExperimentConfig, k, None)).__name__](**v)
            if isinstance(v, dict)
            else v
        )
        for k, v in schema.items()
    }) if False else _dict_to_config(schema)


def _dict_to_config(d: dict) -> ExperimentConfig:
    """Convert a plain dict (from OmegaConf) to a nested ExperimentConfig."""
    return ExperimentConfig(
        name=d.get("name", "default"),
        market=MarketConfig(**d["market"]) if "market" in d else MarketConfig(),
        model=ModelConfig(**d["model"]) if "model" in d else ModelConfig(),
        sampler=SamplerConfig(**d["sampler"]) if "sampler" in d else SamplerConfig(),
        training=TrainingConfig(**d["training"]) if "training" in d else TrainingConfig(),
        mc=MCConfig(**d["mc"]) if "mc" in d else MCConfig(),
        output_dir=d.get("output_dir", "results"),
        device=d.get("device", "cuda"),
    )
