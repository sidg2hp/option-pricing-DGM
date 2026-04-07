"""Factory function for building neural network models from configuration.

Centralises model construction so that experiment scripts only need to
call ``build_model(config)`` without knowing the architecture details.
"""

from typing import Callable

import torch.nn as nn

from configs.base_config import ExperimentConfig, MarketConfig, ModelConfig
from models.dgm_network import DGMNetwork
from models.mlp_network import MLPNetwork


from pde.payoffs import get_payoff_fn

def build_model(
    config: ExperimentConfig,
    payoff_fn: Callable = None,
) -> nn.Module:
    """Construct a neural network model from the experiment configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    payoff_fn : callable, optional
        Provided for backwards compatibility, but we now generate both
        the clipped and unclipped functions from the config directly.

    Returns
    -------
    nn.Module
        A ``DGMNetwork`` or ``MLPNetwork`` instance.
    """
    p_fn, p_uncl = get_payoff_fn(config.market.payoff_type, config.market.K)

    arch = config.model.architecture.lower()
    if arch == "dgm":
        return DGMNetwork(config.model, config.market, p_fn, p_uncl)
    if arch == "mlp":
        return MLPNetwork(config.model, config.market, p_fn, p_uncl)
    raise ValueError(
        f"Unknown architecture '{arch}'. Choose 'dgm' or 'mlp'."
    )
