"""Standard MLP baseline with skip connections for ablation comparison.

Architecture: Linear -> [Block x L] -> Linear
Each block: Linear -> Activation -> Linear -> Activation + skip connection.
Uses the same hard terminal constraint wrapper as DGMNetwork.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from configs.base_config import MarketConfig, ModelConfig
from models.dgm_layer import _get_activation
from pde.payoffs import smoothed_payoff


class _ResidualBlock(nn.Module):
    """Two-layer residual block with activation and skip connection."""

    def __init__(self, dim: int, activation: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            _get_activation(activation),
            nn.Linear(dim, dim),
            _get_activation(activation),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MLPNetwork(nn.Module):
    r"""MLP baseline for ablation comparison with DGM.

    Architecture: ``Linear -> [ResBlock x L] -> Linear``
    where each residual block contains two linear layers with
    activations and a skip connection.

    Input is the concatenated ``[t, x]`` vector of shape ``(N, 1+d)``.
    Output is a scalar ``u(t, x)`` of shape ``(N, 1)``.

    The same hard terminal constraint ansatz is available:

    .. math::

        u_\theta(t, x) = \Phi(x)\,e^{-r\tau} + \tau\,\hat{u}_\theta(t, x)

    Parameters
    ----------
    config : ModelConfig
        Architecture hyper-parameters.
    market : MarketConfig
        Market parameters.
    payoff_fn : callable
        Payoff function in log-price coordinates.
    """

    def __init__(
        self,
        config: ModelConfig,
        market: MarketConfig,
        payoff_fn: Callable[[torch.Tensor], torch.Tensor],
        payoff_unclipped_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.d = market.d
        self.r = market.r
        self.T = market.T
        self.K = market.K
        self.use_hard = config.use_hard_terminal_constraint
        self.hard_smoothing_eps = config.hard_constraint_smoothing_eps
        self.payoff_fn = payoff_fn
        self.payoff_unclipped_fn = payoff_unclipped_fn

        input_dim = 1 + self.d
        hidden = config.hidden_size
        act = config.activation
        n_blocks = config.num_dgm_layers

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            _get_activation(act),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hidden, act) for _ in range(n_blocks)]
        )
        self.output_layer = nn.Linear(hidden, 1)

        nn.init.xavier_uniform_(self.input_layer[0].weight)
        nn.init.zeros_(self.input_layer[0].bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _raw_output(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([t, x], dim=-1)
        h = self.input_layer(inp)
        h = self.blocks(h)
        return self.output_layer(h)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute option price u(t, x).

        Parameters
        ----------
        t : torch.Tensor
            Shape ``(N, 1)``.
        x : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 1)``.
        """
        u_hat = self._raw_output(t, x)
        if self.use_hard:
            tau = self.T - t
            if self.payoff_unclipped_fn is not None and self.hard_smoothing_eps > 0:
                phi = smoothed_payoff(self.payoff_unclipped_fn, x, self.hard_smoothing_eps)
            else:
                phi = self.payoff_fn(x)
            return phi * torch.exp(-self.r * tau) + tau * u_hat
        return u_hat

    @torch.no_grad()
    def get_price(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.forward(t, x)

    def get_delta(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r"""Delta in original price coordinates. See DGMNetwork.get_delta."""
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = self.forward(t, x)
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=False
        )[0]
        S = self.K * torch.exp(x)
        return (du_dx / S).detach()

    def get_gamma(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r"""Diagonal gamma in original price coordinates. See DGMNetwork.get_gamma."""
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = self.forward(t, x)
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        gamma_list = []
        for i in range(self.d):
            d2u_dxi2 = torch.autograd.grad(
                du_dx[:, i].sum(), x, create_graph=False, retain_graph=True
            )[0][:, i:i+1]
            gamma_list.append(d2u_dxi2 - du_dx[:, i:i+1])
        gamma_logprice = torch.cat(gamma_list, dim=-1)
        S = self.K * torch.exp(x)
        return (gamma_logprice / (S**2)).detach()
