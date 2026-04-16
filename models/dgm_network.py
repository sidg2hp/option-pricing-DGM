"""Full DGM network for solving the multi-asset Black-Scholes PDE.

Architecture:
    1. Initial embedding: S^1 = sigma(W * [t, x] + b)
    2. L DGM layers: S^{l+1} = DGMLayer(input=[t,x], hidden=S^l)
    3. Output: scalar = W_out * S^{L+1} + b_out

Optional hard terminal constraint wrapper:
    u(t, x) = Phi(x)*exp(-r*tau) + tau * u_hat(t, x)
which ensures u(T, x) = Phi(x) exactly.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from configs.base_config import MarketConfig, ModelConfig
from models.dgm_layer import DGMLayer, _get_activation
from pde.payoffs import smoothed_payoff


class DGMNetwork(nn.Module):
    r"""DGM network for multi-asset Black-Scholes PDE.

    The network takes :math:`(t, \mathbf{x}) \in [0,T] \times \mathbb{R}^d`
    and outputs a scalar option price :math:`u(t, \mathbf{x})`.

    When ``use_hard_terminal_constraint`` is enabled (default), the
    output is wrapped as:

    .. math::

        u_\theta(t, \mathbf{x}) = \Phi(\mathbf{x})\,e^{-r\,(T-t)}
            + (T - t)\,\hat{u}_\theta(t, \mathbf{x})

    so that :math:`u_\theta(T, \mathbf{x}) = \Phi(\mathbf{x})` exactly.

    Parameters
    ----------
    config : ModelConfig
        Architecture hyper-parameters.
    market : MarketConfig
        Market parameters (needed for hard constraint and Greeks).
    payoff_fn : callable
        The payoff function :math:`\Phi(\mathbf{x})` in log-price
        coordinates.  Signature: ``(x: Tensor) -> Tensor``.
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

        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            _get_activation(act),
        )

        self.dgm_layers = nn.ModuleList(
            [DGMLayer(input_dim, hidden, act) for _ in range(config.num_dgm_layers)]
        )

        self.output_layer = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.initial_layer[0], self.output_layer]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _raw_output(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute the raw network output u_hat(t, x).

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
        inp = torch.cat([t, x], dim=-1)
        S = self.initial_layer(inp)
        for layer in self.dgm_layers:
            S = layer(inp, S)
        return self.output_layer(S)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute the option price u(t, x).

        Parameters
        ----------
        t : torch.Tensor
            Time, shape ``(N, 1)``.  Should have ``requires_grad=True``
            when evaluating the PDE operator.
        x : torch.Tensor
            Log-price coordinates, shape ``(N, d)``.  Should have
            ``requires_grad=True`` when evaluating the PDE operator.

        Returns
        -------
        torch.Tensor
            Option price, shape ``(N, 1)``.
        """
        u_hat = self._raw_output(t, x)

        if self.use_hard:
            tau = self.T - t
            if self.training and self.payoff_unclipped_fn is not None and self.hard_smoothing_eps > 0:
                phi = smoothed_payoff(self.payoff_unclipped_fn, x, self.hard_smoothing_eps)
            else:
                phi = self.payoff_fn(x)
            return phi * torch.exp(-self.r * tau) + tau * u_hat

        return u_hat

    @torch.no_grad()
    def get_price(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """No-grad wrapper for inference.

        Parameters
        ----------
        t : torch.Tensor
            Shape ``(N, 1)``.
        x : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Option price, shape ``(N, 1)``.
        """
        return self.forward(t, x)

    def get_delta(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the delta in original price coordinates.

        .. math::

            \Delta_i = \frac{\partial V}{\partial S_i}
                     = \frac{1}{S_i}\,\frac{\partial u}{\partial x_i}
                     = \frac{1}{K\,e^{x_i}}\,\frac{\partial u}{\partial x_i}

        Parameters
        ----------
        t : torch.Tensor
            Shape ``(N, 1)``, requires_grad not needed.
        x : torch.Tensor
            Shape ``(N, d)``, will be set to requires_grad internally.

        Returns
        -------
        torch.Tensor
            Delta values, shape ``(N, d)``.
        """
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        u = self.forward(t, x)
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=False
        )[0]
        S = self.K * torch.exp(x)
        return (du_dx / S).detach()

    def get_gamma(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r"""Compute diagonal gamma in original price coordinates.

        .. math::

            \Gamma_i = \frac{\partial^2 V}{\partial S_i^2}
                     = \frac{1}{S_i^2}\left(
                       \frac{\partial^2 u}{\partial x_i^2}
                       - \frac{\partial u}{\partial x_i}\right)

        Parameters
        ----------
        t : torch.Tensor
            Shape ``(N, 1)``.
        x : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Gamma values, shape ``(N, d)``.
        """
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
