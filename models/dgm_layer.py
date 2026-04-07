"""DGM gating layer as described in Sirignano & Spiliopoulos (2018).

Implements the recurrent update mechanism where the original input is
re-injected at every layer, enabling the network to maintain a strong
connection to the input coordinates throughout its depth.
"""

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Parameters
    ----------
    name : str
        ``"tanh"`` or ``"softplus"``.  ReLU is intentionally excluded
        because its zero second derivative destroys PDE residuals.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If an unsupported activation is requested.
    """
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError(
        f"Unsupported activation '{name}'. Use 'tanh' or 'softplus'. "
        "ReLU is prohibited (zero second derivative)."
    )


class DGMLayer(nn.Module):
    r"""Single DGM gating layer implementing the recurrent update.

    The update equations are:

    .. math::

        Z^l &= \sigma(U^z \cdot \text{input} + W^z \cdot S^l + b^z) \\
        G^l &= \sigma(U^g \cdot \text{input} + W^g \cdot S^l + b^g) \\
        R^l &= \sigma(U^r \cdot \text{input} + W^r \cdot S^l + b^r) \\
        H^l &= \sigma(U^h \cdot \text{input}
               + W^h \cdot (S^l \odot R^l) + b^h) \\
        S^{l+1} &= (1 - G^l) \odot H^l + Z^l \odot S^l

    where *input* is the concatenated :math:`[t, x]` vector passed at
    every layer, :math:`S^l` is the hidden state from the previous
    layer, :math:`\sigma` is the activation function, and
    :math:`\odot` denotes element-wise multiplication.

    Parameters
    ----------
    input_dim : int
        Dimension of the input vector (``1 + d`` for time + log-prices).
    hidden_dim : int
        Width of the hidden state.
    activation : str
        Activation function name (``"tanh"`` or ``"softplus"``).
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "tanh"):
        super().__init__()
        self.activation = _get_activation(activation)

        self.Uz = nn.Linear(input_dim, hidden_dim)
        self.Wz = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.Ug = nn.Linear(input_dim, hidden_dim)
        self.Wg = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.Ur = nn.Linear(input_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.Uh = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for weight matrices, zero for biases."""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, input_vec: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Compute one DGM layer update.

        Parameters
        ----------
        input_vec : torch.Tensor
            Original input ``[t, x]``, shape ``(N, 1+d)``.
        S : torch.Tensor
            Hidden state from previous layer, shape ``(N, hidden_dim)``.

        Returns
        -------
        torch.Tensor
            Updated hidden state, shape ``(N, hidden_dim)``.
        """
        Z = self.activation(self.Uz(input_vec) + self.Wz(S))
        G = self.activation(self.Ug(input_vec) + self.Wg(S))
        R = self.activation(self.Ur(input_vec) + self.Wr(S))
        H = self.activation(self.Uh(input_vec) + self.Wh(S * R))
        return (1 - G) * H + Z * S
