"""Replication of Zhou et al. (2021): MC + Neural Network regression.

The method generates training data by running Monte Carlo simulations
at randomised initial conditions, then trains a feedforward neural
network to regress the discounted payoff as a function of the initial
asset prices.  The NN is a regression surrogate, **not** a PDE solver.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from configs.base_config import MarketConfig
from utils.math_utils import validate_correlation_matrix
from utils.random import seed_everything


class ZhouNNModel(nn.Module):
    """Feedforward NN matching the architecture in Zhou et al. (2021).

    4 hidden layers, 100 neurons each, ReLU activation (as in the
    original paper).  Output: scalar price V(0, S0).

    Parameters
    ----------
    d : int
        Number of input assets.
    """

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZhouEtAlPricer:
    r"""Replication of Zhou et al. (2021) MC + NN method.

    Algorithm:
        1. Generate ``n_samples`` random initial conditions
           :math:`S_{0,k}` from a log-uniform distribution.
        2. For each :math:`S_{0,k}`, simulate a single MC path to
           get the discounted payoff :math:`V_k = e^{-rT}\Phi(S_T^k)`.
        3. Train an NN :math:`f_\theta(S_0) \approx V(0, S_0)` via MSE.

    The NN approximates the price at ``t=0`` only.

    Parameters
    ----------
    market : MarketConfig
        Market parameters.
    payoff_fn : callable
        Payoff function on terminal prices, ``(S_T: ndarray) -> ndarray``.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        market: MarketConfig,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        seed: int = 42,
    ):
        self.market = market
        self.payoff_fn = payoff_fn
        self.seed = seed
        self.device = torch.device("cpu")
        self.model = ZhouNNModel(market.d).to(self.device)

    def generate_training_data(
        self, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ``(S0, V)`` training pairs via Monte Carlo.

        Parameters
        ----------
        n_samples : int
            Number of training samples.

        Returns
        -------
        S0_train : np.ndarray, shape ``(n_samples, d)``
        V_train : np.ndarray, shape ``(n_samples,)``
        """
        m = self.market
        rng = np.random.RandomState(self.seed)
        d = m.d
        sigma = np.array(m.sigma)
        rho = np.array(m.rho)
        r = m.r
        T = m.T
        K = m.K

        L = validate_correlation_matrix(rho)

        S0_train = K * np.exp(rng.uniform(-0.5, 0.5, size=(n_samples, d)))

        # Average multiple MC paths per initial condition to reduce label noise
        n_mc = 50
        V_train = np.zeros(n_samples)
        for _ in range(n_mc):
            eps = rng.randn(n_samples, d)
            Z = eps @ L.T
            drift = (r - 0.5 * sigma**2) * T
            diffusion = sigma * np.sqrt(T) * Z
            S_T = S0_train * np.exp(drift + diffusion)
            payoffs = self.payoff_fn(S_T)
            V_train += np.exp(-r * T) * payoffs
        V_train /= n_mc

        return S0_train, V_train

    def train(
        self,
        n_samples: int = 100_000,
        n_epochs: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """Train the Zhou et al. NN.

        Parameters
        ----------
        n_samples : int
            Number of MC training samples.
        n_epochs : int
            Training epochs.
        batch_size : int
            Mini-batch size.
        lr : float
            Learning rate.

        Returns
        -------
        dict
            ``{"loss": [...]}``: per-epoch MSE loss.
        """
        seed_everything(self.seed)
        S0_np, V_np = self.generate_training_data(n_samples)

        S0_t = torch.tensor(S0_np, dtype=torch.float32, device=self.device)
        V_t = torch.tensor(V_np, dtype=torch.float32, device=self.device).unsqueeze(-1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(S0_t, V_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_history = []
        for epoch in tqdm(range(n_epochs), desc="Zhou NN Training"):
            epoch_loss = 0.0
            n_batches = 0
            for S0_batch, V_batch in loader:
                pred = self.model(S0_batch)
                loss = criterion(pred, V_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            loss_history.append(epoch_loss / n_batches)

        return {"loss": loss_history}

    def predict(self, S0: np.ndarray) -> np.ndarray:
        """Predict price at t=0 for given initial prices.

        Parameters
        ----------
        S0 : np.ndarray
            Initial prices, shape ``(M, d)`` or ``(d,)``.

        Returns
        -------
        np.ndarray
            Predicted prices, shape ``(M,)`` or scalar.
        """
        S0 = np.atleast_2d(S0)
        S0_t = torch.tensor(S0, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(S0_t).cpu().numpy().ravel()
        return pred
