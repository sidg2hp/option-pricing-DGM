"""Risk-neutral sampling: draws from a Gaussian matching the risk-neutral measure.

Interior samples are drawn from a Gaussian centred at the expected
log-price at mid-maturity, with covariance matching the diffusion at
maturity.  Terminal samples enrich near the at-the-money surface.
"""

from typing import Tuple

import numpy as np
import torch

from pde.boundary import compute_domain_bounds
from samplers.base_sampler import BaseSampler
from utils.math_utils import validate_correlation_matrix


class RiskNeutralSampler(BaseSampler):
    r"""Sample from a distribution matching the risk-neutral measure.

    Interior samples:
        - :math:`t \sim \mathrm{Uniform}[0, T]`
        - :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)` where

          .. math::

              \mu_{x,i} = (r - \sigma_i^2/2) \cdot T/2

          (expected log-price at mid-maturity) and

          .. math::

              \Sigma_{x,ij} = \rho_{ij}\,\sigma_i\,\sigma_j\,T

          (covariance at maturity).

    Terminal samples:
        - :math:`t = T`
        - :math:`x \sim \mathcal{N}(\mu_x, \Sigma_x)` with 20 %
          enrichment near the at-the-money surface :math:`x \approx 0`.

    Boundary samples:
        - :math:`t \sim \mathrm{Uniform}[0, T]`
        - One coordinate set to
          :math:`x_i^{\min} = -k\,\sigma_i\,\sqrt{T}`.

    Parameters
    ----------
    sigma : np.ndarray
        Volatilities, shape ``(d,)``.
    rho : np.ndarray
        Correlation matrix, shape ``(d, d)``.
    r : float
        Risk-free rate.
    T : float
        Maturity.
    domain_std_multiplier : float
        Domain truncation factor.
    device : torch.device
        Target device.
    """

    def __init__(
        self,
        sigma: np.ndarray,
        rho: np.ndarray,
        r: float,
        T: float,
        domain_std_multiplier: float = 4.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device)
        sigma = np.asarray(sigma, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        self.d = len(sigma)
        self.T = T
        self.r = r

        L = validate_correlation_matrix(rho)
        self.L_torch = torch.tensor(L, dtype=torch.float32, device=device)

        self.mu_x = torch.tensor(
            (r - 0.5 * sigma**2) * T / 2.0,
            dtype=torch.float32,
            device=device,
        )

        cov = rho * np.outer(sigma, sigma) * T
        L_cov = np.linalg.cholesky(cov)
        self.L_cov = torch.tensor(L_cov, dtype=torch.float32, device=device)

        x_min, x_max = compute_domain_bounds(sigma, T, domain_std_multiplier)
        self.x_min = torch.tensor(x_min, dtype=torch.float32, device=device)
        self.x_max = torch.tensor(x_max, dtype=torch.float32, device=device)

        self.sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

    def sample_interior(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(n, 1, device=self.device) * self.T
        z = torch.randn(n, self.d, device=self.device)
        x = self.mu_x + z @ self.L_cov.T
        return t, x

    def sample_terminal(self, n: int) -> torch.Tensor:
        n_atm = int(0.2 * n)
        n_normal = n - n_atm

        z = torch.randn(n_normal, self.d, device=self.device)
        x_normal = self.mu_x + z @ self.L_cov.T

        x_atm = torch.randn(n_atm, self.d, device=self.device) * 0.1

        return torch.cat([x_normal, x_atm], dim=0)

    def sample_boundary(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(n, 1, device=self.device) * self.T
        z = torch.randn(n, self.d, device=self.device)
        x = self.mu_x + z @ self.L_cov.T

        dims = torch.randint(0, self.d, (n,), device=self.device)
        for i in range(n):
            x[i, dims[i]] = self.x_min[dims[i]]
        return t, x
