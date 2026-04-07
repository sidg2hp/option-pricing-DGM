"""Specialised sampler for boundary points on the truncated domain."""

from typing import Tuple

import numpy as np
import torch

from pde.boundary import compute_domain_bounds
from samplers.base_sampler import BaseSampler


class BoundarySampler(BaseSampler):
    """Sample points on the boundary of the truncated computational domain.

    For each sample, one randomly chosen dimension is set to either
    ``x_min`` (lower boundary, asset worthless) or ``x_max`` (upper
    boundary, deep in-the-money).  The remaining dimensions are drawn
    from a Gaussian centred at zero.

    Parameters
    ----------
    sigma : np.ndarray
        Volatilities, shape ``(d,)``.
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
        T: float,
        domain_std_multiplier: float = 4.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device)
        sigma = np.asarray(sigma, dtype=np.float64)
        self.d = len(sigma)
        self.T = T
        x_min, x_max = compute_domain_bounds(sigma, T, domain_std_multiplier)
        self.x_min = torch.tensor(x_min, dtype=torch.float32, device=device)
        self.x_max = torch.tensor(x_max, dtype=torch.float32, device=device)

    def sample_interior(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("BoundarySampler only samples boundary points.")

    def sample_terminal(self, n: int) -> torch.Tensor:
        raise NotImplementedError("BoundarySampler only samples boundary points.")

    def sample_boundary(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(n, 1, device=self.device) * self.T
        x = torch.randn(n, self.d, device=self.device) * 0.3

        x = torch.clamp(x, self.x_min, self.x_max)

        n_lower = n // 2
        dims_lower = torch.randint(0, self.d, (n_lower,), device=self.device)
        for i in range(n_lower):
            x[i, dims_lower[i]] = self.x_min[dims_lower[i]]

        dims_upper = torch.randint(0, self.d, (n - n_lower,), device=self.device)
        for i in range(n - n_lower):
            x[n_lower + i, dims_upper[i]] = self.x_max[dims_upper[i]]

        return t, x
