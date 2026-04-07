"""Uniform sampling in time and space over the truncated PDE domain."""

from typing import Tuple

import numpy as np
import torch

from pde.boundary import compute_domain_bounds
from samplers.base_sampler import BaseSampler


class UniformSampler(BaseSampler):
    """Sample ``(t, x)`` uniformly in the truncated PDE domain.

    Parameters
    ----------
    sigma : np.ndarray
        Volatilities, shape ``(d,)``.
    T : float
        Maturity.
    domain_std_multiplier : float
        Domain truncation in standard deviations.
    device : torch.device
        Target device for tensors.
    """

    def __init__(
        self,
        sigma: np.ndarray,
        T: float,
        domain_std_multiplier: float = 4.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device)
        self.T = T
        self.d = len(sigma)
        x_min, x_max = compute_domain_bounds(sigma, T, domain_std_multiplier)
        self.x_min = torch.tensor(x_min, dtype=torch.float32, device=device)
        self.x_max = torch.tensor(x_max, dtype=torch.float32, device=device)

    def sample_interior(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(n, 1, device=self.device) * self.T
        x = (
            torch.rand(n, self.d, device=self.device)
            * (self.x_max - self.x_min)
            + self.x_min
        )
        return t, x

    def sample_terminal(self, n: int) -> torch.Tensor:
        return (
            torch.rand(n, self.d, device=self.device)
            * (self.x_max - self.x_min)
            + self.x_min
        )

    def sample_boundary(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(n, 1, device=self.device) * self.T
        x = (
            torch.rand(n, self.d, device=self.device)
            * (self.x_max - self.x_min)
            + self.x_min
        )
        # For each sample, randomly pick one dimension to set to the lower bound
        dims = torch.randint(0, self.d, (n,), device=self.device)
        for i in range(n):
            x[i, dims[i]] = self.x_min[dims[i]]
        return t, x
