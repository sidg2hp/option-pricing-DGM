"""Abstract base class for collocation-point samplers."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseSampler(ABC):
    """Interface for sampling collocation points in the PDE domain.

    All samplers must produce three kinds of points:

    * **Interior** points ``(t, x)`` in ``[0, T) x R^d``
    * **Terminal** points ``x`` at ``t = T``
    * **Boundary** points ``(t, x)`` on the truncated domain boundary

    Parameters
    ----------
    device : torch.device
        Target device for returned tensors.
    """

    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def sample_interior(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample interior collocation points.

        Returns
        -------
        t : torch.Tensor, shape ``(n, 1)``
        x : torch.Tensor, shape ``(n, d)``
        """

    @abstractmethod
    def sample_terminal(self, n: int) -> torch.Tensor:
        """Sample points at the terminal time.

        Returns
        -------
        x : torch.Tensor, shape ``(n, d)``
        """

    @abstractmethod
    def sample_boundary(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample boundary collocation points.

        Returns
        -------
        t : torch.Tensor, shape ``(n, 1)``
        x : torch.Tensor, shape ``(n, d)``
        """
