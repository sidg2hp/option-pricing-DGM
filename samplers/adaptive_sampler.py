"""Residual-based adaptive importance sampling for DGM.

During training, the PDE residual is evaluated on a large candidate set
and points are resampled proportionally to the squared residual,
concentrating computation in regions where the solution is least
accurate.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from samplers.base_sampler import BaseSampler
from samplers.risk_neutral_sampler import RiskNeutralSampler


class AdaptiveSampler(BaseSampler):
    r"""Residual-based adaptive importance sampling.

    Algorithm
    ---------
    1. Evaluate PDE residual on ``n_candidates = multiplier * n``
       candidate points drawn from the base sampler.
    2. Compute importance weights
       :math:`w_k = |\mathcal{L}[u_\theta](t_k, x_k)|^2`.
    3. Normalise: :math:`p_k = w_k / \sum w_k`.
    4. Resample ``n`` points according to :math:`p_k` (with
       replacement).
    5. Return importance-weighted correction factors for unbiased loss.

    For the first ``warmup_steps`` training steps the base sampler is
    used directly (the model is too random early on for residuals to
    be informative).

    Parameters
    ----------
    base_sampler : RiskNeutralSampler
        Sampler used to generate candidate points.
    n_candidates_multiplier : int
        Candidate set size as a multiple of the requested batch size.
    warmup_steps : int
        Number of initial training steps before adaptive resampling
        activates.
    device : torch.device
        Target device.
    """

    def __init__(
        self,
        base_sampler: RiskNeutralSampler,
        n_candidates_multiplier: int = 10,
        warmup_steps: int = 2000,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device)
        self.base = base_sampler
        self.multiplier = n_candidates_multiplier
        self.warmup_steps = warmup_steps
        self._step = 0
        self._model: Optional[nn.Module] = None
        self._operator: Optional[Callable] = None

    def set_model_and_operator(
        self, model: nn.Module, operator: Callable
    ) -> None:
        """Register the model and PDE operator for residual evaluation.

        Parameters
        ----------
        model : nn.Module
        operator : callable
            Signature ``(model_fn, t, x, ...) -> residual``.
        """
        self._model = model
        self._operator = operator

    def step(self) -> None:
        """Increment the internal step counter."""
        self._step += 1

    def sample_interior(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._step < self.warmup_steps or self._model is None:
            return self.base.sample_interior(n)

        n_cand = self.multiplier * n
        t_cand, x_cand = self.base.sample_interior(n_cand)
        t_cand.requires_grad_(True)
        x_cand.requires_grad_(True)

        with torch.no_grad():
            try:
                residual = self._operator(t_cand, x_cand)
                weights = residual.detach().squeeze() ** 2
            except Exception:
                return self.base.sample_interior(n)

        weights = weights + 1e-8
        probs = weights / weights.sum()

        indices = torch.multinomial(probs, n, replacement=True)
        t_sel = t_cand[indices].detach()
        x_sel = x_cand[indices].detach()
        return t_sel, x_sel

    def sample_terminal(self, n: int) -> torch.Tensor:
        return self.base.sample_terminal(n)

    def sample_boundary(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base.sample_boundary(n)
