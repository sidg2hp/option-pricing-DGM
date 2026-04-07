"""Learning rate scheduler with cosine annealing and warm restarts."""

import math

import torch.optim as optim


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    T_0: int = 10_000,
    T_mult: int = 2,
    eta_min: float = 1e-5,
) -> optim.lr_scheduler._LRScheduler:
    """Create a learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    scheduler_type : str
        ``"cosine_warm_restarts"`` or ``"cosine"``.
    T_0 : int
        Period of the first cosine cycle (in steps).
    T_mult : int
        Multiplicative factor for successive cycle lengths.
    eta_min : float
        Minimum learning rate.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
    """
    if scheduler_type == "cosine_warm_restarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_0, eta_min=eta_min
        )
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
