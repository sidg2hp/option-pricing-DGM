"""Centralized random seed management for reproducibility.

Provides a single entry point to seed all random number generators used
throughout the codebase: Python stdlib, NumPy, and PyTorch (CPU + CUDA).
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all random number generators for full reproducibility.

    Parameters
    ----------
    seed : int
        The global random seed. Default is 42.

    Notes
    -----
    Sets ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False`` to ensure deterministic
    CUDA behaviour at the cost of some performance.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
