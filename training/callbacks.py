"""Training callbacks: early stopping, checkpointing, and LR logging."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of evaluations with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best: Optional[float] = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CheckpointSaver:
    """Save model checkpoint when evaluation metric improves.

    Parameters
    ----------
    save_dir : str
        Directory for checkpoint files.
    """

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self._ensure_dir()
        self.best_metric: Optional[float] = None

    def _ensure_dir(self):
        """Create save directory, ignoring filesystem errors."""
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # Directory might already exist or NFS is flaky

    def _save_with_retry(self, path: Path, data: dict, max_retries: int = 5) -> bool:
        """Save a checkpoint with multiple retries for HPC filesystem resilience."""
        import time
        import logging
        logger = logging.getLogger(__name__)

        for attempt in range(max_retries):
            try:
                self._ensure_dir()
                torch.save(data, str(path))
                return True
            except Exception as e:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                logger.warning(
                    f"Checkpoint save failed (attempt {attempt+1}/{max_retries}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        logger.error(f"Failed to save checkpoint after {max_retries} attempts. Continuing training...")
        return False

    def __call__(
        self, model: nn.Module, metric: float, step: int
    ) -> bool:
        """Save if this is the best metric so far.

        Returns True if saved.
        """
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            path = self.save_dir.absolute() / "best_model.pt"
            self._save_with_retry(
                path,
                {"step": step, "metric": metric, "state_dict": model.state_dict()},
            )
            return True
        return False

    def save_latest(self, model: nn.Module, step: int) -> None:
        path = self.save_dir.absolute() / "latest_model.pt"
        self._save_with_retry(
            path,
            {"step": step, "state_dict": model.state_dict()},
        )
