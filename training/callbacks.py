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
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: Optional[float] = None

    def __call__(
        self, model: nn.Module, metric: float, step: int
    ) -> bool:
        """Save if this is the best metric so far.

        Returns True if saved.
        """
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            try:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                path = self.save_dir.absolute() / "best_model.pt"
                torch.save(
                    {"step": step, "metric": metric, "state_dict": model.state_dict()},
                    str(path),
                )
            except Exception:
                import time
                time.sleep(5)  # Wait for NFS metadata sync
                self.save_dir.mkdir(parents=True, exist_ok=True)
                path = self.save_dir.absolute() / "best_model.pt"
                torch.save(
                    {"step": step, "metric": metric, "state_dict": model.state_dict()},
                    str(path),
                )
            return True
        return False

    def save_latest(self, model: nn.Module, step: int) -> None:
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir.absolute() / "latest_model.pt"
            torch.save(
                {"step": step, "state_dict": model.state_dict()},
                str(path),
            )
        except Exception:
            import time
            time.sleep(5)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir.absolute() / "latest_model.pt"
            torch.save(
                {"step": step, "state_dict": model.state_dict()},
                str(path),
            )
