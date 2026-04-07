"""Structured logging utilities with optional Weights & Biases integration.

Falls back gracefully to local JSON logging when W&B is unavailable.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger with console output.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class MetricsLogger:
    """Log scalar metrics to a JSON-lines file and optionally to W&B.

    Parameters
    ----------
    log_dir : str
        Directory where the metrics file will be written.
    experiment_name : str
        Name of the experiment (used as W&B run name).
    use_wandb : bool
        Whether to initialise W&B logging.
    wandb_project : str
        W&B project name.
    config : dict or None
        Configuration dict to attach to the W&B run.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "default",
        use_wandb: bool = False,
        wandb_project: str = "dgm_option_pricing",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self._file = open(self.metrics_path, "a", encoding="utf-8")
        self._wandb_run = None

        if use_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config or {},
                    reinit=True,
                )
            except Exception:
                pass

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log a dictionary of metrics at a given training step.

        Parameters
        ----------
        metrics : dict
            Scalar metrics to record.
        step : int
            Global training step.
        """
        record = {"step": step, "timestamp": time.time(), **metrics}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        if self._wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception:
                pass

    def close(self) -> None:
        """Flush and close all log handles."""
        self._file.close()
        if self._wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
