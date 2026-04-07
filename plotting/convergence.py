"""Training convergence and loss decomposition plots."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import apply_style, save_figure


def plot_training_convergence(
    loss_history: Dict[str, List[float]],
    save_dir: str,
    filename: str = "training_convergence",
) -> None:
    """Plot training loss curves on a log-y scale.

    Parameters
    ----------
    loss_history : dict
        Keys: ``loss_total``, ``loss_pde``, ``loss_terminal``, ``loss_boundary``.
    save_dir : str
    filename : str
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))

    for key, label in [
        ("loss_total", "Total"),
        ("loss_pde", "PDE Residual"),
        ("loss_terminal", "Terminal"),
        ("loss_boundary", "Boundary"),
    ]:
        if key in loss_history and len(loss_history[key]) > 0:
            vals = np.array(loss_history[key])
            if np.any(vals > 0):
                ax.semilogy(vals, label=label, alpha=0.8)

    ax.set_xlabel("Logging Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Convergence")
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")


def plot_loss_decomposition(
    loss_history: Dict[str, List[float]],
    save_dir: str,
    filename: str = "loss_decomposition",
) -> None:
    """Plot PDE and terminal loss components separately.

    Parameters
    ----------
    loss_history : dict
    save_dir : str
    filename : str
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if "loss_pde" in loss_history and loss_history["loss_pde"]:
        ax1.semilogy(loss_history["loss_pde"], color="#E74C3C")
        ax1.set_title("PDE Residual Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")

    if "loss_boundary" in loss_history and loss_history["loss_boundary"]:
        ax2.semilogy(loss_history["loss_boundary"], color="#3498DB")
        ax2.set_title("Boundary Loss")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")

    fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")
