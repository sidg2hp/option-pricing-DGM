"""Scaling study plots: error vs dimension and training time vs dimension."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import apply_style, save_figure


def plot_scaling_error(
    results: Dict[int, Dict[str, float]],
    save_dir: str,
    filename: str = "scaling_error",
) -> None:
    """Plot relative L2 error vs number of assets on a log-log scale.

    Parameters
    ----------
    results : dict
        ``{d: {"rel_l2_error": float, ...}}``
    save_dir : str
    filename : str
    """
    apply_style()
    dims = sorted(results.keys())
    errors = [results[d]["rel_l2_error"] for d in dims]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(dims, errors, "o-", color="#2C3E50", markersize=6)

    if len(dims) >= 3:
        log_d = np.log(np.array(dims, dtype=float))
        log_e = np.log(np.array(errors))
        mask = np.isfinite(log_e)
        if mask.sum() >= 2:
            alpha, intercept = np.polyfit(log_d[mask], log_e[mask], 1)
            d_fit = np.linspace(min(dims), max(dims), 50)
            e_fit = np.exp(intercept) * d_fit**alpha
            ax.loglog(d_fit, e_fit, "--", color="#E74C3C", alpha=0.7,
                      label=f"Fit: error ~ $d^{{{alpha:.2f}}}$")
            ax.legend()

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Relative $L^2$ Error")
    ax.set_title("DGM Scaling: Error vs Dimension")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")


def plot_scaling_time(
    results: Dict[int, Dict[str, float]],
    save_dir: str,
    filename: str = "scaling_time",
) -> None:
    """Plot training wall-clock time vs number of assets.

    Parameters
    ----------
    results : dict
        ``{d: {"train_time_seconds": float, ...}}``
    save_dir : str
    filename : str
    """
    apply_style()
    dims = sorted(results.keys())
    times = [results[d]["train_time_seconds"] for d in dims]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(dims, times, "s-", color="#3498DB", markersize=6)

    if len(dims) >= 3:
        log_d = np.log(np.array(dims, dtype=float))
        log_t = np.log(np.array(times))
        mask = np.isfinite(log_t)
        if mask.sum() >= 2:
            beta, intercept = np.polyfit(log_d[mask], log_t[mask], 1)
            d_fit = np.linspace(min(dims), max(dims), 50)
            t_fit = np.exp(intercept) * d_fit**beta
            ax.loglog(d_fit, t_fit, "--", color="#E74C3C", alpha=0.7,
                      label=f"Fit: time ~ $d^{{{beta:.2f}}}$")
            ax.legend()

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("DGM Scaling: Training Time vs Dimension")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")
