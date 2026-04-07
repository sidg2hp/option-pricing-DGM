"""Pointwise error heatmaps and PDE residual maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import apply_style, save_figure


def plot_error_heatmap(
    S1_grid: np.ndarray,
    S2_grid: np.ndarray,
    errors: np.ndarray,
    save_dir: str,
    filename: str = "error_heatmap",
    title: str = "Pointwise Error",
) -> None:
    """Plot a 2-D pointwise error heatmap.

    Parameters
    ----------
    S1_grid, S2_grid : np.ndarray
        Meshgrid arrays.
    errors : np.ndarray
        Error values on the grid.
    save_dir : str
    filename : str
    title : str
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(S1_grid, S2_grid, errors, cmap="hot", shading="auto")
    fig.colorbar(im, ax=ax, label="$|u_{\\theta} - u_{ref}|$")
    ax.set_xlabel("$S_1$")
    ax.set_ylabel("$S_2$")
    ax.set_title(title)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")


def plot_residual_map_multi_time(
    residual_grids: dict,
    x1_range: np.ndarray,
    x2_range: np.ndarray,
    save_dir: str,
    filename: str = "residual_map",
) -> None:
    """Plot PDE residual heatmaps at multiple time snapshots.

    Parameters
    ----------
    residual_grids : dict
        ``{t_value: residual_array}`` mapping.
    x1_range, x2_range : np.ndarray
        Grid axes.
    save_dir : str
    filename : str
    """
    apply_style()
    times = sorted(residual_grids.keys())
    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3), squeeze=False)
    X1, X2 = np.meshgrid(x1_range, x2_range, indexing="ij")

    vmax = max(np.max(residual_grids[t]) for t in times)
    for idx, t_val in enumerate(times):
        ax = axes[0, idx]
        im = ax.pcolormesh(
            X1, X2, residual_grids[t_val],
            cmap="hot", shading="auto", vmin=0, vmax=vmax,
        )
        ax.set_title(f"$t = {t_val:.2f}$")
        ax.set_xlabel("$x_1$")
        if idx == 0:
            ax.set_ylabel("$x_2$")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="$|\\mathcal{L}[u_{\\theta}]|$")
    fig.suptitle("PDE Residual Map", y=1.02)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")
