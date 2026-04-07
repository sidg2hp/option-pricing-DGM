"""3-D and 2-D option price surface plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from plotting.style import apply_style, save_figure


def plot_price_surface_2d(
    model: nn.Module,
    K: float,
    T: float,
    device: torch.device,
    save_dir: str,
    S_range: tuple = (0.5, 2.0),
    n_grid: int = 50,
    t_val: float = 0.0,
    filename: str = "price_surface_2d",
) -> None:
    """Plot 3-D price surface V(S1, S2) at fixed t for a 2-asset option.

    Parameters
    ----------
    model : nn.Module
    K : float
        Strike price.
    T : float
        Maturity.
    device : torch.device
    save_dir : str
    S_range : tuple
        S/K range as fractions.
    n_grid : int
    t_val : float
        Time at which to evaluate.
    filename : str
        Output file stem.
    """
    apply_style()
    s1 = np.linspace(S_range[0] * K, S_range[1] * K, n_grid)
    s2 = np.linspace(S_range[0] * K, S_range[1] * K, n_grid)
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")

    x1 = np.log(S1 / K)
    x2 = np.log(S2 / K)
    x_flat = np.stack([x1.ravel(), x2.ravel()], axis=-1)

    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device)
    t_t = torch.full((x_t.shape[0], 1), t_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        prices = model(t_t, x_t).cpu().numpy().reshape(S1.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S1, S2, prices, cmap="viridis", alpha=0.8, edgecolor="none")
    ax.set_xlabel("$S_1$")
    ax.set_ylabel("$S_2$")
    ax.set_zlabel("$V(t, S_1, S_2)$")
    ax.set_title(f"Price Surface at $t = {t_val}$")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")
