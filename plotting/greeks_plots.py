"""Delta and Gamma surface plots for Greeks comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from plotting.style import apply_style, save_figure


def plot_delta_1d(
    model: nn.Module,
    K: float,
    T: float,
    device: torch.device,
    analytical_delta_fn,
    save_dir: str,
    filename: str = "delta_1d",
) -> None:
    """Plot DGM delta vs analytical delta for 1-D case.

    Parameters
    ----------
    model : nn.Module
    K : float
    T : float
    device : torch.device
    analytical_delta_fn : callable
        ``(S, K, r, sigma, T, t) -> delta``
    save_dir : str
    filename : str
    """
    apply_style()
    S_vals = np.linspace(0.5 * K, 1.5 * K, 100)
    x_vals = np.log(S_vals / K)

    x_t = torch.tensor(x_vals.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.zeros(len(x_vals), 1, dtype=torch.float32, device=device)

    delta_dgm = model.get_delta(t_t, x_t).cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(S_vals, delta_dgm, "-", label="DGM", linewidth=2)

    if analytical_delta_fn is not None:
        delta_exact = analytical_delta_fn(S_vals)
        ax.plot(S_vals, delta_exact, "--", label="Analytical", linewidth=1.5)

    ax.set_xlabel("$S$")
    ax.set_ylabel("$\\Delta$")
    ax.set_title("Delta: DGM vs Analytical")
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")


def plot_gamma_1d(
    model: nn.Module,
    K: float,
    T: float,
    device: torch.device,
    analytical_gamma_fn,
    save_dir: str,
    filename: str = "gamma_1d",
) -> None:
    """Plot DGM gamma vs analytical gamma for 1-D case."""
    apply_style()
    S_vals = np.linspace(0.5 * K, 1.5 * K, 100)
    x_vals = np.log(S_vals / K)

    x_t = torch.tensor(x_vals.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.zeros(len(x_vals), 1, dtype=torch.float32, device=device)

    gamma_dgm = model.get_gamma(t_t, x_t).cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(S_vals, gamma_dgm, "-", label="DGM", linewidth=2)

    if analytical_gamma_fn is not None:
        gamma_exact = analytical_gamma_fn(S_vals)
        ax.plot(S_vals, gamma_exact, "--", label="Analytical", linewidth=1.5)

    ax.set_xlabel("$S$")
    ax.set_ylabel("$\\Gamma$")
    ax.set_title("Gamma: DGM vs Analytical")
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")


def plot_delta_surface_2d(
    model: nn.Module,
    K: float,
    T: float,
    device: torch.device,
    save_dir: str,
    filename: str = "delta_surface_2d",
    asset_index: int = 0,
) -> None:
    """Plot delta surface for a 2-asset option."""
    apply_style()
    n = 40
    s1 = np.linspace(0.5 * K, 1.5 * K, n)
    s2 = np.linspace(0.5 * K, 1.5 * K, n)
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")

    x1 = np.log(S1 / K)
    x2 = np.log(S2 / K)
    x_flat = np.stack([x1.ravel(), x2.ravel()], axis=-1)

    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device)
    t_t = torch.zeros(x_t.shape[0], 1, dtype=torch.float32, device=device)

    delta = model.get_delta(t_t, x_t).cpu().numpy()[:, asset_index].reshape(S1.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S1, S2, delta, cmap="coolwarm", alpha=0.8, edgecolor="none")
    ax.set_xlabel("$S_1$")
    ax.set_ylabel("$S_2$")
    ax.set_zlabel(f"$\\Delta_{asset_index + 1}$")
    ax.set_title(f"Delta Surface (Asset {asset_index + 1})")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/{filename}")
