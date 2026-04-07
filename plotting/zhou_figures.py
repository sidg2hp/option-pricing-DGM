"""Reproduces all figures from Zhou et al. (2021) plus DGM extension figures.

Each function generates one figure and saves it as PDF and PNG.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from configs.base_config import MarketConfig
from plotting.style import apply_style, save_figure


def plot_zhou_figure_1_price_surface_2d(
    dgm_model: nn.Module,
    market: MarketConfig,
    device: torch.device,
    save_dir: str,
    mc_points: Optional[np.ndarray] = None,
) -> None:
    """Figure Z-1: 3-D price surface V(S1, S2) at t=0 for d=2."""
    apply_style()
    K = market.K
    n = 50
    s_range = np.linspace(0.5 * K, 2.0 * K, n)
    S1, S2 = np.meshgrid(s_range, s_range, indexing="ij")
    x1, x2 = np.log(S1 / K), np.log(S2 / K)
    x_flat = np.stack([x1.ravel(), x2.ravel()], axis=-1)

    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device)
    t_t = torch.zeros(x_t.shape[0], 1, dtype=torch.float32, device=device)

    with torch.no_grad():
        prices = dgm_model(t_t, x_t).cpu().numpy().reshape(S1.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S1, S2, prices, cmap="viridis", alpha=0.85, edgecolor="none")

    if mc_points is not None:
        ax.scatter(mc_points[:, 0], mc_points[:, 1], mc_points[:, 2],
                   c="red", s=2, alpha=0.3, label="MC samples")

    ax.set_xlabel("$S_1$")
    ax.set_ylabel("$S_2$")
    ax.set_zlabel("$V(0, S_1, S_2)$")
    ax.set_title("Price Surface at $t=0$")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/zhou_fig1_price_surface_2d")


def plot_zhou_figure_2_training_convergence(
    loss_history: Dict[str, List[float]],
    save_dir: str,
) -> None:
    """Figure Z-2: Training loss vs step."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))

    for key, label, color in [
        ("loss_total", "Total Loss", "#2C3E50"),
        ("loss_pde", "PDE Residual", "#E74C3C"),
        ("loss_boundary", "Boundary", "#3498DB"),
    ]:
        if key in loss_history and loss_history[key]:
            vals = np.array(loss_history[key])
            if np.any(vals > 0):
                ax.semilogy(vals, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Convergence")
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/zhou_fig2_training_loss")


def plot_zhou_figure_3_error_vs_n_assets(
    results: Dict[int, Dict[str, float]],
    save_dir: str,
    zhou_results: Optional[Dict[int, float]] = None,
) -> None:
    """Figure Z-4: Relative error vs number of assets d."""
    apply_style()
    dims = sorted(results.keys())
    errors = [results[d].get("rel_l2_error", 0) * 100 for d in dims]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(d) for d in dims], errors, color="#2C3E50", alpha=0.8, label="DGM")

    if zhou_results:
        zhou_dims = sorted(zhou_results.keys())
        zhou_errors = [zhou_results[d] * 100 for d in zhou_dims]
        x_pos = [dims.index(d) for d in zhou_dims if d in dims]
        ax.bar([str(d) for d in zhou_dims], zhou_errors,
               color="#E74C3C", alpha=0.5, label="Zhou et al.", width=0.35)

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Relative L2 Error (%)")
    ax.set_title("Error vs Number of Assets")
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/zhou_fig4_error_vs_d")


def plot_zhou_figure_4_price_comparison_table(
    dgm_prices: np.ndarray,
    mc_prices: np.ndarray,
    mc_errors: np.ndarray,
    analytical_prices: Optional[np.ndarray],
    S0_labels: List[str],
    save_dir: str,
) -> None:
    """Figure Z-3: Price comparison table across methods."""
    apply_style()
    n = len(S0_labels)

    fig, ax = plt.subplots(figsize=(8, max(2, 0.6 * n + 1.5)))
    ax.axis("off")

    cols = ["$S_0$", "DGM", "MC", "MC Std Err"]
    if analytical_prices is not None:
        cols.append("Analytical")

    cell_text = []
    for i in range(n):
        row = [S0_labels[i], f"{dgm_prices[i]:.6f}", f"{mc_prices[i]:.6f}",
               f"{mc_errors[i]:.6f}"]
        if analytical_prices is not None:
            row.append(f"{analytical_prices[i]:.6f}")
        cell_text.append(row)

    table = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title("Price Comparison", pad=20)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/zhou_fig3_price_table")

    import csv
    csv_path = Path(save_dir) / "zhou_fig3_price_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(cell_text)


def plot_zhou_figure_5_scatter_dgm_vs_mc(
    dgm_prices: np.ndarray,
    mc_prices: np.ndarray,
    save_dir: str,
) -> None:
    """Figure Z-5: Scatter plot of DGM vs MC prices."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(mc_prices, dgm_prices, s=10, alpha=0.6, color="#2C3E50")
    lims = [min(mc_prices.min(), dgm_prices.min()),
            max(mc_prices.max(), dgm_prices.max())]
    ax.plot(lims, lims, "--", color="#E74C3C", linewidth=1, label="Perfect fit")

    ax.set_xlabel("MC Price")
    ax.set_ylabel("DGM Price")
    ax.set_title("DGM vs MC Prices")
    ax.legend()
    ax.set_aspect("equal")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/zhou_fig5_scatter_dgm_vs_mc")


def plot_zhou_figure_6_delta(
    model: nn.Module,
    K: float,
    device: torch.device,
    save_dir: str,
    analytical_fn=None,
) -> None:
    """Figure Z-6: Delta vs S at t=0."""
    apply_style()
    S = np.linspace(0.5 * K, 1.5 * K, 100)
    x = np.log(S / K).reshape(-1, 1)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    t_t = torch.zeros(len(x), 1, dtype=torch.float32, device=device)
    delta = model.get_delta(t_t, x_t).cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(S, delta, "-", label="DGM", color="#2C3E50", linewidth=2)
    if analytical_fn is not None:
        ax.plot(S, analytical_fn(S), "--", label="Analytical", color="#E74C3C", linewidth=1.5)
    ax.set_xlabel("$S$")
    ax.set_ylabel("$\\Delta$")
    ax.set_title("Delta at $t=0$")
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/zhou_fig6_delta")


def plot_zhou_figure_7_delta_surface(
    model: nn.Module,
    K: float,
    device: torch.device,
    save_dir: str,
) -> None:
    """Figure Z-7: 3-D delta surface for d=2."""
    apply_style()
    n = 40
    s = np.linspace(0.5 * K, 1.5 * K, n)
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    x_flat = np.stack([np.log(S1 / K).ravel(), np.log(S2 / K).ravel()], axis=-1)
    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device)
    t_t = torch.zeros(x_t.shape[0], 1, dtype=torch.float32, device=device)

    delta = model.get_delta(t_t, x_t).cpu().numpy()[:, 0].reshape(S1.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S1, S2, delta, cmap="coolwarm", alpha=0.85, edgecolor="none")
    ax.set_xlabel("$S_1$")
    ax.set_ylabel("$S_2$")
    ax.set_zlabel("$\\Delta_1$")
    ax.set_title("Delta Surface at $t=0$")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/zhou_fig7_delta_surface")


def plot_zhou_figure_8_computational_cost(
    timing: Dict[str, Dict[str, float]],
    save_dir: str,
) -> None:
    """Figure Z-8: Computational cost comparison."""
    apply_style()
    dims = sorted([int(k) for k in timing.keys()])

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, color, marker in [
        ("dgm_time", "#2C3E50", "o"),
        ("mc_time", "#E74C3C", "s"),
    ]:
        times = [timing[str(d)].get(method, 0) for d in dims]
        label = "DGM" if "dgm" in method else "Monte Carlo"
        ax.semilogy(dims, times, f"{marker}-", color=color, label=label)

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Computational Cost Comparison")
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/zhou_fig8_timing")


def plot_dgm_pde_residual_map(
    model: nn.Module,
    operator: Callable,
    market: MarketConfig,
    device: torch.device,
    save_dir: str,
) -> None:
    """DGM Figure D-1: PDE residual heatmap at multiple times."""
    apply_style()
    from evaluation.diagnostics import compute_residual_grid

    sigma = np.array(market.sigma)
    x_range = np.linspace(-0.6, 0.6, 40)
    times = [0.0, 0.25, 0.5, 0.75, 0.99]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3), squeeze=True)

    for idx, t_val in enumerate(times):
        res = compute_residual_grid(model, operator, x_range, x_range, t_val, device)
        im = axes[idx].pcolormesh(x_range, x_range, res, cmap="hot", shading="auto")
        axes[idx].set_title(f"$t = {t_val}$")
        axes[idx].set_xlabel("$x_1$")
        if idx == 0:
            axes[idx].set_ylabel("$x_2$")

    fig.colorbar(im, ax=axes.tolist(), label="$|\\mathcal{L}[u_{\\theta}]|$", shrink=0.8)
    fig.suptitle("PDE Residual Map", y=1.02)
    fig.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/dgm_fig1_residual_map")


def plot_dgm_full_price_surface_time_evolution(
    model: nn.Module,
    market: MarketConfig,
    device: torch.device,
    save_dir: str,
) -> None:
    """DGM Figure D-4: Price surface at 6 time snapshots."""
    apply_style()
    K = market.K
    n = 40
    s = np.linspace(0.5 * K, 1.5 * K, n)
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    x_flat = np.stack([np.log(S1 / K).ravel(), np.log(S2 / K).ravel()], axis=-1)
    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device)

    times = [0.0, 0.2, 0.4, 0.6, 0.8, market.T * 0.99]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), subplot_kw={"projection": "3d"})

    for idx, t_val in enumerate(times):
        ax = axes[idx // 3, idx % 3]
        t_t = torch.full((x_t.shape[0], 1), t_val, dtype=torch.float32, device=device)
        with torch.no_grad():
            prices = model(t_t, x_t).cpu().numpy().reshape(S1.shape)
        ax.plot_surface(S1, S2, prices, cmap="viridis", alpha=0.8, edgecolor="none")
        ax.set_title(f"$t = {t_val:.1f}$")
        ax.set_xlabel("$S_1$")
        ax.set_ylabel("$S_2$")

    fig.suptitle("Price Surface Time Evolution", y=1.0)
    fig.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/dgm_fig4_time_evolution")


def plot_dgm_adaptive_sampling_comparison(
    uniform_errors: List[float],
    adaptive_errors: List[float],
    save_dir: str,
) -> None:
    """DGM Figure D-5: Adaptive vs uniform sampling error curves."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(uniform_errors, label="Uniform Sampling", color="#2C3E50")
    ax.semilogy(adaptive_errors, label="Adaptive Sampling", color="#E74C3C")
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel("Relative $L^2$ Error")
    ax.set_title("Adaptive vs Uniform Sampling")
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/dgm_fig5_adaptive_sampling")


def plot_dgm_ablation_architecture(
    results: Dict[str, float],
    save_dir: str,
) -> None:
    """DGM Figure D-7: Architecture ablation."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(results.keys())
    errors = [results[n] * 100 for n in names]
    colors = ["#2C3E50", "#E74C3C", "#3498DB"][:len(names)]
    ax.bar(names, errors, color=colors, alpha=0.8)
    ax.set_ylabel("Relative L2 Error (%)")
    ax.set_title("Architecture Comparison")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/dgm_fig7_ablation_architecture")


def plot_dgm_ablation_activation(
    results: Dict[str, float],
    save_dir: str,
) -> None:
    """DGM Figure D-8: Activation function ablation."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(results.keys())
    errors = [results[n] * 100 for n in names]
    colors = ["#2C3E50", "#E74C3C"][:len(names)]
    ax.bar(names, errors, color=colors, alpha=0.8)
    ax.set_ylabel("Relative L2 Error (%)")
    ax.set_title("Activation Function Comparison")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/dgm_fig8_ablation_activation")


def plot_all_zhou_figures(results_bundle: Dict, save_dir: str) -> None:
    """Master function: generate all figures from a results bundle.

    Parameters
    ----------
    results_bundle : dict
        Must contain keys appropriate for each figure function.
    save_dir : str
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if "loss_history" in results_bundle:
        plot_zhou_figure_2_training_convergence(
            results_bundle["loss_history"], save_dir
        )

    if "scaling_results" in results_bundle:
        plot_zhou_figure_3_error_vs_n_assets(
            results_bundle["scaling_results"], save_dir
        )

    if "dgm_prices" in results_bundle and "mc_prices" in results_bundle:
        plot_zhou_figure_5_scatter_dgm_vs_mc(
            results_bundle["dgm_prices"],
            results_bundle["mc_prices"],
            save_dir,
        )

    if "timing" in results_bundle:
        plot_zhou_figure_8_computational_cost(
            results_bundle["timing"], save_dir
        )
