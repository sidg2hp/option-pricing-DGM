"""Publication-quality comparative plots for the paper.

Generates figures for scaling, three-way comparison, hybrid MC,
ablations, and training convergence overlays.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import apply_style, save_figure


def plot_scaling_bar_chart(
    results: Dict[int, Dict],
    save_dir: str,
) -> None:
    """Fig 1: Relative L² error vs dimension d (bar chart)."""
    apply_style()
    dims = sorted(results.keys())
    errors = [results[d]["rel_l2_error"] * 100 for d in dims]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(d) for d in dims], errors, color="#2C3E50", alpha=0.85, width=0.6)

    # Add value labels on bars
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{err:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Relative $L^2$ Error (%)")
    ax.set_title("DGM Scaling: Error vs Dimension")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/fig1_scaling_error")


def plot_comparison_grouped_bar(
    results: Dict[str, Dict],
    save_dir: str,
) -> None:
    """Fig 2: Three-way comparison bar chart: mean relative error by d."""
    apply_style()
    dims = sorted(results.keys(), key=int)
    dgm_errors = [results[d]["dgm_mean_rel_error"] * 100 for d in dims]
    zhou_errors = [results[d]["zhou_mean_rel_error"] * 100 for d in dims]

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - width / 2, dgm_errors, width, label="DGM (PDE)", color="#2C3E50", alpha=0.85)
    ax.bar(x + width / 2, zhou_errors, width, label="Zhou NN (Regression)", color="#E74C3C", alpha=0.85)

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Mean Relative Error vs MC (%)")
    ax.set_title("DGM vs Zhou et al.: Error Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/fig2_comparison_error")


def plot_variance_reduction_vs_d(
    results: Dict[str, Dict],
    save_dir: str,
) -> None:
    """Fig 3: Variance reduction vs dimension d."""
    apply_style()
    dims = sorted(results.keys(), key=int)
    vr = [results[d]["variance_reduction"] * 100 for d in dims]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(d) for d in dims], vr, color="#3498DB", alpha=0.85, width=0.6)

    for i, (d, v) in enumerate(zip(dims, vr)):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Variance Reduction (%)")
    ax.set_title("Hybrid DGM-MC: Variance Reduction vs Dimension")
    ax.set_ylim(0, 105)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/fig3_variance_reduction")


def plot_ablation_bar_chart(
    results: Dict[str, Dict],
    save_dir: str,
) -> None:
    """Fig 4: Ablation bar chart (all variants)."""
    apply_style()

    # Filter out entries with NaN or None errors
    valid = {k: v for k, v in results.items()
             if v.get("rel_l2_error") is not None and np.isfinite(v["rel_l2_error"])}

    names = list(valid.keys())
    errors = [valid[n]["rel_l2_error"] * 100 for n in names]

    # Prettier display names
    display_names = {
        "dgm_tanh": "DGM\n(tanh)",
        "mlp_tanh": "MLP\n(tanh)",
        "dgm_softplus": "DGM\n(softplus)",
        "soft_constraint": "Soft\nconstraint",
        "uniform_sampling": "Uniform\nsampling",
        "risk_neutral_sampling": "Risk-neutral\nsampling",
        "adaptive_sampling": "Adaptive\nsampling",
    }

    labels = [display_names.get(n, n) for n in names]
    colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, errors, color=colors[:len(names)], alpha=0.85, width=0.7)

    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{err:.2f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Relative $L^2$ Error (%)")
    ax.set_title("Ablation Study ($d=2$, Basket Call)")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/fig4_ablation")


def plot_convergence_overlay(
    loss_histories: Dict[int, List[float]],
    save_dir: str,
) -> None:
    """Fig 5: Training convergence overlay for multiple dimensions."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))

    colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]

    for i, (d, losses) in enumerate(sorted(loss_histories.items())):
        vals = np.array(losses)
        mask = np.isfinite(vals) & (vals > 0)
        if mask.any():
            ax.semilogy(np.where(mask)[0], vals[mask],
                        label=f"$d={d}$", color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel("Step (×500)")
    ax.set_ylabel("Total Loss (log scale)")
    ax.set_title("Training Convergence Across Dimensions")
    ax.legend()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/fig5_convergence_overlay")


def plot_comparison_price_table(
    results: Dict[str, Dict],
    d: int,
    save_dir: str,
) -> None:
    """Generate a price comparison table figure for a given d."""
    apply_style()
    r = results[str(d)]
    n = len(r["S0_test"])

    fig, ax = plt.subplots(figsize=(8, max(2, 0.6 * n + 1.5)))
    ax.axis("off")

    cols = ["$S_0$", "DGM", "Zhou NN", "MC ref.", "MC SE"]
    cell_text = []
    for i in range(n):
        cell_text.append([
            f"{r['S0_test'][i]:.2f}",
            f"{r['dgm_prices'][i]:.6f}",
            f"{r['zhou_prices'][i]:.6f}",
            f"{r['mc_prices'][i]:.6f}",
            f"{r['mc_errors'][i]:.6f}",
        ])

    table = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title(f"Price Comparison ($d={d}$)", pad=20)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{save_dir}/fig_price_table_d{d}")
