"""Generate all NeurIPS-quality publication figures from experiment results.

Produces figures for:
  1. Dimensional scaling (error vs d)
  2. Method comparison grouped bar (DGM vs Zhou vs MC)
  3. Detailed price comparison line plots per dimension
  4. Hybrid MC variance reduction
  5. Ablation study (bar + training time Pareto)
  6. Training convergence overlay (PDE loss curves from ablation)
  7. Error decomposition by moneyness (ITM / ATM / OTM)
  8. Predicted vs Reference scatter plot
  9. SE reduction factor bar chart
  10. Summary comparison table

Usage:
    python experiments/generate_neurips_figures.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── NeurIPS-quality style ──────────────────────────────────────────────
# Colorblind-friendly palette (Tableau 10 colorblind-safe subset)
CB_BLUE    = "#0072B2"
CB_ORANGE  = "#E69F00"
CB_GREEN   = "#009E73"
CB_RED     = "#D55E00"
CB_PURPLE  = "#CC79A7"
CB_CYAN    = "#56B4E9"
CB_YELLOW  = "#F0E442"
CB_BLACK   = "#000000"

NEURIPS_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "text.usetex": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.framealpha": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
}


def apply_neurips_style():
    plt.rcParams.update(NEURIPS_STYLE)


def save_fig(fig, stem):
    """Save as both PDF (vector) and PNG (preview)."""
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── Helper constants ───────────────────────────────────────────────────
FIG_DIR = "figures/neurips"
S0_TEST = [0.80, 0.90, 1.00, 1.10, 1.20]


def main():
    apply_neurips_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load main data sources
    comp_path = "results/paper/comparison_summary.json"
    abl_path = "results/ablations/ablation_summary.json"
    hmc_path = "results/hybrid_mc/hybrid_mc_summary.json"

    comp = load_json(comp_path) if os.path.exists(comp_path) else {}
    abl = load_json(abl_path) if os.path.exists(abl_path) else {}
    hmc = load_json(hmc_path) if os.path.exists(hmc_path) else {}

    if not comp:
        print("ERROR: comparison_summary.json not found!")
        return

    dims = sorted([int(k) for k in comp.keys()])
    print(f"Found comparison data for d = {dims}")
    print(f"Found ablation data: {list(abl.keys())}")
    print(f"Found hybrid MC data for d = {list(hmc.keys())}")
    print()

    # ================================================================
    # FIGURE 1: Dimensional Scaling — Error vs d (line + marker)
    # ================================================================
    print("Generating Fig 1: Dimensional Scaling...")
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    dgm_errors = [comp[str(d)]["dgm_mean_rel_error"] * 100 for d in dims]
    zhou_errors = [comp[str(d)]["zhou_mean_rel_error"] * 100 for d in dims]

    ax.plot(dims, dgm_errors, 'o-', color=CB_BLUE, label="DGM (PDE solver)",
            markerfacecolor='white', markeredgewidth=1.8, markersize=8, zorder=5)
    ax.plot(dims, zhou_errors, 's--', color=CB_RED, label="Zhou et al. (MC regression)",
            markerfacecolor='white', markeredgewidth=1.8, markersize=7, zorder=5)

    # Annotate values
    for i, d in enumerate(dims):
        ax.annotate(f"{dgm_errors[i]:.1f}%", (d, dgm_errors[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=7.5,
                    color=CB_BLUE, fontweight='bold')
        ax.annotate(f"{zhou_errors[i]:.1f}%", (d, zhou_errors[i]),
                    textcoords="offset points", xytext=(0, -14), ha='center', fontsize=7.5,
                    color=CB_RED, fontweight='bold')

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Mean Relative Error vs MC (%)")
    ax.set_title("Pricing Error vs. Problem Dimension")
    ax.set_xticks(dims)
    ax.set_ylim(0, max(max(dgm_errors), max(zhou_errors)) * 1.35)
    ax.legend(loc='upper left', frameon=True)
    save_fig(fig, f"{FIG_DIR}/fig1_scaling_error")
    print("  [OK] fig1_scaling_error")

    # ================================================================
    # FIGURE 2: Grouped Bar — DGM vs Zhou (all dimensions)
    # ================================================================
    print("Generating Fig 2: Method Comparison Bar Chart...")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    x = np.arange(len(dims))
    width = 0.32

    b1 = ax.bar(x - width/2, dgm_errors, width, label="DGM (PDE solver)",
                color=CB_BLUE, alpha=0.9, edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + width/2, zhou_errors, width, label="Zhou et al. (MC regression)",
                color=CB_RED, alpha=0.9, edgecolor='white', linewidth=0.5)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.12,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Mean Relative Error (%)")
    ax.set_title("DGM vs Zhou et al.: Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$d={d}$" for d in dims])
    ax.legend(fontsize=8, frameon=True)
    ax.set_ylim(0, max(max(dgm_errors), max(zhou_errors)) * 1.3)
    save_fig(fig, f"{FIG_DIR}/fig2_method_comparison")
    print("  [OK] fig2_method_comparison")

    # ================================================================
    # FIGURE 3: Price Comparison Line Plots (subplots per d)
    # ================================================================
    print("Generating Fig 3: Price Comparison Lines...")
    n_dims = len(dims)
    n_cols = min(3, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows),
                             squeeze=False)

    for idx, d in enumerate(dims):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        r = comp[str(d)]

        ax.plot(S0_TEST, r["mc_prices"], 'k-', label="MC ref.", linewidth=2, alpha=0.7, zorder=3)
        ax.plot(S0_TEST, r["dgm_prices"], 'o-', color=CB_BLUE, label="DGM",
                markersize=5, markerfacecolor='white', markeredgewidth=1.2, zorder=4)
        ax.plot(S0_TEST, r["zhou_prices"], 's--', color=CB_RED, label="Zhou",
                markersize=4, markerfacecolor='white', markeredgewidth=1.2, zorder=4)
        ax.plot(S0_TEST, r["cv_prices"], '^:', color=CB_GREEN, label="Hybrid MC",
                markersize=4, markerfacecolor='white', markeredgewidth=1.2, zorder=4)

        # MC confidence band
        mc_lo = [r["mc_prices"][i] - 1.96 * r["mc_errors"][i] for i in range(5)]
        mc_hi = [r["mc_prices"][i] + 1.96 * r["mc_errors"][i] for i in range(5)]
        ax.fill_between(S0_TEST, mc_lo, mc_hi, color='gray', alpha=0.15, label="MC 95% CI")

        ax.set_title(f"$d = {d}$", fontsize=10)
        ax.set_xlabel("$S_0 / K$", fontsize=8)
        if col == 0:
            ax.set_ylabel("Option Price", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=6.5, loc='upper left', frameon=True)

    # Hide unused subplots
    for idx in range(n_dims, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Option Prices: DGM vs Zhou vs MC Reference", fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, f"{FIG_DIR}/fig3_price_comparison")
    print("  [OK] fig3_price_comparison")

    # ================================================================
    # FIGURE 4: Predicted vs True Scatter (DGM & Zhou)
    # ================================================================
    print("Generating Fig 4: Predicted vs True Scatter...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    all_mc = []
    all_dgm = []
    all_zhou = []
    all_dims_scatter = []
    for d in dims:
        r = comp[str(d)]
        all_mc.extend(r["mc_prices"])
        all_dgm.extend(r["dgm_prices"])
        all_zhou.extend(r["zhou_prices"])
        all_dims_scatter.extend([d] * len(r["mc_prices"]))

    all_mc = np.array(all_mc)
    all_dgm = np.array(all_dgm)
    all_zhou = np.array(all_zhou)

    # DGM scatter
    for d in dims:
        mask = np.array(all_dims_scatter) == d
        ax1.scatter(all_mc[mask], all_dgm[mask], s=40, alpha=0.8,
                    label=f"$d={d}$", zorder=5, edgecolors='white', linewidth=0.5)
    lim = [0, max(all_mc.max(), all_dgm.max()) * 1.1]
    ax1.plot(lim, lim, 'k--', alpha=0.4, linewidth=1, label="$y = x$")
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_xlabel("MC Reference Price")
    ax1.set_ylabel("DGM Predicted Price")
    ax1.set_title("DGM vs MC Reference")
    ax1.legend(fontsize=6.5, frameon=True, ncol=2)
    ax1.set_aspect('equal')

    # Zhou scatter
    for d in dims:
        mask = np.array(all_dims_scatter) == d
        ax2.scatter(all_mc[mask], all_zhou[mask], s=40, alpha=0.8,
                    label=f"$d={d}$", zorder=5, edgecolors='white', linewidth=0.5)
    lim2 = [0, max(all_mc.max(), all_zhou.max()) * 1.1]
    ax2.plot(lim2, lim2, 'k--', alpha=0.4, linewidth=1, label="$y = x$")
    ax2.set_xlim(lim2)
    ax2.set_ylim(lim2)
    ax2.set_xlabel("MC Reference Price")
    ax2.set_ylabel("Zhou Predicted Price")
    ax2.set_title("Zhou et al. vs MC Reference")
    ax2.legend(fontsize=6.5, frameon=True, ncol=2)
    ax2.set_aspect('equal')

    plt.tight_layout()
    save_fig(fig, f"{FIG_DIR}/fig4_scatter_pred_vs_true")
    print("  [OK] fig4_scatter_pred_vs_true")

    # ================================================================
    # FIGURE 5: Error by Moneyness (OTM / ATM / ITM breakdown)
    # ================================================================
    print("Generating Fig 5: Error by Moneyness...")
    # S0=0.8,0.9 = OTM; S0=1.0 = ATM; S0=1.1,1.2 = ITM
    moneyness_labels = ["OTM\n($S_0<K$)", "ATM\n($S_0=K$)", "ITM\n($S_0>K$)"]
    moneyness_idx = [[0, 1], [2], [3, 4]]  # indices into S0_TEST

    dgm_by_money = []
    zhou_by_money = []
    for group in moneyness_idx:
        dgm_errs_group = []
        zhou_errs_group = []
        for d in dims:
            r = comp[str(d)]
            for i in group:
                mc_p = r["mc_prices"][i]
                if mc_p > 1e-8:
                    dgm_errs_group.append(abs(r["dgm_prices"][i] - mc_p) / mc_p)
                    zhou_errs_group.append(abs(r["zhou_prices"][i] - mc_p) / mc_p)
        dgm_by_money.append(np.mean(dgm_errs_group) * 100)
        zhou_by_money.append(np.mean(zhou_errs_group) * 100)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x = np.arange(3)
    width = 0.32
    ax.bar(x - width/2, dgm_by_money, width, label="DGM", color=CB_BLUE, alpha=0.9)
    ax.bar(x + width/2, zhou_by_money, width, label="Zhou et al.", color=CB_RED, alpha=0.9)

    for i in range(3):
        ax.text(x[i] - width/2, dgm_by_money[i] + 0.15, f"{dgm_by_money[i]:.1f}%",
                ha='center', fontsize=7.5)
        ax.text(x[i] + width/2, zhou_by_money[i] + 0.15, f"{zhou_by_money[i]:.1f}%",
                ha='center', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(moneyness_labels)
    ax.set_ylabel("Mean Relative Error (%)")
    ax.set_title("Pricing Error by Moneyness Region")
    ax.legend(frameon=True)
    save_fig(fig, f"{FIG_DIR}/fig5_error_by_moneyness")
    print("  [OK] fig5_error_by_moneyness")

    # ================================================================
    # FIGURE 6: Hybrid MC — Variance Reduction + SE Comparison
    # ================================================================
    if hmc:
        print("Generating Fig 6: Hybrid MC Analysis...")
        hmc_dims = sorted(hmc.keys(), key=int)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

        # Left: SE comparison (log scale)
        vanilla_se = [hmc[d]["vanilla_se"] for d in hmc_dims]
        cv_se = [hmc[d]["cv_se"] for d in hmc_dims]

        x = np.arange(len(hmc_dims))
        width = 0.3
        ax1.bar(x - width/2, vanilla_se, width, label="Vanilla MC", color=CB_RED, alpha=0.9)
        ax1.bar(x + width/2, cv_se, width, label="DGM-CV MC", color=CB_GREEN, alpha=0.9)
        ax1.set_yscale("log")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"$d={d}$" for d in hmc_dims])
        ax1.set_ylabel("Standard Error (log scale)")
        ax1.set_title("MC Standard Error Comparison")
        ax1.legend(fontsize=7, frameon=True)

        # Right: SE reduction factor
        se_reduction = [hmc[d]["vanilla_se"] / hmc[d]["cv_se"] for d in hmc_dims]
        bars = ax2.bar([f"$d={d}$" for d in hmc_dims], se_reduction,
                       color=CB_CYAN, alpha=0.9, width=0.5,
                       edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, se_reduction):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     f"{val:.0f}×", ha="center", va="bottom", fontsize=8, fontweight='bold')
        ax2.set_ylabel("SE Reduction Factor")
        ax2.set_title("Variance Reduction Magnitude")

        plt.tight_layout()
        save_fig(fig, f"{FIG_DIR}/fig6_hybrid_mc")
        print("  [OK] fig6_hybrid_mc")

    # ================================================================
    # FIGURE 7: Ablation Study — Bar Chart + Training Time
    # ================================================================
    if abl:
        print("Generating Fig 7: Ablation Study...")

        display_names = {
            "dgm_tanh": "DGM\n(baseline)",
            "mlp_tanh": "MLP\n(baseline)",
            "soft_constraint": "Soft\nconstraint",
            "uniform_sampling": "Uniform\nsampling",
            "risk_neutral_sampling": "Risk-neutral\nsampling",
            "adaptive_sampling": "Adaptive\nsampling",
        }
        abl_colors = [CB_BLUE, CB_RED, CB_GREEN, CB_ORANGE, CB_PURPLE, CB_CYAN]

        valid_names = []
        errors = []
        times = []
        for name in ["dgm_tanh", "mlp_tanh", "soft_constraint",
                      "uniform_sampling", "risk_neutral_sampling", "adaptive_sampling"]:
            if name in abl:
                entry = abl[name]
                err = entry.get("rel_l2_error")
                if err is not None and np.isfinite(err):
                    valid_names.append(name)
                    errors.append(err * 100)
                    t = entry.get("train_time_seconds") or entry.get("train_time", 0)
                    times.append(t / 3600)  # convert to hours

        if valid_names:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

            # Left: Error bar chart
            labels = [display_names.get(n, n) for n in valid_names]
            bars = ax1.bar(labels, errors,
                           color=abl_colors[:len(valid_names)], alpha=0.9,
                           width=0.6, edgecolor='white', linewidth=0.5)
            for bar, err in zip(bars, errors):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                         f"{err:.2f}%", ha="center", va="bottom", fontsize=7.5)
            ax1.set_ylabel("Relative $L^2$ Error (%)")
            ax1.set_title("Ablation: Pricing Accuracy ($d=2$)")

            # Right: Accuracy vs Training Time (Pareto)
            for i, name in enumerate(valid_names):
                ax2.scatter(times[i], errors[i], s=80, color=abl_colors[i],
                            edgecolors='black', linewidth=0.8, zorder=5)
                ax2.annotate(display_names.get(name, name).replace('\n', ' '),
                             (times[i], errors[i]),
                             textcoords="offset points", xytext=(8, 5),
                             fontsize=7, ha='left')
            ax2.set_xlabel("Training Time (hours)")
            ax2.set_ylabel("Relative $L^2$ Error (%)")
            ax2.set_title("Accuracy vs. Computational Cost")

            plt.tight_layout()
            save_fig(fig, f"{FIG_DIR}/fig7_ablation")
            print("  [OK] fig7_ablation")

    # ================================================================
    # FIGURE 8: Training Convergence — PDE Loss Curves
    # ================================================================
    if abl:
        # Only plot configs that have loss_history
        configs_with_history = {}
        for name in ["dgm_tanh", "mlp_tanh"]:
            if name in abl and "loss_history" in abl[name]:
                configs_with_history[name] = abl[name]

        if configs_with_history:
            print("Generating Fig 8: Training Convergence...")
            fig, ax = plt.subplots(figsize=(5, 3.5))

            style_map = {
                "dgm_tanh": {"color": CB_BLUE, "label": "DGM (tanh)", "ls": "-"},
                "mlp_tanh": {"color": CB_RED, "label": "MLP (tanh)", "ls": "--"},
            }

            for name, data in configs_with_history.items():
                pde_loss = data["loss_history"]["loss_pde"]
                steps = np.linspace(0, 50000, len(pde_loss))
                style = style_map.get(name, {"color": CB_BLACK, "label": name, "ls": "-"})

                # Smooth with moving average for readability
                window = max(1, len(pde_loss) // 50)
                if window > 1:
                    kernel = np.ones(window) / window
                    smoothed = np.convolve(pde_loss, kernel, mode='valid')
                    steps_smooth = steps[:len(smoothed)]
                else:
                    smoothed = pde_loss
                    steps_smooth = steps

                ax.semilogy(steps_smooth, smoothed, color=style["color"],
                            linestyle=style["ls"], label=style["label"], linewidth=1.5)

            ax.set_xlabel("Training Step")
            ax.set_ylabel("PDE Residual Loss (log scale)")
            ax.set_title("Training Convergence: DGM vs MLP")
            ax.legend(frameon=True)
            ax.set_xlim(0, 50000)
            save_fig(fig, f"{FIG_DIR}/fig8_convergence")
            print("  [OK] fig8_convergence")

    # ================================================================
    # FIGURE 9: Component-wise Loss Curves (PDE + Boundary)
    # ================================================================
    if abl and "dgm_tanh" in abl and "loss_history" in abl["dgm_tanh"]:
        print("Generating Fig 9: Component-wise Loss...")
        data = abl["dgm_tanh"]["loss_history"]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        n = len(data["loss_pde"])
        steps = np.linspace(0, 50000, n)

        ax.semilogy(steps, data["loss_pde"], color=CB_BLUE, label="PDE loss ($\\mathcal{J}_{PDE}$)",
                    linewidth=1.2, alpha=0.8)
        ax.semilogy(steps, data["loss_boundary"], color=CB_RED, label="Boundary loss ($\\mathcal{J}_{bnd}$)",
                    linewidth=1.2, alpha=0.8)
        ax.semilogy(steps, data["loss_total"], color=CB_BLACK, label="Total loss",
                    linewidth=1.8, alpha=0.6, linestyle='--')

        # Mark smoothing annealing end (30% of training)
        ax.axvline(x=50000 * 0.3, color=CB_ORANGE, linestyle=':', alpha=0.6, linewidth=1)
        ax.text(50000 * 0.3 + 500, ax.get_ylim()[1] * 0.3, "$\\epsilon \\to 0$",
                fontsize=8, color=CB_ORANGE)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Component-wise Loss Decomposition ($d=2$, DGM)")
        ax.legend(frameon=True, fontsize=7.5)
        ax.set_xlim(0, 50000)
        save_fig(fig, f"{FIG_DIR}/fig9_component_loss")
        print("  [OK] fig9_component_loss")

    # ================================================================
    # FIGURE 10: Comprehensive Summary Table
    # ================================================================
    print("Generating Fig 10: Summary Table...")
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")

    cols = ["$d$", "DGM Error", "Zhou Error", "MC SE",
            "Hybrid MC SE", "VR Factor"]
    cell_text = []
    for d in dims:
        r = comp[str(d)]
        hmc_d = hmc.get(str(d), {})
        vr = hmc_d.get("vanilla_se", 0) / hmc_d.get("cv_se", 1) if hmc_d.get("cv_se", 0) > 0 else "—"
        vr_str = f"{vr:.0f}×" if isinstance(vr, float) else vr

        cell_text.append([
            str(d),
            f"{r['dgm_mean_rel_error']*100:.2f}%",
            f"{r['zhou_mean_rel_error']*100:.2f}%",
            f"{np.mean(r['mc_errors']):.2e}",
            f"{np.mean(r.get('cv_errors', [0])):.2e}",
            vr_str,
        ])

    table = ax.table(cellText=cell_text, colLabels=cols,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 1.6)

    # Style header
    for j in range(len(cols)):
        table[0, j].set_facecolor(CB_BLUE)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(1, len(cell_text) + 1):
        for j in range(len(cols)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#F0F4F8")

    ax.set_title("Summary of Results Across Dimensions", pad=25, fontsize=12)
    save_fig(fig, f"{FIG_DIR}/fig10_summary_table")
    print("  [OK] fig10_summary_table")

    # ================================================================
    # FIGURE 11: Detailed Price Tables (per dimension)
    # ================================================================
    print("Generating Fig 11: Detailed Price Tables...")
    for d in dims:
        r = comp[str(d)]
        n = len(r["S0_test"])

        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * n + 1.5)))
        ax.axis("off")

        cols = ["$S_0/K$", "DGM", "Zhou NN", "Hybrid MC", "MC ref.", "MC SE"]
        cell_text = []
        for i in range(n):
            cell_text.append([
                f"{r['S0_test'][i]:.2f}",
                f"{r['dgm_prices'][i]:.6f}",
                f"{r['zhou_prices'][i]:.6f}",
                f"{r['cv_prices'][i]:.6f}",
                f"{r['mc_prices'][i]:.6f}",
                f"{r['mc_errors'][i]:.2e}",
            ])

        table = ax.table(cellText=cell_text, colLabels=cols,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        for j in range(len(cols)):
            table[0, j].set_facecolor(CB_BLUE)
            table[0, j].set_text_props(color="white", fontweight="bold")

        ax.set_title(f"Detailed Price Comparison ($d={d}$)", pad=20, fontsize=11)
        save_fig(fig, f"{FIG_DIR}/fig11_prices_d{d}")
        print(f"  [OK] fig11_prices_d{d}")

    print(f"\n[OK] All {len(os.listdir(FIG_DIR))//2} figures generated in {FIG_DIR}/")
    print("  (Each saved as both PDF and PNG)")


if __name__ == "__main__":
    main()
