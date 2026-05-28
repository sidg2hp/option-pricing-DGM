"""Generate ALL NeurIPS-quality publication figures.

Complete figure set for DGM option pricing paper:
  1.  Scaling: DGM error vs dimension (comparison-based, accurate)
  2.  4-way grouped bar: DGM vs Zhou vs Hybrid MC vs MC
  3.  Price comparison line plots per dimension (all 4 methods)
  4.  Predicted vs Reference scatter
  5.  Error by moneyness (OTM/ATM/ITM)
  6.  Hybrid MC variance reduction
  7.  Ablation study (bar + Pareto)
  8.  Training convergence (DGM vs MLP)
  9.  Component-wise loss decomposition
  10. Scaling: training time vs dimension
  11. Scaling: parameter count vs dimension
  12. Summary table (all results)
  13. Detailed price tables per dimension
  14. Formatted text tables for paper

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

# -- Colorblind-friendly palette (Okabe-Ito) --------------------------------
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
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_json(path):
    with open(path) as f:
        return json.load(f)


FIG_DIR = "figures/publication"
S0_TEST = [0.80, 0.90, 1.00, 1.10, 1.20]


def main():
    apply_neurips_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load data sources
    comp_path = "results/publication/comparison_summary.json"
    abl_path = "results/ablations/ablation_summary.json"
    hmc_path = "results/hybrid_mc/hybrid_mc_summary.json"

    comp = load_json(comp_path) if os.path.exists(comp_path) else {}
    abl = load_json(abl_path) if os.path.exists(abl_path) else {}
    hmc = load_json(hmc_path) if os.path.exists(hmc_path) else {}

    # Load scaling results
    scaling = {}
    for d in [1, 2, 3, 5, 7, 10]:
        for p in [f"results/scaling/d_{d}/result.json",
                  f"results_from_cluster/scaling/d_{d}/result.json"]:
            if os.path.exists(p):
                scaling[d] = load_json(p)
                break

    comp_dims = sorted([int(k) for k in comp.keys()])
    all_dims = sorted(set(comp_dims) | set(scaling.keys()))

    print(f"Comparison data: d = {comp_dims}")
    print(f"Scaling data: d = {sorted(scaling.keys())}")
    print(f"Hybrid MC data: d = {sorted(hmc.keys(), key=int) if hmc else []}")
    print(f"Ablation configs: {list(abl.keys())}")
    print()

    # ================================================================
    # FIG 1: Scaling error (comparison-based, most accurate)
    # ================================================================
    print("Fig 1: Scaling error vs dimension...")
    fig, ax = plt.subplots(figsize=(5, 3.5))

    dgm_errs = [comp[str(d)]["dgm_mean_rel_error"] * 100 for d in comp_dims]
    zhou_errs = [comp[str(d)]["zhou_mean_rel_error"] * 100 for d in comp_dims]

    ax.plot(comp_dims, dgm_errs, 'o-', color=CB_BLUE, label="DGM (PDE solver)",
            markerfacecolor='white', markeredgewidth=1.8, markersize=8, zorder=5)
    ax.plot(comp_dims, zhou_errs, 's--', color=CB_RED, label="Zhou et al. (MC regression)",
            markerfacecolor='white', markeredgewidth=1.8, markersize=7, zorder=5)

    for i, d in enumerate(comp_dims):
        ax.annotate(f"{dgm_errs[i]:.1f}%", (d, dgm_errs[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=7.5, color=CB_BLUE, fontweight='bold')
        ax.annotate(f"{zhou_errs[i]:.1f}%", (d, zhou_errs[i]),
                    textcoords="offset points", xytext=(0, -14), ha='center',
                    fontsize=7.5, color=CB_RED, fontweight='bold')

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Mean Relative Error vs MC (%)")
    ax.set_title("Pricing Error vs. Problem Dimension")
    ax.set_xticks(comp_dims)
    ax.set_ylim(0, max(max(dgm_errs), max(zhou_errs)) * 1.4)
    ax.legend(loc='upper left', frameon=True)
    save_fig(fig, f"{FIG_DIR}/fig1_scaling_error")
    print("  [OK] fig1_scaling_error")

    # ================================================================
    # FIG 2: 4-way grouped bar (DGM vs Zhou vs Hybrid MC vs MC SE)
    # ================================================================
    print("Fig 2: 4-way method comparison...")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    x = np.arange(len(comp_dims))
    width = 0.2

    # Compute Hybrid MC mean relative error vs MC reference
    hmc_errs = []
    for d in comp_dims:
        r = comp[str(d)]
        mc_p = np.array(r["mc_prices"])
        cv_p = np.array(r["cv_prices"])
        mask = mc_p > 1e-8
        hmc_errs.append(np.mean(np.abs(cv_p[mask] - mc_p[mask]) / mc_p[mask]) * 100)

    b1 = ax.bar(x - 1.5*width, dgm_errs, width, label="DGM (PDE)", color=CB_BLUE, alpha=0.9)
    b2 = ax.bar(x - 0.5*width, zhou_errs, width, label="Zhou (regression)", color=CB_RED, alpha=0.9)
    b3 = ax.bar(x + 0.5*width, hmc_errs, width, label="Hybrid DGM-MC", color=CB_GREEN, alpha=0.9)

    # MC SE as very small bars (scaled up for visibility)
    mc_se_pct = []
    for d in comp_dims:
        r = comp[str(d)]
        mc_p = np.array(r["mc_prices"])
        mc_e = np.array(r["mc_errors"])
        mask = mc_p > 1e-8
        mc_se_pct.append(np.mean(mc_e[mask] / mc_p[mask]) * 100)
    b4 = ax.bar(x + 1.5*width, mc_se_pct, width, label="MC std. error", color=CB_ORANGE, alpha=0.9)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.08,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=6)

    ax.set_xlabel("Number of Assets $d$")
    ax.set_ylabel("Mean Relative Error vs MC (%)")
    ax.set_title("Four-Way Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$d={d}$" for d in comp_dims])
    ax.legend(fontsize=7, frameon=True, ncol=2)
    ax.set_ylim(0, max(max(dgm_errs), max(hmc_errs)) * 1.35)
    save_fig(fig, f"{FIG_DIR}/fig2_4way_comparison")
    print("  [OK] fig2_4way_comparison")

    # ================================================================
    # FIG 3: Price comparison lines (all 4 methods per dimension)
    # ================================================================
    print("Fig 3: Price comparison lines...")
    n_dims = len(comp_dims)
    n_cols = min(3, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 2.8 * n_rows),
                             squeeze=False)

    for idx, d in enumerate(comp_dims):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        r = comp[str(d)]

        ax.plot(S0_TEST, r["mc_prices"], 'k-', label="MC ref.", linewidth=2.2, alpha=0.6, zorder=3)
        ax.plot(S0_TEST, r["dgm_prices"], 'o-', color=CB_BLUE, label="DGM",
                markersize=5, markerfacecolor='white', markeredgewidth=1.2, zorder=5)
        ax.plot(S0_TEST, r["zhou_prices"], 's--', color=CB_RED, label="Zhou",
                markersize=4, markerfacecolor='white', markeredgewidth=1.2, zorder=4)
        ax.plot(S0_TEST, r["cv_prices"], '^:', color=CB_GREEN, label="Hybrid MC",
                markersize=4, markerfacecolor='white', markeredgewidth=1.2, zorder=4)

        # MC 95% CI band
        mc_lo = [r["mc_prices"][i] - 1.96 * r["mc_errors"][i] for i in range(5)]
        mc_hi = [r["mc_prices"][i] + 1.96 * r["mc_errors"][i] for i in range(5)]
        ax.fill_between(S0_TEST, mc_lo, mc_hi, color='gray', alpha=0.15, label="MC 95% CI")

        ax.set_title(f"$d = {d}$", fontsize=10)
        ax.set_xlabel("$S_0 / K$", fontsize=8)
        if col == 0:
            ax.set_ylabel("Option Price", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=5.5, loc='upper left', frameon=True)

    for idx in range(n_dims, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Option Prices: DGM vs Zhou vs Hybrid MC vs MC Reference", fontsize=11, y=1.02)
    plt.tight_layout()
    save_fig(fig, f"{FIG_DIR}/fig3_price_comparison")
    print("  [OK] fig3_price_comparison")

    # ================================================================
    # FIG 4: Predicted vs True scatter
    # ================================================================
    print("Fig 4: Scatter plots...")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    methods = [
        ("DGM", "dgm_prices", CB_BLUE),
        ("Zhou et al.", "zhou_prices", CB_RED),
        ("Hybrid MC", "cv_prices", CB_GREEN),
    ]

    for ax_idx, (name, key, color) in enumerate(methods):
        ax = axes[ax_idx]
        for d in comp_dims:
            r = comp[str(d)]
            ax.scatter(r["mc_prices"], r[key], s=40, alpha=0.8,
                       label=f"$d={d}$", zorder=5, edgecolors='white', linewidth=0.5)
        lim = [0, 0.28]
        ax.plot(lim, lim, 'k--', alpha=0.4, linewidth=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("MC Reference")
        ax.set_ylabel(f"{name} Price")
        ax.set_title(f"{name} vs MC")
        if ax_idx == 0:
            ax.legend(fontsize=6, frameon=True, ncol=2)
        ax.set_aspect('equal')

    plt.tight_layout()
    save_fig(fig, f"{FIG_DIR}/fig4_scatter")
    print("  [OK] fig4_scatter")

    # ================================================================
    # FIG 5: Error by moneyness
    # ================================================================
    print("Fig 5: Error by moneyness...")
    moneyness_labels = ["OTM\n($S_0<K$)", "ATM\n($S_0=K$)", "ITM\n($S_0>K$)"]
    moneyness_idx = [[0, 1], [2], [3, 4]]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    all_methods = [
        ("DGM", "dgm_prices", CB_BLUE),
        ("Zhou", "zhou_prices", CB_RED),
        ("Hybrid MC", "cv_prices", CB_GREEN),
    ]
    n_methods = len(all_methods)
    width = 0.22
    x = np.arange(3)

    for m_idx, (name, key, color) in enumerate(all_methods):
        errs_by_money = []
        for group in moneyness_idx:
            errs = []
            for d in comp_dims:
                r = comp[str(d)]
                for i in group:
                    mc_p = r["mc_prices"][i]
                    if mc_p > 1e-8:
                        errs.append(abs(r[key][i] - mc_p) / mc_p)
            errs_by_money.append(np.mean(errs) * 100)

        offset = (m_idx - (n_methods-1)/2) * width
        bars = ax.bar(x + offset, errs_by_money, width, label=name, color=color, alpha=0.9)
        for bar, val in zip(bars, errs_by_money):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{val:.1f}%", ha='center', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(moneyness_labels)
    ax.set_ylabel("Mean Relative Error (%)")
    ax.set_title("Pricing Error by Moneyness Region")
    ax.legend(frameon=True, fontsize=7)
    save_fig(fig, f"{FIG_DIR}/fig5_error_by_moneyness")
    print("  [OK] fig5_error_by_moneyness")

    # ================================================================
    # FIG 6: Hybrid MC variance reduction
    # ================================================================
    if hmc:
        print("Fig 6: Hybrid MC analysis...")
        hmc_dims = sorted(hmc.keys(), key=int)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

        vanilla_se = [hmc[d]["vanilla_se"] for d in hmc_dims]
        cv_se = [hmc[d]["cv_se"] for d in hmc_dims]

        x = np.arange(len(hmc_dims))
        w = 0.3
        ax1.bar(x - w/2, vanilla_se, w, label="Vanilla MC", color=CB_RED, alpha=0.9)
        ax1.bar(x + w/2, cv_se, w, label="DGM-CV MC", color=CB_GREEN, alpha=0.9)
        ax1.set_yscale("log")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"$d={d}$" for d in hmc_dims])
        ax1.set_ylabel("Standard Error (log scale)")
        ax1.set_title("MC Standard Error Comparison")
        ax1.legend(fontsize=7, frameon=True)

        se_reduction = [hmc[d]["vanilla_se"] / hmc[d]["cv_se"] for d in hmc_dims]
        bars = ax2.bar([f"$d={d}$" for d in hmc_dims], se_reduction,
                       color=CB_CYAN, alpha=0.9, width=0.5)
        for bar, val in zip(bars, se_reduction):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     f"{val:.0f}x", ha="center", fontsize=8, fontweight='bold')
        ax2.set_ylabel("SE Reduction Factor")
        ax2.set_title("Variance Reduction Magnitude")

        plt.tight_layout()
        save_fig(fig, f"{FIG_DIR}/fig6_hybrid_mc")
        print("  [OK] fig6_hybrid_mc")

    # ================================================================
    # FIG 7: Ablation study
    # ================================================================
    if abl:
        print("Fig 7: Ablation study...")
        display_names = {
            "dgm_tanh": "DGM\n(baseline)",
            "mlp_tanh": "MLP\n(baseline)",
            "soft_constraint": "Soft\nconstraint",
            "uniform_sampling": "Uniform\nsampling",
            "risk_neutral_sampling": "Risk-neutral\nsampling",
            "adaptive_sampling": "Adaptive\nsampling",
        }
        abl_colors = [CB_BLUE, CB_RED, CB_GREEN, CB_ORANGE, CB_PURPLE, CB_CYAN]

        valid_names, errors, times = [], [], []
        for name in ["dgm_tanh", "mlp_tanh", "soft_constraint",
                      "uniform_sampling", "risk_neutral_sampling", "adaptive_sampling"]:
            if name in abl:
                entry = abl[name]
                err = entry.get("rel_l2_error")
                if err is not None and np.isfinite(err):
                    valid_names.append(name)
                    errors.append(err * 100)
                    t = entry.get("train_time_seconds") or entry.get("train_time", 0)
                    times.append(t / 3600)

        if valid_names:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

            labels = [display_names.get(n, n) for n in valid_names]
            bars = ax1.bar(labels, errors, color=abl_colors[:len(valid_names)],
                           alpha=0.9, width=0.6)
            for bar, err in zip(bars, errors):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                         f"{err:.2f}%", ha="center", va="bottom", fontsize=7)
            ax1.set_ylabel("Relative $L^2$ Error (%)")
            ax1.set_title("Ablation: Pricing Accuracy ($d=2$)")

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
    # FIG 8: Training convergence (DGM vs MLP)
    # ================================================================
    configs_with_history = {n: abl[n] for n in ["dgm_tanh", "mlp_tanh"]
                            if n in abl and "loss_history" in abl[n]}
    if configs_with_history:
        print("Fig 8: Training convergence...")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        style_map = {
            "dgm_tanh": {"color": CB_BLUE, "label": "DGM (tanh)", "ls": "-"},
            "mlp_tanh": {"color": CB_RED, "label": "MLP (tanh)", "ls": "--"},
        }
        for name, data in configs_with_history.items():
            pde_loss = data["loss_history"]["loss_pde"]
            steps = np.linspace(0, 50000, len(pde_loss))
            style = style_map.get(name, {"color": CB_BLACK, "label": name, "ls": "-"})
            window = max(1, len(pde_loss) // 50)
            if window > 1:
                kernel = np.ones(window) / window
                smoothed = np.convolve(pde_loss, kernel, mode='valid')
                steps_smooth = steps[:len(smoothed)]
            else:
                smoothed, steps_smooth = pde_loss, steps
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
    # FIG 9: Component-wise loss
    # ================================================================
    if abl and "dgm_tanh" in abl and "loss_history" in abl["dgm_tanh"]:
        print("Fig 9: Component-wise loss...")
        data = abl["dgm_tanh"]["loss_history"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        n = len(data["loss_pde"])
        steps = np.linspace(0, 50000, n)
        ax.semilogy(steps, data["loss_pde"], color=CB_BLUE,
                    label=r"PDE loss ($\mathcal{J}_{PDE}$)", linewidth=1.2, alpha=0.8)
        ax.semilogy(steps, data["loss_boundary"], color=CB_RED,
                    label=r"Boundary loss ($\mathcal{J}_{bnd}$)", linewidth=1.2, alpha=0.8)
        ax.semilogy(steps, data["loss_total"], color=CB_BLACK,
                    label="Total loss", linewidth=1.8, alpha=0.6, linestyle='--')
        ax.axvline(x=50000 * 0.3, color=CB_ORANGE, linestyle=':', alpha=0.6, linewidth=1)
        ax.text(50000 * 0.3 + 500, ax.get_ylim()[1] * 0.3, r"$\epsilon \to 0$",
                fontsize=8, color=CB_ORANGE)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Component-wise Loss Decomposition ($d=2$, DGM)")
        ax.legend(frameon=True, fontsize=7.5)
        ax.set_xlim(0, 50000)
        save_fig(fig, f"{FIG_DIR}/fig9_component_loss")
        print("  [OK] fig9_component_loss")

    # ================================================================
    # FIG 10: Parameter count vs dimension
    # ================================================================
    if scaling:
        print("Fig 10: Parameter efficiency...")
        s_dims = sorted(scaling.keys())
        params = [scaling[d].get("n_params", 0) for d in s_dims]
        s_errs = [scaling[d].get("rel_l2_error", 0) * 100 for d in s_dims]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

        # Left: params vs d
        ax1.bar([f"$d={d}$" for d in s_dims], [p/1e6 for p in params],
                color=CB_BLUE, alpha=0.9, width=0.5)
        for i, (d, p) in enumerate(zip(s_dims, params)):
            ax1.text(i, p/1e6 + 0.05, f"{p/1e6:.1f}M",
                     ha='center', fontsize=7.5, fontweight='bold')
        ax1.set_ylabel("Parameters (millions)")
        ax1.set_title("Model Size vs. Dimension")

        # Right: error vs params (efficiency)
        for i, d in enumerate(s_dims):
            ax2.scatter(params[d-1] if d-1 < len(params) else params[i],
                       s_errs[i], s=80, zorder=5, edgecolors='black', linewidth=0.8)
            ax2.annotate(f"$d={d}$", (params[i], s_errs[i]),
                        textcoords="offset points", xytext=(8, 3), fontsize=8)
        ax2.set_xlabel("Parameters")
        ax2.set_ylabel("Relative $L^2$ Error (%)")
        ax2.set_title("Parameter Efficiency")
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

        plt.tight_layout()
        save_fig(fig, f"{FIG_DIR}/fig10_parameters")
        print("  [OK] fig10_parameters")

    # ================================================================
    # FIG 11: Summary table
    # ================================================================
    print("Fig 11: Summary table...")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")

    cols = ["$d$", "DGM Err.", "Zhou Err.", "Hybrid MC Err.",
            "MC SE", "CV SE", "VR Factor"]
    cell_text = []
    for d in comp_dims:
        r = comp[str(d)]
        hmc_d = hmc.get(str(d), {})
        mc_p = np.array(r["mc_prices"])
        cv_p = np.array(r["cv_prices"])
        mask = mc_p > 1e-8
        hmc_err = np.mean(np.abs(cv_p[mask] - mc_p[mask]) / mc_p[mask]) * 100

        vr = hmc_d.get("vanilla_se", 0) / hmc_d.get("cv_se", 1) if hmc_d.get("cv_se", 0) > 0 else None
        vr_str = f"{vr:.0f}x" if vr else "--"

        cell_text.append([
            str(d),
            f"{r['dgm_mean_rel_error']*100:.2f}%",
            f"{r['zhou_mean_rel_error']*100:.2f}%",
            f"{hmc_err:.2f}%",
            f"{np.mean(r['mc_errors']):.2e}",
            f"{np.mean(r.get('cv_errors', [0])):.2e}",
            vr_str,
        ])

    table = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    for j in range(len(cols)):
        table[0, j].set_facecolor(CB_BLUE)
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell_text) + 1):
        for j in range(len(cols)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#F0F4F8")

    ax.set_title("Summary: Four-Way Comparison Across Dimensions", pad=25, fontsize=12)
    save_fig(fig, f"{FIG_DIR}/fig11_summary_table")
    print("  [OK] fig11_summary_table")

    # ================================================================
    # FIG 12: Detailed price tables per dimension
    # ================================================================
    print("Fig 12: Detailed price tables...")
    for d in comp_dims:
        r = comp[str(d)]
        n = len(r["S0_test"])

        fig, ax = plt.subplots(figsize=(9, max(2.5, 0.5 * n + 1.5)))
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

        ax.set_title(f"Detailed Prices ($d={d}$)", pad=20, fontsize=11)
        save_fig(fig, f"{FIG_DIR}/fig12_prices_d{d}")
        print(f"  [OK] fig12_prices_d{d}")

    # ================================================================
    # TEXT OUTPUT: Formatted tables for paper
    # ================================================================
    print("\n" + "=" * 80)
    print("FORMATTED TABLES FOR PAPER (copy-paste into LaTeX)")
    print("=" * 80)

    print("\n--- Table 1: Mean relative error vs MC reference ---\n")
    print(f"{'d':>4}  {'DGM (PDE)':>12}  {'Zhou (regr.)':>14}  {'Hybrid MC':>12}  {'MC SE':>10}")
    print("-" * 60)
    for d in comp_dims:
        r = comp[str(d)]
        mc_p = np.array(r["mc_prices"])
        cv_p = np.array(r["cv_prices"])
        mask = mc_p > 1e-8
        hmc_err = np.mean(np.abs(cv_p[mask] - mc_p[mask]) / mc_p[mask]) * 100
        mc_se = np.mean(r["mc_errors"])
        print(f"{d:>4}  {r['dgm_mean_rel_error']*100:>11.2f}%  "
              f"{r['zhou_mean_rel_error']*100:>13.2f}%  "
              f"{hmc_err:>11.2f}%  "
              f"{mc_se:>10.2e}")

    for d in comp_dims:
        r = comp[str(d)]
        print(f"\n--- Table: Detailed prices at d={d} ---\n")
        print(f"{'S0/K':>6}  {'DGM':>10}  {'Zhou NN':>10}  {'Hybrid MC':>10}  {'MC ref.':>10}")
        print("-" * 55)
        for i in range(len(r["S0_test"])):
            print(f"{r['S0_test'][i]:>6.2f}  {r['dgm_prices'][i]:>10.6f}  "
                  f"{r['zhou_prices'][i]:>10.6f}  {r['cv_prices'][i]:>10.6f}  "
                  f"{r['mc_prices'][i]:>10.6f}")

    n_figs = len([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])
    print(f"\n\n[OK] All {n_figs} figures generated in {FIG_DIR}/")
    print("     (Each saved as both PDF and PNG)")


if __name__ == "__main__":
    main()
