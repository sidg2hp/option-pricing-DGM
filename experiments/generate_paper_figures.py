"""Generate all publication-quality figures from experiment results.

Run this after all experiments are complete to produce figures/ directory content.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting.style import apply_style, save_figure


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    apply_style()
    os.makedirs("figures", exist_ok=True)

    # ================================================================
    # Fig 1: Scaling bar chart (DGM error vs dimension)
    # ================================================================
    scaling_dims = []
    scaling_errors = []
    for d in [1, 2, 3, 5]:
        path = f"results/scaling/d_{d}/result.json"
        if os.path.exists(path):
            r = load_json(path)
            scaling_dims.append(d)
            scaling_errors.append(r["rel_l2_error"] * 100)

    if scaling_dims:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        bars = ax.bar([str(d) for d in scaling_dims], scaling_errors,
                      color=["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71"], alpha=0.85, width=0.55)
        for bar, err in zip(bars, scaling_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{err:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xlabel("Number of Assets $d$")
        ax.set_ylabel("Relative $L^2$ Error (%)")
        ax.set_title("DGM Pricing Error vs Dimension")
        save_figure(fig, "figures/fig1_scaling_error")
        print("  ✓ fig1_scaling_error")

    # ================================================================
    # Fig 2: DGM vs Zhou comparison (grouped bar)
    # ================================================================
    comp_dims = []
    dgm_errs = []
    zhou_errs = []
    for d in [1, 2, 3, 5]:
        path = f"results/paper/comparison/d_{d}/comparison_results.json"
        if os.path.exists(path):
            r = load_json(path)
            comp_dims.append(d)
            dgm_errs.append(r["dgm_mean_rel_error"] * 100)
            zhou_errs.append(r["zhou_mean_rel_error"] * 100)

    if comp_dims:
        x = np.arange(len(comp_dims))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 3.8))
        b1 = ax.bar(x - width/2, dgm_errs, width, label="DGM (PDE solver)",
                     color="#2C3E50", alpha=0.85)
        b2 = ax.bar(x + width/2, zhou_errs, width, label="Zhou NN (MC regression)",
                     color="#E74C3C", alpha=0.85)
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("Number of Assets $d$")
        ax.set_ylabel("Mean Relative Error vs MC (%)")
        ax.set_title("DGM vs Zhou et al.: Pricing Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in comp_dims])
        ax.legend(fontsize=8)
        save_figure(fig, "figures/fig2_dgm_vs_zhou")
        print("  ✓ fig2_dgm_vs_zhou")

    # ================================================================
    # Fig 3: Variance reduction vs dimension (hybrid MC)
    # ================================================================
    hmc_path = "results/hybrid_mc/hybrid_mc_summary.json"
    if os.path.exists(hmc_path):
        hmc = load_json(hmc_path)
        hmc_dims = sorted(hmc.keys(), key=int)
        vr_vals = [hmc[d]["variance_reduction"] * 100 for d in hmc_dims]
        se_reductions = []
        for d in hmc_dims:
            r = hmc[d]
            se_reductions.append(r["vanilla_se"] / r["cv_se"] if r["cv_se"] > 0 else 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

        # Left: VR percentage
        ax1.bar([str(d) for d in hmc_dims], vr_vals,
                color="#3498DB", alpha=0.85, width=0.5)
        ax1.set_ylim(99.999, 100.0005)
        ax1.set_xlabel("Number of Assets $d$")
        ax1.set_ylabel("Variance Reduction (%)")
        ax1.set_title("Hybrid DGM-MC: Variance Reduction")

        # Right: SE reduction factor
        ax2.bar([str(d) for d in hmc_dims],
                [r["vanilla_se"] for r in [hmc[d] for d in hmc_dims]],
                width=0.35, label="Vanilla MC", color="#E74C3C", alpha=0.85)
        ax2.bar([str(d) for d in hmc_dims],
                [r["cv_se"] for r in [hmc[d] for d in hmc_dims]],
                width=0.35, label="DGM-CV MC", color="#2ECC71", alpha=0.85)
        ax2.set_yscale("log")
        ax2.set_xlabel("Number of Assets $d$")
        ax2.set_ylabel("Standard Error (log scale)")
        ax2.set_title("Standard Error Comparison")
        ax2.legend(fontsize=8)

        plt.tight_layout()
        save_figure(fig, "figures/fig3_hybrid_mc")
        print("  ✓ fig3_hybrid_mc")

    # ================================================================
    # Fig 4: Ablation bar chart
    # ================================================================
    ablation_data = {}
    for name in ["dgm_tanh", "mlp_tanh", "dgm_softplus",
                  "soft_constraint", "uniform_sampling",
                  "risk_neutral_sampling", "adaptive_sampling"]:
        for p in [f"results/ablations/{name}/ablation_result.json",
                  f"results/ablations/{name}/ablation_{name}/results.json"]:
            if os.path.exists(p):
                ablation_data[name] = load_json(p)
                break

    if ablation_data:
        display = {
            "dgm_tanh": "DGM\n(tanh)",
            "mlp_tanh": "MLP\n(tanh)",
            "soft_constraint": "Soft\nconstraint",
            "uniform_sampling": "Uniform\nsampling",
            "risk_neutral_sampling": "Risk-neutral\nsampling",
            "adaptive_sampling": "Adaptive\nsampling",
        }
        colors = ["#2C3E50", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C"]

        # Filter valid results
        valid_names = [n for n in ablation_data
                       if ablation_data[n].get("rel_l2_error") is not None
                       and np.isfinite(ablation_data[n]["rel_l2_error"])]

        labels = [display.get(n, n) for n in valid_names]
        errors = [ablation_data[n]["rel_l2_error"] * 100 for n in valid_names]

        fig, ax = plt.subplots(figsize=(7.5, 4))
        bars = ax.bar(labels, errors, color=colors[:len(valid_names)], alpha=0.85, width=0.65)
        for bar, err in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{err:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Relative $L^2$ Error (%)")
        ax.set_title("Ablation Study ($d=2$, Basket Call, 50k steps)")
        save_figure(fig, "figures/fig4_ablation")
        print("  ✓ fig4_ablation")

    # ================================================================
    # Fig 5: Per-dimension price comparison tables
    # ================================================================
    for d in [1, 2, 3, 5]:
        path = f"results/paper/comparison/d_{d}/comparison_results.json"
        if not os.path.exists(path):
            continue
        r = load_json(path)
        n = len(r["S0_test"])

        fig, ax = plt.subplots(figsize=(7, max(2.2, 0.5 * n + 1.5)))
        ax.axis("off")

        cols = ["$S_0$", "DGM", "Zhou NN", "MC ref.", "MC SE"]
        cell_text = []
        for i in range(n):
            cell_text.append([
                f"{r['S0_test'][i]:.2f}",
                f"{r['dgm_prices'][i]:.6f}",
                f"{r['zhou_prices'][i]:.6f}",
                f"{r['mc_prices'][i]:.6f}",
                f"{r['mc_errors'][i]:.2e}",
            ])

        table = ax.table(cellText=cell_text, colLabels=cols,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Color header
        for j in range(len(cols)):
            table[0, j].set_facecolor("#2C3E50")
            table[0, j].set_text_props(color="white", fontweight="bold")

        ax.set_title(f"Price Comparison ($d={d}$)", pad=20, fontsize=12)
        save_figure(fig, f"figures/fig5_price_table_d{d}")
        print(f"  ✓ fig5_price_table_d{d}")

    # ================================================================
    # Fig 6: Comprehensive summary table
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    cols = ["$d$", "DGM Error", "Zhou Error", "Hybrid MC VR", "DGM Params"]
    cell_text = []
    for d in [1, 2, 3, 5]:
        comp_path = f"results/paper/comparison/d_{d}/comparison_results.json"
        scale_path = f"results/scaling/d_{d}/result.json"
        hmc_d = hmc.get(str(d), {}) if os.path.exists(hmc_path) else {}

        comp = load_json(comp_path) if os.path.exists(comp_path) else {}
        scale = load_json(scale_path) if os.path.exists(scale_path) else {}

        cell_text.append([
            str(d),
            f"{comp.get('dgm_mean_rel_error', 0)*100:.2f}%",
            f"{comp.get('zhou_mean_rel_error', 0)*100:.2f}%",
            f"{hmc_d.get('variance_reduction', 0)*100:.4f}%",
            f"{scale.get('n_params', 'N/A'):,}" if isinstance(scale.get('n_params'), int) else "N/A",
        ])

    table = ax.table(cellText=cell_text, colLabels=cols,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.6)
    for j in range(len(cols)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary of Results Across Dimensions", pad=20, fontsize=12)
    save_figure(fig, "figures/fig6_summary_table")
    print("  ✓ fig6_summary_table")

    print("\n✓ All figures generated in figures/")


if __name__ == "__main__":
    main()
