import json
import re

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

# Update "Three-Way Method Comparison" to "Four-Way Method Comparison"
readme = readme.replace("### Three-Way Method Comparison (DGM vs Zhou vs MC)", "### Four-Way Method Comparison (DGM vs Zhou vs Hybrid MC vs MC)")

# Replace the first image in comparison section with the new 4-way comparison image
readme = re.sub(r'src="figures/fig2_dgm_vs_zhou\.png"', 'src="figures/publication/fig2_4way_comparison.png"', readme)
readme = readme.replace("Mean relative error vs MC: DGM (PDE) vs Zhou NN (regression) across dimensions.", "Mean relative error vs MC: Four-way comparison across dimensions.")

# Replace the scatter plot and price surface
readme = re.sub(r'src="figures/zhou_fig1_price_surface_2d\.png" width="450"', 'src="figures/publication/fig3_price_comparison.png" width="800"', readme)
readme = re.sub(r'<img src="figures/zhou_fig5_scatter_dgm_vs_mc\.png" width="350" alt="DGM vs MC scatter"/>\s*</p>\s*<p align="center"><em>Left: DGM 2D price surface at t=0. Right: DGM vs MC scatter plot \(5 test points\)\.</em></p>', '</p>\n<p align="center"><em>Price comparison across dimensions.</em></p>', readme)

readme = re.sub(r'<p align="center">\s*<img src="figures/zhou_fig2_training_loss\.png" width="450" alt="Zhou training loss"/>\s*<img src="figures/zhou_fig3_price_table\.png" width="350" alt="Price comparison table"/>\s*</p>\s*<p align="center"><em>Left: Training loss convergence over 200k steps \(note log scale\)\. Right: Price comparison table\.</em></p>', 
'<p align="center">\n  <img src="figures/publication/fig4_scatter.png" width="800" alt="Scatter plots"/>\n</p>\n<p align="center"><em>Scatter plots: Predicted vs True MC Reference.</em></p>', readme)

# Update Scaling Error image
readme = re.sub(r'src="figures/fig1_scaling_error\.png"', 'src="figures/publication/fig1_scaling_error.png"', readme)

# Update Ablation image
readme = re.sub(r'src="figures/fig4_ablation\.png"', 'src="figures/publication/fig7_ablation.png"', readme)

# Update Hybrid MC image
readme = re.sub(r'src="figures/fig3_hybrid_mc\.png"', 'src="figures/publication/fig6_hybrid_mc.png"', readme)

# Also add the new Parameter Efficiency and Error by Moneyness
readme = readme.replace("### Dimensional Scaling Study", "### Error by Moneyness\n\n<p align=\"center\">\n  <img src=\"figures/publication/fig5_error_by_moneyness.png\" width=\"500\" alt=\"Error by moneyness\"/>\n</p>\n<p align=\"center\"><em>Pricing Error by Moneyness Region.</em></p>\n\n### Dimensional Scaling Study")
readme = readme.replace("### Ablation Study", "### Parameter Efficiency\n\n<p align=\"center\">\n  <img src=\"figures/publication/fig10_parameters.png\" width=\"700\" alt=\"Parameter Efficiency\"/>\n</p>\n<p align=\"center\"><em>Model Size and Parameter Efficiency vs. Dimension.</em></p>\n\n### Ablation Study")

# Update Scaling Table
new_scaling_table = """| $d$ | Relative $L^2$ error | Parameters | Hessian method |
|:-:|:-:|:-:|:-:|
| 1 | **0.16%** | 1,061,889 | Exact |
| 2 | **3.20%** | 1,066,241 | Exact |
| 3 | **5.03%** | 1,070,593 | Exact |
| 5 | **11.23%** | 1,079,297 | Exact |
| 7 | **13.14%** | 4,273,153 | Hutchinson |
| 10 | **28.47%** | 4,299,265 | Hutchinson |"""

readme = re.sub(r'\| \$d\$ \| Relative \$L\^2\$ error \| Parameters \| Hessian method \|\n\|:-:\|:-:\|:-:\|:-:\|\n(?:\| .*? \|\n)*', new_scaling_table + "\n", readme)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("README updated successfully!")
