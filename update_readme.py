
import json
import numpy as np
import re

with open('results/paper/comparison_summary.json', 'r', encoding='utf-8') as f:
    d_data = json.load(f)

table_str = '**Mean relative error vs MC reference:**\n\n'
table_str += '| $ | **DGM (PDE)** | **Zhou NN (regression)** | **Hybrid MC** |\n'
table_str += '|:-:|:-:|:-:|:-:|\n'
for d in ['1', '2', '3', '5']:
    v = d_data[d]
    mc = np.array(v['mc_prices'])
    dgm = np.array(v['dgm_prices'])
    zhou = np.array(v['zhou_prices'])
    cv = np.array(v['cv_prices'])
    err_dgm = np.mean(np.abs(dgm - mc) / mc)
    err_zhou = np.mean(np.abs(zhou - mc) / mc)
    err_cv = np.mean(np.abs(cv - mc) / mc)
    table_str += f'| {d} | {err_dgm:.2%} | {err_zhou:.2%} | **{err_cv:.2%}** |\n'

for d in ['1', '2', '3', '5']:
    table_str += f'\n**Detailed prices at d={d}:**\n\n'
    table_str += '| $ | **DGM** | **Zhou NN** | **Hybrid MC** | **MC ref.** |\n'
    table_str += '|:-:|:-:|:-:|:-:|:-:|\n'
    S0_test_values = [0.8, 0.9, 1.0, 1.1, 1.2]
    v = d_data[d]
    mc = v['mc_prices']
    dgm = v['dgm_prices']
    zhou = v['zhou_prices']
    cv = v['cv_prices']
    for i in range(5):
        table_str += f'| {S0_test_values[i]:.2f} | {dgm[i]:.6f} | {zhou[i]:.6f} | {cv[i]:.6f} | {mc[i]:.6f} |\n'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

pattern = re.compile(r'\*\*Mean relative error vs MC reference:\*\*.*?\n\n\*\*Analysis\*\*:', re.DOTALL)
readme = pattern.sub(table_str + '\n**Analysis**:', readme)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)
print('Updated README.md')

