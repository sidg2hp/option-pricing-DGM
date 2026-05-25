import json
with open('results/paper/comparison_summary.json', 'r') as f:
    d = json.load(f)
print('DGM:', d['3']['dgm_prices'])
print('CV:', d['3']['cv_prices'])
print('MC:', d['3']['mc_prices'])
