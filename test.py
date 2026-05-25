import json
with open('results/paper/comparison_summary.json', 'r') as f:
    d = json.load(f)
for k in d:
    print(f"d={k}: CV={d[k]['cv_prices']}")
