import json
import os

def print_tables():
    file_path = "results/paper/comparison_summary.json"
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    # First print the mean relative error summary
    print("Mean relative error vs MC reference:\n")
    print("d\tDGM (PDE)\tZhou NN (regression)")
    
    # Sort dims to print in order
    dims = sorted([int(k) for k in data.keys()])
    
    for d in dims:
        d_str = str(d)
        dgm_err = data[d_str]["metrics"]["dgm_mean_rel_err"] * 100
        zhou_err = data[d_str]["metrics"]["zhou_mean_rel_err"] * 100
        print(f"{d}\t{dgm_err:.2f}%\t{zhou_err:.2f}%")

    print("")
    
    # Print detailed tables for each dimension
    S0_values = [0.8, 0.9, 1.0, 1.1, 1.2]
    for d in dims:
        d_str = str(d)
        print(f"Detailed prices at d={d}:\n")
        print("S0\tDGM\tZhou NN\tHybrid MC\tMC ref.")
        
        dgm_prices = data[d_str]["prices"]["dgm"]
        zhou_prices = data[d_str]["prices"]["zhou"]
        cv_prices = data[d_str]["prices"]["cv_prices"]
        mc_prices = data[d_str]["prices"]["mc"]
        
        for i, s0 in enumerate(S0_values):
            print(f"{s0:.2f}\t{dgm_prices[i]:.6f}\t{zhou_prices[i]:.6f}\t{cv_prices[i]:.6f}\t{mc_prices[i]:.6f}")
        print("")

if __name__ == "__main__":
    print_tables()
