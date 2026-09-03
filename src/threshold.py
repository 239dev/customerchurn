"""Day 10-11: cost-based threshold optimization, plot, and sensitivity analysis.

Usage: python -m src.threshold
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from .evaluate import expected_cost, threshold_sweep, analytical_optimal_threshold

BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": "#52514e", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.grid": True, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_cost_curve(sweep, opt_threshold, out_path="reports/figures/threshold_cost_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep["threshold"], sweep["cost"] / 1000, lw=2.5, color=BLUE)
    ax.axvline(opt_threshold, color=RED, ls="--", lw=1.5,
               label=f"Empirical optimum = {opt_threshold:.2f}")
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.5, label="Default = 0.50")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Expected total cost ($000s)")
    ax.set_title("Retention cost is minimized well below the default threshold", fontsize=11, loc="left")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def sensitivity_analysis(y_test, proba):
    results = []
    for eff in [0.15, 0.30, 0.50]:
        for ltv in [800, 1400, 2200]:
            s = threshold_sweep(y_test, proba, ltv=ltv, eff=eff)
            best_t = s.loc[s["cost"].idxmin(), "threshold"]
            results.append({"effectiveness": eff, "ltv": ltv, "optimal_threshold": round(best_t, 3)})
    sens = pd.DataFrame(results).pivot(index="effectiveness", columns="ltv", values="optimal_threshold")
    return sens


def plot_sensitivity(sens, out_path="reports/figures/threshold_sensitivity.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    seq = ["#cde2fb", "#6da7ec", "#184f95"]
    for i, ltv in enumerate(sens.columns):
        ax.plot(sens.index, sens[ltv], marker="o", color=seq[i], lw=2, label=f"LTV = ${ltv}")
    ax.set_xlabel("Offer effectiveness")
    ax.set_ylabel("Optimal threshold")
    ax.set_title("Optimal threshold stays well under 0.5 across every scenario", fontsize=11, loc="left")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    proba = np.load("data/processed/proba_xgb.npy")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    analytical_t = analytical_optimal_threshold()
    sweep = threshold_sweep(y_test, proba)
    opt = sweep.loc[sweep["cost"].idxmin()]
    default_row = sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]

    print(f"Analytical threshold: {analytical_t:.4f}")
    print(f"Empirical threshold : {opt['threshold']:.4f}  cost=${opt['cost']:,.0f}")
    print(f"Default (0.50)      : cost=${default_row['cost']:,.0f}")

    plot_cost_curve(sweep, opt["threshold"])

    sens = sensitivity_analysis(y_test, proba)
    print("\nSensitivity (rows=effectiveness, cols=LTV):")
    print(sens)
    plot_sensitivity(sens)
    sens.to_csv("reports/sensitivity.csv")

    print("\nSaved threshold_cost_curve.png and threshold_sensitivity.png")


if __name__ == "__main__":
    main()
