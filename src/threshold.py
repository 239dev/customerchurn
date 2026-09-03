"""Day 10-11: cost-based threshold optimization, plot, and sensitivity analysis.

Usage: python -m src.threshold
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .evaluate import threshold_sweep, analytical_optimal_threshold
from .viz import apply_style, title_block, caption, BLUE, RED, MUTED, INK

apply_style()


def plot_cost_curve(sweep, opt_threshold, opt_cost, default_cost,
                     out_path="reports/figures/threshold_cost_curve.png"):
    savings = default_cost - opt_cost
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sweep["threshold"], sweep["cost"] / 1000, lw=2.5, color=BLUE)

    ax.axvline(opt_threshold, color=RED, ls="--", lw=1.3, zorder=1)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.3, zorder=1)

    # Direct callouts instead of relying on a legend to connect lines to meaning.
    ax.annotate(
        f"Optimal: {opt_threshold:.2f}\n(${opt_cost/1000:,.0f}k)",
        xy=(opt_threshold, opt_cost / 1000), xytext=(opt_threshold + 0.10, opt_cost / 1000 - 8),
        fontsize=9.5, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=RED, lw=1),
    )
    ax.annotate(
        f"Default: 0.50\n(${default_cost/1000:,.0f}k)",
        xy=(0.5, default_cost / 1000), xytext=(0.56, default_cost / 1000 + 4),
        fontsize=9.5, color=MUTED,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
    )

    ax.set_xlabel("Classification threshold (act on customers scored above this)")
    ax.set_ylabel("Expected retention cost, $000s (lower is better)")
    title_block(
        ax,
        f"Acting at 0.50 costs ${savings/1000:,.0f}k more than necessary on this test set",
        subtitle="Total expected cost of missed churners + wasted offers, at every possible action threshold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
    fig, ax = plt.subplots(figsize=(7, 4.6))
    seq = ["#cde2fb", "#6da7ec", "#184f95"]
    for i, ltv in enumerate(sens.columns):
        ax.plot(sens.index, sens[ltv], marker="o", markersize=7, color=seq[i], lw=2.2,
                label=f"Customer value: ${ltv}")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.text(sens.index.max(), 0.52, "Default threshold (0.50)", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xlabel("Offer effectiveness (share of would-be churners actually saved)")
    ax.set_ylabel("Cost-minimizing threshold")
    title_block(
        ax,
        "The right threshold is almost always well below the 0.50 default",
        subtitle="How the optimal threshold shifts as the retention-offer assumptions change",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_ylim(0, max(0.85, sens.values.max() * 1.15))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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

    plot_cost_curve(sweep, opt["threshold"], opt["cost"], default_row["cost"])

    sens = sensitivity_analysis(y_test, proba)
    print("\nSensitivity (rows=effectiveness, cols=LTV):")
    print(sens)
    plot_sensitivity(sens)
    sens.to_csv("reports/sensitivity.csv")

    print("\nSaved threshold_cost_curve.png and threshold_sensitivity.png")


if __name__ == "__main__":
    main()
