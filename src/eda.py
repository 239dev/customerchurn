"""Day 2-3 exploratory analysis: five charts, each with a written finding.

Usage: python -m src.eda
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np

from .data import load_clean

FIG_DIR = "reports/figures"

# --- palette (validated categorical + sequential ramp) ---
BLUE = "#2a78d6"     # slot 1 -- "No" / stay
ORANGE = "#eb6834"   # slot 2 -- "Yes" / churn
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": SECONDARY_INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def churn_rate_by(df, col):
    return (df.groupby(col)["Churn"].mean().sort_values(ascending=False) * 100)


def chart_contract(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    rates = churn_rate_by(df, "Contract")
    ax.bar(rates.index, rates.values, color=BLUE, width=0.55)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Month-to-month customers churn at ~4x the rate of one-year,\n~15x the rate of two-year contracts", fontsize=11, loc="left")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", color=INK, fontsize=10)
    ax.set_ylim(0, max(rates.values) * 1.2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_contract.png", dpi=150)
    plt.close()
    return rates


def chart_tenure(df):
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49-72mo"]
    )
    rates = df.groupby("tenure_bucket", observed=True)["Churn"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rates.index.astype(str), rates.values, color=BLUE, width=0.55)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn risk is concentrated in a customer's first six months", fontsize=11, loc="left")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", color=INK, fontsize=10)
    ax.set_ylim(0, max(rates.values) * 1.25)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_tenure.png", dpi=150)
    plt.close()
    return rates


def chart_internet(df):
    rates = churn_rate_by(df, "InternetService")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rates.index, rates.values, color=BLUE, width=0.5)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Fiber optic customers churn far more than DSL,\ndespite paying more per month", fontsize=11, loc="left")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", color=INK, fontsize=10)
    ax.set_ylim(0, max(rates.values) * 1.25)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_internet.png", dpi=150)
    plt.close()
    return rates


def chart_monthly_charges(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for val, color, label in [(0, BLUE, "Stayed"), (1, ORANGE, "Churned")]:
        sns.kdeplot(df.loc[df["Churn"] == val, "MonthlyCharges"], ax=ax,
                    color=color, fill=True, alpha=0.25, linewidth=2, label=label)
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Density")
    ax.set_title("Churners skew toward higher monthly charges", fontsize=11, loc="left")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/monthly_charges_by_churn.png", dpi=150)
    plt.close()


def chart_correlation(df):
    numeric = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = sns.color_palette(SEQ_BLUE, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax, cbar=True,
                linewidths=2, linecolor=SURFACE, vmin=-1, vmax=1,
                annot_kws={"color": INK})
    ax.set_title("tenure and TotalCharges are highly correlated (r=0.83)", fontsize=11, loc="left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    return corr


def main():
    df = load_clean()
    print("=== Churn by Contract ===")
    print(chart_contract(df))
    print("\n=== Churn by Tenure Bucket ===")
    print(chart_tenure(df))
    print("\n=== Churn by Internet Service ===")
    print(chart_internet(df))
    chart_monthly_charges(df)
    print("\n=== Correlation ===")
    print(chart_correlation(df))
    print(f"\nSaved 5 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
