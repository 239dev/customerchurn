"""Five EDA charts, each built around one finding.

python -m src.eda
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .data import load_clean
from .viz import apply_style, title_block, caption, BLUE, ORANGE, INK, DIVERGING

FIG_DIR = "reports/figures"
apply_style()


def churn_rate_by(df, col):
    return (df.groupby(col)["Churn"].mean().sort_values(ascending=False) * 100)


def chart_contract(df):
    rates = churn_rate_by(df, "Contract")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.bar(rates.index, rates.values, color=BLUE, width=0.55)
    ax.set_ylabel("Churn rate (%)")
    ratio_1yr = rates.iloc[0] / rates.iloc[1]
    ratio_2yr = rates.iloc[0] / rates.iloc[2]
    title_block(
        ax,
        f"Month-to-month customers churn ~{ratio_1yr:.0f}x more than 1-year,\n"
        f"~{ratio_2yr:.0f}x more than 2-year contracts",
        subtitle="Churn rate by contract length, all 7,043 customers",
    )
    for i, v in enumerate(rates.values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(rates.values) * 1.25)
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_contract.png", dpi=150, bbox_inches="tight")
    plt.close()
    return rates


def chart_tenure(df):
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]
    )
    rates = df.groupby("tenure_bucket", observed=True)["Churn"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.bar(rates.index.astype(str), rates.values, color=BLUE, width=0.55)
    ax.set_ylabel("Churn rate (%)")
    title_block(
        ax,
        f"A customer's risk of leaving falls from {rates.iloc[0]:.0f}% to "
        f"{rates.iloc[-1]:.0f}% the longer they stay",
        subtitle="Churn rate by tenure, all 7,043 customers -- risk is front-loaded into the first year",
    )
    for i, v in enumerate(rates.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("Time as a customer")
    ax.set_ylim(0, max(rates.values) * 1.25)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_tenure.png", dpi=150, bbox_inches="tight")
    plt.close()
    return rates


def chart_internet(df):
    rates = churn_rate_by(df, "InternetService")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.bar(rates.index, rates.values, color=BLUE, width=0.5)
    ax.set_ylabel("Churn rate (%)")
    title_block(
        ax,
        "Fiber optic customers churn more than DSL customers,\ndespite paying more per month",
        subtitle="Churn rate by internet service type -- inconsistent with a price-driven explanation",
    )
    for i, v in enumerate(rates.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylim(0, max(rates.values) * 1.25)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_internet.png", dpi=150, bbox_inches="tight")
    plt.close()
    return rates


def chart_monthly_charges(df):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for val, color, label in [(0, BLUE, "Stayed"), (1, ORANGE, "Churned")]:
        sns.kdeplot(df.loc[df["Churn"] == val, "MonthlyCharges"], ax=ax,
                    color=color, fill=True, alpha=0.25, linewidth=2, label=label)
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Share of customers")
    ax.set_yticklabels([])
    title_block(
        ax,
        "Customers who churn tend to pay more per month",
        subtitle="Distribution of monthly charges, split by outcome -- a marginal view, isolated further via SHAP",
    )
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/monthly_charges_by_churn.png", dpi=150, bbox_inches="tight")
    plt.close()


def chart_correlation(df):
    numeric = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].rename(
        columns={"tenure": "Tenure", "MonthlyCharges": "Monthly\nCharges",
                 "TotalCharges": "Total\nCharges", "Churn": "Churn"}
    )
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cmap = sns.blend_palette(DIVERGING, as_cmap=True)  # diverging, not sequential -- values go negative
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax, cbar=True,
                linewidths=2, linecolor="#fcfcfb", vmin=-1, vmax=1, center=0,
                annot_kws={"color": INK, "fontsize": 10})
    title_block(
        ax,
        "Tenure and total charges move together closely (r=0.83)",
        subtitle="Correlation between numeric fields -- expected, since total charges accrue over tenure",
    )
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    return corr


def main():
    df = load_clean()
    print("Churn by contract:")
    print(chart_contract(df))
    print("\nChurn by tenure bucket:")
    print(chart_tenure(df))
    print("\nChurn by internet service:")
    print(chart_internet(df))
    chart_monthly_charges(df)
    print("\nCorrelation:")
    print(chart_correlation(df))
    print(f"\nSaved 5 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
