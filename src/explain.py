"""Day 8-9: SHAP explainability for the tuned XGBoost model.

Usage: python -m src.explain
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from .viz import apply_style, title_block, caption, humanize_all, humanize_feature_name, BLUE, ORANGE, INK, SURFACE

FIG_DIR = "reports/figures"
apply_style()


def main():
    # SHAP's TreeExplainer needs direct access to the XGBoost booster, so this
    # explains the raw (uncalibrated) model. Calibration reshapes probabilities,
    # not feature rankings, so the explanation still holds for the deployed
    # (calibrated) model in models/churn_model.joblib.
    artifact = joblib.load("models/churn_model_xgb_raw.joblib")
    model = artifact["model"]
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    X_test = pd.read_csv("data/processed/X_test.csv")
    X_test_t = prep.transform(X_test)
    raw_names = prep.get_feature_names_out()
    feature_names = humanize_all(raw_names)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_t)

    # --- Bar chart: mean absolute impact, ranked ---
    shap.summary_plot(shap_values, X_test_t, feature_names=feature_names,
                       plot_type="bar", show=False, max_display=12, color=BLUE)
    fig = plt.gcf()
    fig.set_facecolor(SURFACE)
    ax = plt.gca()
    ax.set_facecolor(SURFACE)
    title_block(
        ax,
        "Contract type and tenure are the two biggest drivers of churn risk",
        subtitle="Top 12 factors, ranked by average influence on the model's prediction",
    )
    ax.set_xlabel("Average influence on predicted churn risk (mean |SHAP value|)")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    caption(fig, "Longer bar = this factor moves the model's prediction further, on average, across all customers.")
    plt.savefig(f"{FIG_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Beeswarm: direction + magnitude together ---
    shap.summary_plot(shap_values, X_test_t, feature_names=feature_names, show=False, max_display=12)
    fig = plt.gcf()
    fig.set_facecolor(SURFACE)
    ax = plt.gca()
    ax.set_facecolor(SURFACE)
    title_block(
        ax,
        "Being on a month-to-month contract or new to the service\npushes predictions toward churn",
        subtitle="Each dot is one customer. Color = that customer's value for the factor; position = effect on the prediction",
    )
    ax.set_xlabel("<- pushes prediction toward staying   |   pushes prediction toward churn ->")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    caption(fig, "Red = a high value for that factor (e.g. long tenure, high charges); blue = a low value. "
                 "Dots to the right of center increase predicted churn risk; dots to the left decrease it.")
    plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Rank features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = pd.Series(mean_abs, index=raw_names).sort_values(ascending=False)
    ranking_readable = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    print("=== Top 10 features by mean |SHAP value| ===")
    print(ranking_readable.head(10))
    ranking.to_csv("reports/shap_ranking.csv")
    ranking_readable.to_csv("reports/shap_ranking_readable.csv")

    # --- Dependence plot: tenure ---
    # Built manually (not shap.dependence_plot) so the x-axis shows real
    # months, not the standardized z-scores the model actually trains on --
    # a z-score axis ("-1.3" to "1.6" for "tenure") is unreadable to anyone
    # without the scaler in their head.
    tenure_idx = list(raw_names).index("num__tenure")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(X_test["tenure"], shap_values[:, tenure_idx], color=BLUE, alpha=0.35, s=18, edgecolors="none")
    title_block(
        ax,
        "Churn risk drops sharply over a customer's first year,\nthen levels off",
        subtitle="Each dot is one customer: their actual tenure vs. that tenure's effect on their churn prediction",
    )
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Effect on predicted churn risk")
    ax.axhline(0, color="#c3c2b7", lw=1)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_tenure.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Dependence plot: month-to-month contract ---
    contract_col = "cat__Contract_Month-to-month"
    if contract_col in raw_names:
        c_idx = list(raw_names).index(contract_col)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        shap.dependence_plot(c_idx, shap_values, X_test_t, feature_names=feature_names,
                              ax=ax, show=False, interaction_index=None, color=ORANGE)
        title_block(
            ax,
            "Simply being on a month-to-month contract is the single\nlargest churn risk factor in the model",
            subtitle="0 = not on a month-to-month contract, 1 = on a month-to-month contract",
        )
        ax.set_ylabel("Effect on predicted churn risk")
        ax.set_xticks([0, 1])
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/shap_contract.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Single-customer waterfall: highest-risk customer in test set ---
    proba = np.load("data/processed/proba_xgb.npy")
    top_idx = int(np.argmax(proba))

    # Swap in the customer's real, human-scale values (72 months, not the
    # standardized "-1.282" the model sees internally) for the numeric
    # features only -- the one-hot categorical columns are already 0/1 and
    # need no translation.
    display_row = X_test_t[top_idx].copy()
    for i, name in enumerate(raw_names):
        if name.startswith("num__"):
            col = name[len("num__"):]
            display_row[i] = X_test.iloc[top_idx][col]

    expl = shap.Explanation(
        values=shap_values[top_idx], base_values=explainer.expected_value,
        data=display_row, feature_names=feature_names,
    )
    fig = plt.figure(figsize=(8, 5.5))
    shap.plots.waterfall(expl, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_facecolor(SURFACE)
    fig.suptitle(
        f"Why the model flags this customer as {proba[top_idx]*100:.0f}% likely to churn",
        fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.01,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    caption(fig, "Starting from the average predicted risk across all customers (bottom), each bar shows how much "
                 "one factor about this specific customer pushed the prediction up (red) or down (blue) to reach "
                 "their final score (top).")
    plt.savefig(f"{FIG_DIR}/shap_force_single_customer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nHighest-risk test customer: index {top_idx}, predicted churn prob = {proba[top_idx]:.3f}")

    with open("reports/shap_top_features.json", "w") as f:
        json.dump(ranking_readable.head(10).to_dict(), f, indent=2)

    print(f"\nSaved SHAP figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
