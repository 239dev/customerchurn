"""Day 8-9: SHAP explainability for the tuned XGBoost model.

Usage: python -m src.explain
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

FIG_DIR = "reports/figures"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
})


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
    feature_names = prep.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_t)

    # Beeswarm
    shap.summary_plot(shap_values, X_test_t, feature_names=feature_names, show=False)
    plt.gcf().set_facecolor(SURFACE)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar -- mean absolute impact
    shap.summary_plot(shap_values, X_test_t, feature_names=feature_names,
                       plot_type="bar", show=False)
    plt.gcf().set_facecolor(SURFACE)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Rank features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    print("=== Top 10 features by mean |SHAP value| ===")
    print(ranking.head(10))
    ranking.to_csv("reports/shap_ranking.csv")

    # Dependence plot for tenure
    tenure_idx = list(feature_names).index("num__tenure")
    shap.dependence_plot(tenure_idx, shap_values, X_test_t,
                          feature_names=feature_names, show=False)
    plt.gcf().set_facecolor(SURFACE)
    plt.savefig(f"{FIG_DIR}/shap_tenure.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Dependence plot for contract (month-to-month one-hot col, if present)
    contract_cols = [c for c in feature_names if "Contract" in c]
    if contract_cols:
        c_idx = list(feature_names).index(contract_cols[0])
        shap.dependence_plot(c_idx, shap_values, X_test_t,
                              feature_names=feature_names, show=False)
        plt.gcf().set_facecolor(SURFACE)
        plt.savefig(f"{FIG_DIR}/shap_contract.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Single-customer force plot (highest-risk customer in test set)
    proba = np.load("data/processed/proba_xgb.npy")
    top_idx = int(np.argmax(proba))
    fig = shap.force_plot(
        explainer.expected_value, shap_values[top_idx], X_test_t[top_idx],
        feature_names=feature_names, matplotlib=True, show=False,
    )
    plt.gcf().set_facecolor(SURFACE)
    plt.savefig(f"{FIG_DIR}/shap_force_single_customer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nHighest-risk test customer: index {top_idx}, predicted churn prob = {proba[top_idx]:.3f}")

    with open("reports/shap_top_features.json", "w") as f:
        json.dump(ranking.head(10).to_dict(), f, indent=2)

    print(f"\nSaved SHAP figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
