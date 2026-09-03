"""Extension 1: probability calibration audit.

The cost-optimal threshold in src/threshold.py assumes predicted probabilities
are literal frequencies (a customer scored at 0.30 churns 30% of the time).
This audits that assumption and recomputes the threshold on calibrated
probabilities.

Usage: python -m src.calibration
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from .data import load_clean
from .model import split
from .evaluate import threshold_sweep

BLUE = "#2a78d6"
ORANGE = "#eb6834"
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


def main():
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)

    artifact = joblib.load("models/churn_model_xgb_raw.joblib")
    uncal_model = artifact["model"]
    proba_uncal = uncal_model.predict_proba(X_test)[:, 1]

    brier_uncal = brier_score_loss(y_test, proba_uncal)

    calibrated = CalibratedClassifierCV(uncal_model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    proba_cal = calibrated.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, proba_cal)

    print(f"Brier score, uncalibrated: {brier_uncal:.4f}")
    print(f"Brier score, isotonic:     {brier_cal:.4f}")

    prob_true_u, prob_pred_u = calibration_curve(y_test, proba_uncal, n_bins=10)
    prob_true_c, prob_pred_c = calibration_curve(y_test, proba_cal, n_bins=10)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.5, label="Perfectly calibrated")
    ax.plot(prob_pred_u, prob_true_u, "o-", color=ORANGE, lw=2, label="XGBoost (uncalibrated)")
    ax.plot(prob_pred_c, prob_true_c, "o-", color=BLUE, lw=2, label="XGBoost (isotonic)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn fraction")
    ax.set_title("Isotonic calibration pulls predictions toward the diagonal", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig("reports/figures/calibration.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Recompute cost-optimal threshold on calibrated probabilities
    sweep_uncal = threshold_sweep(y_test, proba_uncal)
    sweep_cal = threshold_sweep(y_test, proba_cal)
    opt_uncal = sweep_uncal.loc[sweep_uncal["cost"].idxmin()]
    opt_cal = sweep_cal.loc[sweep_cal["cost"].idxmin()]

    print(f"\nOptimal threshold, uncalibrated probabilities: {opt_uncal['threshold']:.4f} (cost ${opt_uncal['cost']:,.0f})")
    print(f"Optimal threshold, calibrated probabilities:   {opt_cal['threshold']:.4f} (cost ${opt_cal['cost']:,.0f})")

    summary = {
        "brier_uncalibrated": float(brier_uncal),
        "brier_calibrated": float(brier_cal),
        "optimal_threshold_uncalibrated": float(opt_uncal["threshold"]),
        "optimal_threshold_calibrated": float(opt_cal["threshold"]),
        "cost_uncalibrated": float(opt_uncal["cost"]),
        "cost_calibrated": float(opt_cal["cost"]),
    }
    with open("reports/calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # This is the production artifact: calibrated probabilities + the
    # threshold recomputed against them. models/churn_model_xgb_raw.joblib
    # (uncalibrated) is kept around only because SHAP's TreeExplainer needs
    # direct access to the underlying booster, which CalibratedClassifierCV wraps.
    joblib.dump({"model": calibrated, "threshold": float(opt_cal["threshold"])},
                "models/churn_model.joblib")
    print("\nSaved calibration.png, calibration_summary.json, and models/churn_model.joblib (production artifact)")


if __name__ == "__main__":
    main()
