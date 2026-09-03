"""Checks whether the model's predicted probabilities can be trusted, and if
not, fixes them -- the whole threshold optimization in threshold.py assumes
a customer scored at 0.30 really does churn 30% of the time.

python -m src.calibration
"""
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from .data import load_clean
from .model import split
from .evaluate import threshold_sweep
from .viz import apply_style, title_block, caption, BLUE, ORANGE, MUTED

apply_style()


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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.5, label="Perfect calibration")
    ax.plot(prob_pred_u, prob_true_u, "o-", color=ORANGE, lw=2, markersize=6,
             label=f"Before calibration (Brier {brier_uncal:.3f})")
    ax.plot(prob_pred_c, prob_true_c, "o-", color=BLUE, lw=2, markersize=6,
             label=f"After calibration (Brier {brier_cal:.3f})")
    ax.set_xlabel("Model's predicted churn probability")
    ax.set_ylabel("How often those customers actually churned")
    title_block(
        ax,
        "The raw model overstates confidence -- calibration corrects it",
        subtitle="Customers the model scored around 0.8 actually churned less often than that, before this fix",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    plt.tight_layout()
    caption(fig, "A model is 'calibrated' if, among customers it scores at 80% churn risk, "
                 "about 80% actually churn. Dots above the dotted line mean the model wasn't confident enough; "
                 "below it means overconfident.")
    plt.savefig("reports/figures/calibration.png", dpi=150, bbox_inches="tight")
    plt.close()

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

    # this is the artifact that actually gets deployed -- calibrated
    # probabilities + the threshold recomputed against them
    joblib.dump({"model": calibrated, "threshold": float(opt_cal["threshold"])},
                "models/churn_model.joblib")
    print("\nSaved calibration.png, calibration_summary.json, and models/churn_model.joblib")


if __name__ == "__main__":
    main()
