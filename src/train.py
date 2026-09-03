"""Train and evaluate the churn model end to end.

Usage: python -m src.train
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from .data import load_clean
from .model import split, fit_logreg, fit_xgb, evaluate
from .evaluate import threshold_sweep, analytical_optimal_threshold, OFFER_COST, LTV, EFFECTIVENESS


def main():
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}")

    results = {}

    # --- Majority-class baseline ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_acc = (dummy_pred == y_test).mean()
    results["baseline"] = {"auc": 0.5, "recall_churn": 0.0, "precision_churn": None,
                            "accuracy": dummy_acc}
    print(f"\nMajority-class baseline accuracy: {dummy_acc:.4f} (catches 0 churners)")

    # --- Logistic regression ---
    logreg = fit_logreg(X_train, y_train)
    results["logreg"] = evaluate(logreg, X_test, y_test, name="Logistic Regression")

    # --- XGBoost (tuned) ---
    search = fit_xgb(X_train, y_train)
    print("\nBest XGBoost params:", search.best_params_)
    print("Best CV ROC-AUC:", round(search.best_score_, 4))
    best = search.best_estimator_
    results["xgb"] = evaluate(best, X_test, y_test, name="XGBoost (tuned)")
    results["xgb"]["cv_auc"] = search.best_score_

    # --- Comparison table ---
    table = pd.DataFrame([
        {"Model": "Predict-majority baseline", "CV ROC-AUC": 0.500,
         "Test ROC-AUC": 0.500, "Recall (churn)": 0.00, "Precision (churn)": None},
        {"Model": "Logistic regression", "CV ROC-AUC": None,
         "Test ROC-AUC": round(results["logreg"]["auc"], 3),
         "Recall (churn)": round(results["logreg"]["recall_churn"], 2),
         "Precision (churn)": round(results["logreg"]["precision_churn"], 2)},
        {"Model": "XGBoost (tuned)", "CV ROC-AUC": round(search.best_score_, 3),
         "Test ROC-AUC": round(results["xgb"]["auc"], 3),
         "Recall (churn)": round(results["xgb"]["recall_churn"], 2),
         "Precision (churn)": round(results["xgb"]["precision_churn"], 2)},
    ])
    print("\n=== Model comparison ===")
    print(table.to_string(index=False))
    table.to_csv("reports/model_comparison.csv", index=False)

    # --- Cost-optimal threshold (on XGBoost probabilities) ---
    proba_xgb = results["xgb"]["proba"]
    analytical_t = analytical_optimal_threshold()
    print(f"\nAnalytical optimal threshold: {analytical_t:.4f}")

    sweep = threshold_sweep(y_test, proba_xgb)
    opt = sweep.loc[sweep["cost"].idxmin()]
    default_row = sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]

    n_test = len(y_test)
    savings_per_customer = (default_row["cost"] - opt["cost"]) / n_test
    print(f"Empirical optimal threshold: {opt['threshold']:.4f}  (cost ${opt['cost']:,.0f})")
    print(f"Default threshold 0.50: cost ${default_row['cost']:,.0f}")
    print(f"Savings per customer: ${savings_per_customer:,.2f}")
    print(f"Annualized @ 50,000 customers: ${savings_per_customer * 50_000:,.0f}")

    summary = {
        "n_train": len(X_train), "n_test": len(X_test),
        "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean()),
        "logreg_auc": float(results["logreg"]["auc"]),
        "logreg_recall": float(results["logreg"]["recall_churn"]),
        "logreg_precision": float(results["logreg"]["precision_churn"]),
        "xgb_cv_auc": float(search.best_score_),
        "xgb_auc": float(results["xgb"]["auc"]),
        "xgb_recall": float(results["xgb"]["recall_churn"]),
        "xgb_precision": float(results["xgb"]["precision_churn"]),
        "xgb_best_params": search.best_params_,
        "analytical_threshold": float(analytical_t),
        "empirical_optimal_threshold": float(opt["threshold"]),
        "empirical_optimal_cost": float(opt["cost"]),
        "default_threshold_cost": float(default_row["cost"]),
        "savings_per_customer": float(savings_per_customer),
        "savings_annualized_50k": float(savings_per_customer * 50_000),
        "cost_params": {"OFFER_COST": OFFER_COST, "LTV": LTV, "EFFECTIVENESS": EFFECTIVENESS},
    }
    with open("reports/train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Saved as the "raw" artifact -- src/calibration.py audits this model's
    # probabilities and writes the calibrated version to models/churn_model.joblib,
    # which is the one actually recommended for deployment (see README).
    joblib.dump({"model": best, "threshold": float(opt["threshold"])},
                "models/churn_model_xgb_raw.joblib")
    joblib.dump({"model": logreg}, "models/churn_model_logreg.joblib")
    print("\nSaved models/churn_model_xgb_raw.joblib and reports/train_summary.json")

    # save test split + probs for downstream explain/calibration scripts
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    np.save("data/processed/proba_xgb.npy", proba_xgb)


if __name__ == "__main__":
    main()
