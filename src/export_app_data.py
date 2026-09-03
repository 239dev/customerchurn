"""Bundles a small slice of test-set predictions for the Streamlit app.

The raw Kaggle CSV stays gitignored, but the model's own predictions on it
are fair game and small enough to ship -- this is what powers the live cost
simulator without the app needing the actual dataset.

python -m src.export_app_data
"""
import os
import joblib
import pandas as pd

OUT_PATH = "app/data/test_predictions.csv"


def main():
    artifact = joblib.load("models/churn_model.joblib")
    model = artifact["model"]

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    proba = model.predict_proba(X_test)[:, 1]

    out = X_test[["Contract", "tenure", "InternetService", "MonthlyCharges"]].copy()
    out["churn_actual"] = y_test.values
    out["pred_proba"] = proba

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    main()
