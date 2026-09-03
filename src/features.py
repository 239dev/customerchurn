"""Preprocessing pipeline: numeric scaling + categorical one-hot encoding."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_preprocessor(X):
    categorical = [c for c in X.columns if c not in NUMERIC]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                categorical,
            ),
        ],
        remainder="drop",
    )
