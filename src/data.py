"""Load and clean the Telco Customer Churn dataset."""
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Telco churn dataset.

    Key decisions documented here -- this docstring is interview material.
    """
    df = df.copy()

    # TotalCharges ships as an object dtype because 11 rows contain a
    # blank string. Every one of those rows has tenure == 0, i.e. brand
    # new customers who have not yet been billed. Their true total
    # charges is 0, not missing -- so we impute 0 rather than drop.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all(), (
        "Unexpected NaN in TotalCharges outside tenure==0"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # SeniorCitizen is 0/1 int but is semantically categorical.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Several columns encode "No internet service" / "No phone service"
    # as a third level that is functionally identical to "No" for
    # modeling purposes. Collapsing reduces sparsity after one-hot
    # encoding without losing signal (InternetService already carries it).
    #
    # NOTE: pandas >= 3.0 defaults string columns to its new "str" dtype
    # instead of the legacy "object" dtype, so `df[col].dtype == object`
    # silently stops matching string columns and this collapse becomes a
    # no-op. is_string_dtype covers both.
    string_cols = [c for c in df.columns
                   if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    for col in string_cols:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    # Target as int
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # customerID is an identifier, not a feature
    df = df.drop(columns=["customerID"])

    return df


def load_clean(path: str = RAW_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


if __name__ == "__main__":
    raw = load_raw()
    print("Raw shape:", raw.shape)
    print(raw.dtypes)
    print("\nChurn rate:\n", raw["Churn"].value_counts(normalize=True))

    blank = raw[raw["TotalCharges"].str.strip() == ""]
    print(f"\nBlank TotalCharges rows: {len(blank)}")
    print("Their tenure values:", sorted(blank["tenure"].unique()))

    cleaned = clean(raw)
    print("\nCleaned shape:", cleaned.shape)
    print("Nulls after cleaning:\n", cleaned.isna().sum().sum())
    cleaned.to_csv("data/processed/telco_clean.csv", index=False)
    print("\nSaved cleaned dataset to data/processed/telco_clean.csv")
