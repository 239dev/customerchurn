"""Load and clean the raw Telco churn CSV."""
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is an object column because 11 rows are blank strings
    # instead of numbers. All 11 turn out to be tenure==0 (brand new
    # customers who haven't been billed yet), so 0 is the right fill value,
    # not a drop.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all(), \
        "found a NaN TotalCharges row with tenure != 0, assumption above doesn't hold"
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # stored as 0/1 but it's really a category
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # "No internet service" / "No phone service" show up as their own
    # category on several columns, but they're just "No" -- InternetService
    # already tells you they have no internet. Collapsing avoids a bunch of
    # redundant one-hot columns later.
    #
    # pandas 3.0 switched the default dtype for string columns from object
    # to a new "str" dtype, so `dtype == object` alone silently misses them.
    string_cols = [c for c in df.columns
                   if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    for col in string_cols:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])  # not a feature

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
    print("Nulls after cleaning:", cleaned.isna().sum().sum())
    cleaned.to_csv("data/processed/telco_clean.csv", index=False)
    print("Saved data/processed/telco_clean.csv")
