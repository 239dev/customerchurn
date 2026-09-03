import pandas.testing as pdt

from src.data import clean


def test_blank_total_charges_becomes_zero_not_nan(raw_df):
    df = clean(raw_df)
    new_customer = df.loc[df["tenure"] == 0].iloc[0]
    assert new_customer["TotalCharges"] == 0
    assert not df["TotalCharges"].isna().any()


def test_collapses_no_internet_and_no_phone_service_categories(raw_df):
    df = clean(raw_df)
    # this is the pandas-3.0 dtype regression: these values should never
    # survive cleaning, on any column, regardless of dtype
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "str":
            assert "No internet service" not in df[col].values, col
            assert "No phone service" not in df[col].values, col


def test_senior_citizen_becomes_yes_no(raw_df):
    df = clean(raw_df)
    assert set(df["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_churn_encoded_as_binary_int(raw_df):
    df = clean(raw_df)
    assert df["Churn"].dtype.kind in "iu"
    assert set(df["Churn"].unique()) <= {0, 1}


def test_customer_id_dropped(raw_df):
    df = clean(raw_df)
    assert "customerID" not in df.columns


def test_clean_does_not_mutate_input(raw_df):
    original = raw_df.copy()
    clean(raw_df)
    pdt.assert_frame_equal(raw_df, original)
