import pandas as pd

from src.features import build_preprocessor, NUMERIC


def test_numeric_and_categorical_columns_end_up_in_separate_transformers():
    X = pd.DataFrame({
        "tenure": [1, 24, 60],
        "MonthlyCharges": [50.0, 70.0, 90.0],
        "TotalCharges": [50.0, 1680.0, 5400.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "gender": ["Female", "Male", "Female"],
    })
    prep = build_preprocessor(X)
    prep.fit(X)

    names = prep.get_feature_names_out()
    assert all(f"num__{col}" in names for col in NUMERIC)
    assert any(n.startswith("cat__Contract_") for n in names)


def test_scaled_numeric_columns_are_roughly_standardized():
    X = pd.DataFrame({
        "tenure": [0, 36, 72],
        "MonthlyCharges": [20.0, 60.0, 100.0],
        "TotalCharges": [0.0, 2000.0, 7000.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
    })
    prep = build_preprocessor(X)
    transformed = prep.fit_transform(X)
    numeric_cols = transformed[:, :3]
    assert abs(numeric_cols.mean()) < 1e-8
