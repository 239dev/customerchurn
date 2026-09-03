"""Shared fixtures. raw_df mimics the real Kaggle CSV's schema/quirks on a
handful of rows, so tests don't depend on the (gitignored) actual dataset
being present.
"""
import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    rows = [
        # a normal, long-tenured customer
        dict(customerID="0001-AAA", gender="Female", SeniorCitizen=0, Partner="Yes",
             Dependents="No", tenure=60, PhoneService="Yes", MultipleLines="No",
             InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="No",
             DeviceProtection="Yes", TechSupport="No", StreamingTV="No",
             StreamingMovies="No", Contract="Two year", PaperlessBilling="No",
             PaymentMethod="Bank transfer (automatic)", MonthlyCharges=65.5,
             TotalCharges="3930.5", Churn="No"),

        # brand new customer -- the TotalCharges blank-string case
        dict(customerID="0002-BBB", gender="Male", SeniorCitizen=0, Partner="No",
             Dependents="No", tenure=0, PhoneService="Yes", MultipleLines="No",
             InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
             DeviceProtection="No", TechSupport="No", StreamingTV="No",
             StreamingMovies="No", Contract="Month-to-month", PaperlessBilling="Yes",
             PaymentMethod="Electronic check", MonthlyCharges=70.35,
             TotalCharges=" ", Churn="No"),

        # no internet at all -- OnlineSecurity/Backup/etc all read "No internet service"
        dict(customerID="0003-CCC", gender="Female", SeniorCitizen=1, Partner="No",
             Dependents="No", tenure=12, PhoneService="Yes", MultipleLines="No",
             InternetService="No", OnlineSecurity="No internet service",
             OnlineBackup="No internet service", DeviceProtection="No internet service",
             TechSupport="No internet service", StreamingTV="No internet service",
             StreamingMovies="No internet service", Contract="One year",
             PaperlessBilling="No", PaymentMethod="Mailed check", MonthlyCharges=20.05,
             TotalCharges="240.6", Churn="Yes"),

        # no phone -- MultipleLines reads "No phone service"
        dict(customerID="0004-DDD", gender="Male", SeniorCitizen=0, Partner="Yes",
             Dependents="Yes", tenure=24, PhoneService="No",
             MultipleLines="No phone service", InternetService="DSL",
             OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="No",
             TechSupport="Yes", StreamingTV="Yes", StreamingMovies="No",
             Contract="Month-to-month", PaperlessBilling="Yes",
             PaymentMethod="Credit card (automatic)", MonthlyCharges=45.2,
             TotalCharges="1084.8", Churn="Yes"),
    ]
    return pd.DataFrame(rows)
