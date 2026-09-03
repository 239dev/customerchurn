"""Churn Risk Explorer -- interactive companion to the churn prediction project.

Run locally:  streamlit run app/streamlit_app.py   (from the repo root)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

from src.evaluate import expected_cost, threshold_sweep, analytical_optimal_threshold
from src.viz import apply_style, humanize_all, BLUE, ORANGE, RED, MUTED, INK, SURFACE

st.set_page_config(page_title="Churn Risk Explorer", page_icon="📉", layout="wide")
apply_style()


@st.cache_resource
def load_models():
    prod = joblib.load(PROJECT_ROOT / "models" / "churn_model.joblib")
    raw = joblib.load(PROJECT_ROOT / "models" / "churn_model_xgb_raw.joblib")
    return prod["model"], prod["threshold"], raw["model"]


@st.cache_data
def load_test_predictions():
    return pd.read_csv(PROJECT_ROOT / "app" / "data" / "test_predictions.csv")


prod_model, prod_threshold, raw_model = load_models()
test_preds = load_test_predictions()

st.title("📉 Churn Risk Explorer")
st.caption(
    "Companion to [github.com/239dev/customerchurn](https://github.com/239dev/customerchurn) -- "
    "predicts churn risk for a hypothetical customer, and lets you see how the retention-offer "
    "economics change where the model should actually draw the line."
)

tab_predict, tab_cost = st.tabs(["🔮 Score a customer", "💰 Cost & threshold simulator"])

# ---------------------------------------------------------------------------
# Tab 1: score a hypothetical customer, explained with SHAP
# ---------------------------------------------------------------------------
with tab_predict:
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Account**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has a partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes"])
            online_backup = st.selectbox("Online backup", ["No", "Yes"])
            device_protection = st.selectbox("Device protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])

        with c3:
            st.markdown("**Billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)

        submitted = st.form_submit_button("Score this customer", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": round(tenure * monthly_charges, 2),
        }])

        proba = prod_model.predict_proba(row)[0, 1]
        will_act = proba >= prod_threshold

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted churn probability", f"{proba*100:.1f}%")
        m2.metric("Production threshold", f"{prod_threshold*100:.0f}%")
        m3.metric("Recommendation", "Send retention offer" if will_act else "No action needed")

        st.markdown("#### Why the model scored this customer this way")
        prep = raw_model.named_steps["prep"]
        clf = raw_model.named_steps["clf"]
        row_t = prep.transform(row)
        raw_names = prep.get_feature_names_out()
        feature_names = humanize_all(raw_names)

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(row_t)[0]

        display_row = row_t[0].copy()
        for i, name in enumerate(raw_names):
            if name.startswith("num__"):
                display_row[i] = row.iloc[0][name[len("num__"):]]

        expl = shap.Explanation(
            values=sv, base_values=explainer.expected_value,
            data=display_row, feature_names=feature_names,
        )
        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(expl, max_display=10, show=False)
        plt.gcf().set_facecolor(SURFACE)
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close()

# ---------------------------------------------------------------------------
# Tab 2: cost-based threshold simulator, computed live off the real test set
# ---------------------------------------------------------------------------
with tab_cost:
    st.markdown(
        "The model outputs a probability. Turning it into a decision means picking a threshold "
        "above which you actually make the offer -- and the right cutoff depends on what the offer "
        "costs, what the customer is worth, and how often the offer actually works. Adjust the "
        "assumptions below and see where the model should draw the line."
    )

    s1, s2, s3 = st.columns(3)
    offer_cost = s1.slider("Offer cost ($)", 0, 200, 50, step=5)
    ltv = s2.slider("Customer lifetime value ($)", 200, 3000, 1400, step=100)
    effectiveness = s3.slider("Offer effectiveness (% of churners saved)", 5, 95, 30, step=5) / 100

    y_true = test_preds["churn_actual"].values
    proba = test_preds["pred_proba"].values

    sweep = threshold_sweep(y_true, proba, offer_cost=offer_cost, ltv=ltv, eff=effectiveness)
    opt = sweep.loc[sweep["cost"].idxmin()]
    default_row = sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]
    analytical_t = analytical_optimal_threshold(offer_cost=offer_cost, ltv=ltv, eff=effectiveness)
    n = len(y_true)
    savings_per_customer = (default_row["cost"] - opt["cost"]) / n

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Optimal threshold", f"{opt['threshold']:.2f}")
    k2.metric("Cost at 0.50 default", f"${default_row['cost']:,.0f}")
    k3.metric("Cost at optimal", f"${opt['cost']:,.0f}")
    k4.metric("Savings / customer", f"${savings_per_customer:,.2f}")

    st.caption(f"Analytical prediction (closed form): {analytical_t:.2f}. "
               f"Annualized at 50,000 customers: ${savings_per_customer * 50_000:,.0f}.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sweep["threshold"], sweep["cost"] / 1000, lw=2.5, color=BLUE)
    ax.axvline(opt["threshold"], color=RED, ls="--", lw=1.3)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.3)
    ax.annotate(f"Optimal: {opt['threshold']:.2f}", xy=(opt["threshold"], opt["cost"] / 1000),
                xytext=(opt["threshold"] + 0.08, opt["cost"] / 1000 - sweep["cost"].max() / 12000),
                fontsize=9.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Expected retention cost, $000s (lower is better)")
    ax.set_title("Cost vs. threshold, recomputed live on the real test set (1,409 customers)",
                 fontsize=12, fontweight="bold", loc="left")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
