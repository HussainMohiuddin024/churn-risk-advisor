import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- App setup --------------------
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Churn Risk Advisor",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = BASE_DIR / "churn_model.joblib"
META_PATH = BASE_DIR / "model_meta.json"
SAMPLE_PATH = BASE_DIR / "sample_customers.csv"

RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges"
]

# -------------------- Load model and metadata --------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

model = load_model()
meta = load_metadata()
FEATURE_COLS = meta["feature_columns"]

# -------------------- Feature preparation --------------------
def prepare_input(raw):
    """Encode raw customer data using the training feature columns."""
    encoded = pd.get_dummies(raw[RAW_COLS])
    encoded = encoded.reindex(columns=FEATURE_COLS, fill_value=0)
    return encoded.astype(float)

def predict_probabilities(raw):
    encoded = prepare_input(raw)
    return model.predict_proba(encoded)[:, 1]

def risk_band(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    return "High"

# -------------------- Header --------------------
st.title("📊 Churn Risk Advisor")
st.write(
    "Estimate customer churn risk, explore what-if scenarios, "
    "and score multiple customers from a CSV file."
)

threshold = st.sidebar.slider(
    "High-risk threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(meta.get("threshold", 0.30)),
    step=0.05
)

st.sidebar.caption(
    "The threshold changes the risk label, not the model probability."
)

tab_single, tab_batch, tab_info = st.tabs(
    ["Single Customer", "Batch Scoring", "Model Information"]
)

# -------------------- Single customer --------------------
with tab_single:
    st.subheader("Customer details")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])

        with c3:
            online_security = st.selectbox(
                "Online security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["Yes", "No", "No internet service"]
            )

        c4, c5 = st.columns(2)

        with c4:
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with c5:
            monthly = st.number_input(
                "Monthly charges",
                min_value=18.0,
                max_value=120.0,
                value=70.0,
                step=1.0
            )

        submitted = st.form_submit_button("Predict churn risk")

    if submitted:
        # Approximation for demo input; actual historical totals may differ.
        total = round(tenure * monthly, 2)

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        probability = float(predict_probabilities(customer)[0])
        band = risk_band(probability)

        st.subheader("Prediction result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted churn probability", f"{probability:.1%}")
        r2.metric("Risk band", band)
        r3.metric("Decision threshold", f"{threshold:.0%}")

        if probability >= threshold:
            st.warning("This customer meets the selected high-risk threshold.")
        else:
            st.success("This customer is below the selected high-risk threshold.")

        st.caption(
            "This is a model estimate, not a guarantee. "
            "Total charges are approximated from tenure × monthly charges."
        )

        # What-if analysis
        st.subheader("What-if analysis")
        scenarios = {
            "Current customer": customer.copy(),
            "One-year contract": customer.assign(Contract="One year"),
            "Two-year contract": customer.assign(Contract="Two year"),
            "Automatic credit card": customer.assign(
                PaymentMethod="Credit card (automatic)"
            )
        }

        results = []
        for name, scenario in scenarios.items():
            p = float(predict_probabilities(scenario)[0])
            results.append({
                "Scenario": name,
                "Churn probability": p,
                "Change vs current (percentage points)": (p - probability) * 100
            })

        results_df = pd.DataFrame(results)
        results_df["Churn probability"] = (
            results_df["Churn probability"].map(lambda x: f"{x:.1%}")
        )
        results_df["Change vs current (percentage points)"] = (
            results_df["Change vs current (percentage points)"]
            .map(lambda x: f"{x:+.1f}")
        )
        st.dataframe(results_df, use_container_width=True, hide_index=True)

# -------------------- Batch scoring --------------------
with tab_batch:
    st.subheader("Batch customer scoring")
    st.write("Upload a CSV containing the raw customer columns.")

    uploaded = st.file_uploader("Choose customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)

            if not set(RAW_COLS).issubset(batch.columns):
                missing = sorted(set(RAW_COLS) - set(batch.columns))
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                probabilities = predict_probabilities(batch)
                output = batch.copy()
                output["churn_probability"] = probabilities
                output["risk_band"] = [
                    "High" if p >= threshold else "Low"
                    for p in probabilities
                ]

                st.success(f"Scored {len(output)} customers.")
                st.dataframe(output.head(20), use_container_width=True)

                csv_bytes = output.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scored CSV",
                    data=csv_bytes,
                    file_name="scored_customers.csv",
                    mime="text/csv"
                )
        except Exception as error:
            st.error(f"Could not score this file: {error}")

    elif SAMPLE_PATH.exists():
        st.info(
            "For a quick test, upload the sample_customers.csv file "
            "included with this project."
        )

# -------------------- Model information --------------------
with tab_info:
    st.subheader("Model information")
    st.write("**Model:**", meta.get("model_name", type(model).__name__))
    st.write("**Feature columns:**", len(FEATURE_COLS))
    st.write("**scikit-learn version used in training:**",
             meta.get("sklearn_version", "Not recorded"))
    st.write("**Cross-validation AUC:**", meta.get("cv_auc", "Not recorded"))
    st.write("**Test AUC:**", meta.get("test_auc", "Not recorded"))
    st.write("**Model version:**", meta.get("version", "Not recorded"))

    st.caption(
        "This demonstration supports decision-making only. "
        "Predictions may be inaccurate and should not be treated as certainty."
    )