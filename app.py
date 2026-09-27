"""
Streamlit prototype: Water Quality Risk (Potability) Classifier
Capstone Problem 09

Run with:  streamlit run app.py

Expects these files in the same folder (produced by train_model.ipynb):
  - model.joblib
  - scaler.joblib
  - medians.joblib
  - feature_order.joblib
  - model_name.joblib
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Water Potability Classifier", page_icon="💧", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    medians = joblib.load("medians.joblib")
    feature_order = joblib.load("feature_order.joblib")
    model_name = joblib.load("model_name.joblib")
    return model, scaler, medians, feature_order, model_name


try:
    model, scaler, medians, feature_order, model_name = load_artifacts()
    artifacts_ready = True
except FileNotFoundError:
    artifacts_ready = False

st.title("💧 Water Quality Risk Classifier")
st.caption("Capstone Problem 09 — Educational prototype, not for real-world water safety decisions.")

if not artifacts_ready:
    st.error(
        "Model files not found. Run `train_model.ipynb` first, then copy "
        "`model.joblib`, `scaler.joblib`, `medians.joblib`, `feature_order.joblib`, "
        "and `model_name.joblib` into this folder."
    )
    st.stop()

st.markdown(f"**Model in use:** {model_name}")
st.divider()

# Reasonable real-world ranges used for input validation / sliders.
FIELD_CONFIG = {
    "ph": dict(label="pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1),
    "Hardness": dict(label="Hardness (mg/L)", min_value=0.0, max_value=400.0, value=196.0, step=1.0),
    "Solids": dict(label="Total Dissolved Solids (ppm)", min_value=0.0, max_value=60000.0, value=20000.0, step=100.0),
    "Chloramines": dict(label="Chloramines (ppm)", min_value=0.0, max_value=15.0, value=7.0, step=0.1),
    "Sulfate": dict(label="Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=333.0, step=1.0),
    "Conductivity": dict(label="Conductivity (μS/cm)", min_value=0.0, max_value=800.0, value=425.0, step=1.0),
    "Organic_carbon": dict(label="Organic Carbon (ppm)", min_value=0.0, max_value=30.0, value=14.0, step=0.1),
    "Trihalomethanes": dict(label="Trihalomethanes (μg/L)", min_value=0.0, max_value=130.0, value=66.0, step=0.1),
    "Turbidity": dict(label="Turbidity (NTU)", min_value=0.0, max_value=10.0, value=4.0, step=0.1),
}

st.subheader("Enter water sample measurements")
st.caption("Leave a field at its default only if you genuinely don't know the value — "
           "the app will fall back to the training-set median for that field.")

user_values = {}
cols = st.columns(2)
for i, feat in enumerate(feature_order):
    cfg = FIELD_CONFIG.get(feat, dict(label=feat, min_value=0.0, max_value=1000.0, value=0.0, step=1.0))
    with cols[i % 2]:
        unknown = st.checkbox(f"Unknown: {cfg['label']}", key=f"unknown_{feat}")
        if unknown:
            user_values[feat] = np.nan
            st.caption(f"Will use median: {medians[feat]:.2f}")
        else:
            user_values[feat] = st.number_input(
                cfg["label"],
                min_value=cfg["min_value"],
                max_value=cfg["max_value"],
                value=cfg["value"],
                step=cfg["step"],
                key=feat,
            )

st.divider()

if st.button("Predict potability", type="primary", use_container_width=True):
    input_df = pd.DataFrame([user_values])[feature_order]

    # Same preprocessing as training: fill missing with train medians, then scale.
    input_df = input_df.fillna(medians)
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]  # P(potable = 1)

    if pred == 1:
        st.success(f"✅ Predicted: **Potable** (probability: {proba:.1%})")
    else:
        st.error(f"⚠️ Predicted: **Not Potable** (probability of potable: {proba:.1%})")

    st.caption(
        "This is a prototype trained on a small public dataset for an educational capstone. "
        "It must not be used to make real drinking-water safety decisions."
    )
