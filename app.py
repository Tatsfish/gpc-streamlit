# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:42:25 2026

@author: tatsf
"""

import io
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GPC Classifier", layout="centered")
st.title("GaussianProcessClassifier: CSV → Predict → CSV")

@st.cache_resource
def load_model():
    clf = joblib.load("gpc_model.joblib")
    meta = joblib.load("gpc_meta.joblib")  # {"feature_names": ["d13C","d15N"]}
    return clf, meta

clf, meta = load_model()
feature_names = meta["feature_names"]

st.write("Expected columns in uploaded CSV:", feature_names)
st.write("Predictions: 0 = Pacific coasts, 1 = Sea of Japan, 2 = Offshore Pacific")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    X = df[feature_names].to_numpy()

    # Predict
    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    classes = clf.classes_

    out = df.copy()
    out["Predicted"] = pred

    # Add probability columns in correct class order
    for i, c in enumerate(classes):
        out[f"p_class_{c}"] = proba[:, i]

    st.subheader("Preview")
    st.dataframe(out.head(30))

    # Download as CSV
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )