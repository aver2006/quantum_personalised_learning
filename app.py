
import streamlit as st
import numpy as np
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from model_service import train_models


st.set_page_config(
    page_title="Quantum Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("Quantum Machine Learning for Personalised Student Learning")

st.write(
    "An experimental application comparing Quantum SVC "
    "and Classical SVM for student learning-support classification."
)

st.warning(
    "This is an educational prototype. Predictions are experimental "
    "and should not be used as the sole basis for academic decisions."
)


@st.cache_resource
def load_models():
    return train_models()


with st.spinner("Training quantum and classical models. Please wait..."):
    quantum_model, classical_model, scaler = load_models()

st.success("Both models are ready!")


st.header("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    absences = st.number_input(
        "Number of absences",
        min_value=0,
        max_value=100,
        value=2
    )

    studytime = st.selectbox(
        "Weekly study time",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "More than 10 hours"
        }[x]
    )

with col2:
    g1 = st.number_input(
        "First period grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

    g2 = st.number_input(
        "Second period grade (G2)",
        min_value=0,
        max_value=20,
        value=12
    )


recommendations = {
    0: "Provide advanced learning material and enrichment activities.",
    1: "Provide additional practice questions and learning resources.",
    2: "Provide remedial lessons and individual teacher support."
}

labels = {
    0: "Low support required",
    1: "Moderate support required",
    2: "High support required"
}


if st.button("Predict Learning Support", type="primary"):

    student_data = np.array([[
        absences,
        studytime,
        g1,
        g2
    ]], dtype=float)

    scaled_data = scaler.transform(student_data)

    quantum_prediction = int(quantum_model.predict(scaled_data)[0])
    classical_prediction = int(classical_model.predict(scaled_data)[0])

    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quantum SVC")
        st.metric(
            "Predicted category",
            labels[quantum_prediction]
        )

    with col2:
        st.subheader("Classical SVM")
        st.metric(
            "Predicted category",
            labels[classical_prediction]
        )

    st.subheader("Personalised Learning Recommendation")

    st.info(recommendations[quantum_prediction])

    if quantum_prediction == classical_prediction:
        st.success("Both models agree on the predicted category.")
    else:
        st.warning(
            "The models predict different categories. "
            "Further academic assessment is recommended."
        )

st.divider()

st.header("Experimental Results")

# Model comparison table
st.subheader("Quantum SVC vs Classical SVM")

if os.path.exists("results/model_comparison.csv"):
    results_df = pd.read_csv("results/model_comparison.csv")
    st.dataframe(results_df, use_container_width=True)

# Performance comparison graph
if os.path.exists("results/model_comparison.png"):
    st.subheader("Model Performance Comparison")
    st.image("results/model_comparison.png", use_container_width=True)

# Confusion matrices
if os.path.exists("results/confusion_matrices.png"):
    st.subheader("Confusion Matrices")
    st.image("results/confusion_matrices.png", use_container_width=True)

# Quantum Feature Map Circuits
st.divider()
st.header("Quantum Feature Map Circuits")

# ZFeatureMap
if os.path.exists("results/zfeaturemap_circuit.png"):
    st.subheader("Four-Qubit Z Feature Map")
    st.image(
        "results/zfeaturemap_circuit.png",
        use_container_width=True
    )

# ZZFeatureMap
if os.path.exists("results/zzfeaturemap_circuit.png"):
    st.subheader("Four-Qubit ZZ Feature Map")
    st.image(
        "results/zzfeaturemap_circuit.png",
        use_container_width=True
    )

# PauliFeatureMap
if os.path.exists("results/paulifeaturemap_circuit.png"):
    st.subheader("Four-Qubit Pauli Feature Map")
    st.image(
        "results/paulifeaturemap_circuit.png",
        use_container_width=True
    )
