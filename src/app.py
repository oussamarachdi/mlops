import streamlit as st
import requests
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Breast Cancer Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction Dashboard")
st.markdown("""
This dashboard interacts with the **FastAPI** backend to provide real-time predictions using the trained MLOps pipeline.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

# Feature names from the dataset
features_list = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Default values (approximate means for a quick test)
default_values = [
    14.12, 19.29, 91.96, 654.8, 0.096, 0.104, 0.088, 0.048, 0.181, 0.062,
    0.405, 1.216, 2.866, 40.33, 0.007, 0.025, 0.031, 0.011, 0.020, 0.003,
    16.27, 25.67, 107.26, 880.5, 0.132, 0.254, 0.272, 0.114, 0.290, 0.083
]

st.header("Input Features")
st.info("Adjust the sliders below to input feature values for prediction.")

# Create columns for inputs
cols = st.columns(3)
input_data = []

for i, feature in enumerate(features_list):
    with cols[i % 3]:
        val = st.number_input(
            feature.replace(" ", " ").title(),
            value=default_values[i],
            format="%.4f",
            key=feature
        )
        input_data.append(val)

# Prediction Button
if st.button("🚀 Predict", use_container_width=True):
    with st.spinner("Querying API..."):
        try:
            response = requests.post(
                f"{api_url}/predict",
                json={"features": input_data}
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                probability = result["probability"]
                
                st.divider()
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if prediction == 0:
                        st.success("### Result: Malignant (0)")
                    else:
                        st.info("### Result: Benign (1)")
                
                with res_col2:
                    st.metric("Confidence (Probability)", f"{probability:.2%}")
                    st.progress(probability)
                
                st.balloons()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# Backend Management
st.sidebar.divider()
st.sidebar.header("Backend Management")

if st.sidebar.button("🔄 Reload Model"):
    try:
        response = requests.post(f"{api_url}/reload")
        if response.status_code == 200:
            st.sidebar.success("Model reloaded successfully!")
        else:
            st.sidebar.error(f"Failed to reload: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"Connection Error: {e}")

if st.sidebar.button("🏥 Check Health"):
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            st.sidebar.success(f"Backend Status: {response.json()['status']}")
        else:
            st.sidebar.error("Backend Unhealthy")
    except Exception as e:
        st.sidebar.error(f"Connection Error: {e}")

# Footer
st.divider()
st.caption("MLOps Mini-Project - Breast Cancer Classification Dashboard")
