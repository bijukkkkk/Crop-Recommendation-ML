
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Crop Recommendation System")
st.write(
    "Enter soil nutrients and weather conditions "
    "to get a crop recommendation from the trained model."
)

# Load the trained model
if not os.path.exists("crop_model.joblib"):
    st.error(
        "Model not found. Run crop_recommendation.py first."
    )
    st.stop()

model = joblib.load("crop_model.joblib")

st.subheader("Enter Soil and Weather Details")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input(
        "Nitrogen (N)", min_value=0.0, value=90.0
    )
    phosphorus = st.number_input(
        "Phosphorus (P)", min_value=0.0, value=42.0
    )
    potassium = st.number_input(
        "Potassium (K)", min_value=0.0, value=43.0
    )
    temperature = st.number_input(
        "Temperature (°C)", value=25.0
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )
    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )
    rainfall = st.number_input(
        "Rainfall (mm)", min_value=0.0, value=200.0
    )

if st.button("Recommend Crop", type="primary"):
    sample = pd.DataFrame([{
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    prediction = model.predict(sample)[0]

    st.success(f"Recommended crop: {prediction.title()}")

    st.info(
        "This is a machine-learning prediction based on "
        "the dataset, not a guarantee of crop suitability."
    )
