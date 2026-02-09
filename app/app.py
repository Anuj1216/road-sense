import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model/risk_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚦 RoadSense – Accident Risk Predictor (Prototype)")

weather = st.selectbox("Weather", ["Clear", "Rain", "Fog"])
visibility = st.selectbox("Visibility", ["High", "Medium", "Low"])
vehicles = st.slider("Number of Vehicles Involved", 1, 5)
road = st.selectbox("Road Type", ["Urban", "Highway"])

# Encoding (same order as training)
weather_map = {"Clear": 0, "Rain": 1, "Fog": 2}
visibility_map = {"High": 0, "Medium": 1, "Low": 2}
road_map = {"Urban": 0, "Highway": 1}

if st.button("Predict Risk"):
    input_data = np.array([[ 
        weather_map[weather],
        visibility_map[visibility],
        vehicles,
        road_map[road]
    ]])

    prediction = model.predict(input_data)[0]

    risk_map = {0: "Low Risk 🟢", 1: "Medium Risk 🟠", 2: "High Risk 🔴"}
    st.subheader("Prediction Result:")
    st.success(risk_map[prediction])
