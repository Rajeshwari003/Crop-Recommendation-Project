import streamlit as st
import pandas as pd
import joblib

# Load model
try:
    model_bundle = joblib.load("model.pkl")
    pipe = model_bundle["pipeline"]
    feature_cols = model_bundle["feature_order"]
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    st.stop()

st.title("🌱 Crop Recommendation System")

st.write("Enter soil and weather values to get the best crop recommendation.")

# Input fields
N = st.slider("Nitrogen", 0, 150, 90)
P = st.slider("Phosphorus", 0, 150, 40)
K = st.slider("Potassium", 0, 150, 40)
temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.slider("pH", 0.0, 14.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 200.0)

if st.button("Recommend Crop"):
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_cols)
    prediction = pipe.predict(input_df)[0]
    confidence = pipe.predict_proba(input_df).max()
    st.success(f"🌱 Recommended Crop: **{prediction}**")
    st.info(f"✅ Confidence: {confidence*100:.1f}%")
