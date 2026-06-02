import streamlit as st
import numpy as np
import joblib
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered")
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
st.sidebar.title("About")

st.sidebar.info(
    "This AI system predicts diabetes using Machine Learning."
)

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title(st.markdown(
    "<h1 style='text-align:center;color:#ff4b4b;'>🩺 AI Diabetes Prediction System</h1>",
    unsafe_allow_html=True
))

preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):

    input_data = np.array([[
        preg,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    risk = probability[0][1] * 100

    st.write(f"Diabetes Risk: {risk:.2f}%")

    if prediction[0] == 1:
        st.error("⚠️ Person is Diabetic")
    else:
        st.success("✅ Person is Non-Diabetic")
        st.subheader("Health Tips")

st.write("🥗 Eat healthy food")
st.write("🏃 Exercise daily")
st.write("💧 Drink enough water")
st.write("😴 Sleep properly")
st.markdown("---")
st.write("Developed by Vinay Teja 🚀")