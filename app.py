import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model (TOP level)
# -----------------------------
model = joblib.load("diabetess.pkl")

st.title("🩺 Diabetes Prediction App")

st.write("Devloped by PRINCE RAJPUT")

# -----------------------------
# User Inputs
# -----------------------------
Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0, format="%.2f")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
Age = st.number_input("Age", min_value=1, step=1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Diabetes"):
    
    input_df = pd.DataFrame([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ Diabetes Detected")
    else:
        st.success("✅ No Diabetes Detected")

