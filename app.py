import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="centered"
)

# --- Load Trained Model ---
# @st.cache_resource use karne se model baar-baar load nahi hoga
@st.cache_resource
def load_diabetes_model():
    # Ensure karein ki diabetes.pkl file aapke GitHub repo ke main folder mein hai
    return joblib.load("diabetes.pkl")

# Yahan hum model ko load kar rahe hain
model = load_diabetes_model()

# --- UI Header ---
st.title("🏥 Diabetes Prediction App")
st.write("Enter the patient details below to predict the outcome.")

# --- User Inputs (Matching your CSV columns) ---
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

# --- Prediction Logic ---
if st.button("Predict Result 🩺"):
    # 1. DataFrame banana jo training data (X_train) jaisa dikhe [cite: 10, 11]
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    # 2. Model Prediction [cite: 12, 13]
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.markdown("---")
        if prediction[0] == 1:
            st.error(f"### Result: Positive for Diabetes (Probability: {probability[0][1]:.2%})")
        else:
            st.success(f"### Result: Negative for Diabetes (Probability: {probability[0][0]:.2%})")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- Footer ---
st.markdown("---")
st.write("Developed by PRINCE RAJPUT")
