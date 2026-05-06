import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.title(" Heart Disease Prediction App")

# Numeric inputs
age = st.number_input("Age", 1, 120)
resting_bp = st.number_input("Resting Blood Pressure")
cholesterol = st.number_input("Cholesterol")
fasting_bs = st.selectbox("Fasting Blood Sugar > 120?", ["Yes", "No"])
max_hr = st.number_input("Max Heart Rate")
oldpeak = st.number_input("Oldpeak")

# Categorical inputs
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
rest_ecg = st.selectbox("Resting ECG", ["LVH", "Normal", "ST"])

# Convert values
fasting_bs_val = 1 if fasting_bs == "Yes" else 0

# One-hot encoding (IMPORTANT)
st_slope_val = 1 if st_slope == "Up" else 0

cp_asy = 1 if chest_pain == "ASY" else 0
cp_ata = 1 if chest_pain == "ATA" else 0
cp_nap = 1 if chest_pain == "NAP" else 0
cp_ta = 1 if chest_pain == "TA" else 0

ecg_lvh = 1 if rest_ecg == "LVH" else 0
ecg_normal = 1 if rest_ecg == "Normal" else 0
ecg_st = 1 if rest_ecg == "ST" else 0

if st.button("Predict"):
    data = np.array([[ 
        age,
        resting_bp,
        cholesterol,
        fasting_bs_val,
        max_hr,
        oldpeak,
        st_slope_val,
        cp_asy, cp_ata, cp_nap, cp_ta,
        ecg_lvh, ecg_normal, ecg_st
    ]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")