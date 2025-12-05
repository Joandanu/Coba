import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.joblib")

st.title("Customer Churn Prediction App")
st.write("Masukkan data pelanggan untuk memprediksi Churn")

# ===== INPUT =====

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure", min_value=0, max_value=100)
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method",["Electronic check", "Mailed check", 
                                               "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# ===== DATAFRAME HARUS SESUAI URUTAN FITUR TRAINING =====
data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod
}])

# ===== PREDIKSI =====
if st.button("Prediksi"):
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.error(f"⚠ Pelanggan berpotensi CHURN (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✔ Pelanggan TIDAK Churn (Probabilitas: {prob:.2f})")

# ==== DEBUG: CEK INPUT ====
st.write("=== INPUT RAW ===")
st.write(data)

pre = model.named_steps["preprocessor"]

st.write("=== INPUT AFTER TRANSFORM ===")
transformed = pre.transform(data)
try:
    transformed = transformed.toarray()
except:
    pass
st.write(transformed)
