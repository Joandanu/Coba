import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.joblib")

st.title("Customer Churn Prediction App")
st.write("Masukkan data pelanggan untuk memprediksi Churn")

# ===== INPUT =====

tenure = st.number_input("Tenure", min_value=0, max_value=100)
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges", min_value=0.0)

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# ==========================================
#   ⚠ DI SINI MAPPING DITARUH (PERBAIKAN 2)
# ==========================================

yn_map = {"Yes": 1, "No": 0}

Partner = yn_map[Partner]
Dependents = yn_map[Dependents]
PhoneService = yn_map[PhoneService]
MultipleLines = MultipleLines  # tetap string → nanti di encoder one-hot
PaperlessBilling = yn_map[PaperlessBilling]

OnlineSecurity = OnlineSecurity
OnlineBackup = OnlineBackup
DeviceProtection = DeviceProtection
TechSupport = TechSupport
StreamingTV = StreamingTV
StreamingMovies = StreamingMovies

gender = 1 if gender == "Male" else 0   # sama seperti dataset CSV kamu


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
