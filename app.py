import streamlit as st
import pandas as pd
import joblib

# Load model
bundle = joblib.load("logistic.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
encoder = bundle["encoder"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]

st.title("Customer Churn Prediction App")
st.write("Masukkan data pelanggan untuk memprediksi Churn")

# ===== INPUT =====
tenure = st.number_input("Tenure", min_value=0, max_value=100)
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges", min_value=0.0)

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
OnlineBackup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
DeviceProtection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
TechSupport = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
StreamingTV = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.selectbox("Payment Method", [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check"
])


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

# PREPROCESSING (HARUS SAMA DENGAN TRAINING)
# ======================================================
# 1. One-Hot Encoding
encoded_cat = pd.DataFrame(
    encoder.transform(data[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols)
)

# 2. Gabungkan numerik + encoding
X = pd.concat(
    [data[num_cols].reset_index(drop=True), encoded_cat],
    axis=1
)

# 3. Scaling numerik
X[num_cols] = scaler.transform(X[num_cols])

st.write("num_cols dari bundle:")
st.write(num_cols)


# ===== PREDIKSI =====
if st.button("Prediksi"):
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.error(f"⚠ Pelanggan berpotensi CHURN (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✔ Pelanggan TIDAK Churn (Probabilitas: {prob:.2f})")

# ========== DEBUG TANPA GAMBAR ==========

with st.expander("🧪 Debug Info"):
    st.write("Input user:")
    st.dataframe(data)

    st.write("Jumlah fitur setelah preprocessing:", X.shape[1])
    st.write("20 fitur pertama:")
    st.write(X.iloc[0, :20])
