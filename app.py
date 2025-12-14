import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ======================================================
# LOAD MODEL BUNDLE
# ======================================================
bundle = joblib.load("logistic.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
encoder = bundle["encoder"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]
feature_names = bundle["feature_names"]

# ======================================================
# HEADER
# ======================================================
st.title("📊 Customer Churn Prediction App")
st.write("Aplikasi prediksi churn pelanggan menggunakan **Logistic Regression**")

# ======================================================
# INPUT USER
# ======================================================
st.subheader("🧾 Input Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure", min_value=0, max_value=100)
    MonthlyCharges = st.number_input("Monthly Charges")
    TotalCharges = st.number_input("Total Charges", min_value=0.0)
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])

with col3:
    MultipleLines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
OnlineSecurity = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
OnlineBackup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
DeviceProtection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
TechSupport = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
StreamingTV = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])

# ======================================================
# DATAFRAME INPUT
# ======================================================
data = pd.DataFrame([{
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "gender": gender,
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

# ======================================================
# PREPROCESSING
# ======================================================
encoded_cat = pd.DataFrame(
    encoder.transform(data[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols)
)

X = pd.concat([data[num_cols], encoded_cat], axis=1)
X[num_cols] = scaler.transform(X[num_cols])

# Pastikan urutan fitur sama
X = X[feature_names]

# ======================================================
# PREDIKSI
# ======================================================
if st.button("🔮 Prediksi Churn"):
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    if pred == 1:
        st.error(f"⚠️ Pelanggan berpotensi **CHURN** (Probabilitas: {prob:.2%})")
    else:
        st.success(f"✅ Pelanggan **TIDAK CHURN** (Probabilitas: {prob:.2%})")

# ======================================================
# ANALISIS MODEL
# ======================================================
st.divider()
st.subheader("📈 Analisis Model")

tab1, tab2, tab3 = st.tabs([
    "Confusion Matrix",
    "ROC Curve",
    "Koefisien Logistic Regression"
])

# ---------- TAB 1: CONFUSION MATRIX ----------
with tab1:
    y_true = model.classes_
    y_pred = model.predict(X)

    cm = confusion_matrix([pred], [pred])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Tidak Churn", "Churn"]).plot(ax=ax)
    st.pyplot(fig)

# ---------- TAB 2: ROC CURVE ----------
with tab2:
    y_score = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve([pred], y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# ---------- TAB 3: KOEFISIEN ----------
with tab3:
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    st.write("🔼 Fitur yang meningkatkan peluang **CHURN**:")
    st.dataframe(coef_df.head(10))

    st.write("🔽 Fitur yang menurunkan peluang **CHURN**:")
    st.dataframe(coef_df.tail(10))
