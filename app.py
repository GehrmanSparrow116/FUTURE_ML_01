import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

pipe = load_model()

# =============================
# STYLISH HEADER
# =============================
st.markdown("""
<div style="text-align:center; padding: 10px; background: linear-gradient(90deg, #ff7e5f, #feb47b); border-radius:10px;">
    <h1 style="color:white;">📉 Customer Churn Predictor</h1>
    <p style="color:white; font-size:16px;">Quick, clean & interactive churn prediction tool</p>
</div>
""", unsafe_allow_html=True)

st.write("---")

# =============================
# INPUTS (MINIMAL)
# =============================
with st.container():
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    with col2:
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    with col3:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

# =============================
# CREATE DATAFRAME
# =============================
data = pd.DataFrame({
    "gender":["Male"],
    "SeniorCitizen":[0],
    "Partner":["No"],
    "Dependents":["No"],
    "tenure":[tenure],
    "PhoneService":["Yes"],
    "MultipleLines":["No"],
    "InternetService":[InternetService],
    "OnlineSecurity":["No"],
    "OnlineBackup":["No"],
    "DeviceProtection":["No"],
    "TechSupport":[TechSupport],
    "StreamingTV":["No"],
    "StreamingMovies":["No"],
    "Contract":[Contract],
    "PaperlessBilling":[PaperlessBilling],
    "PaymentMethod":[PaymentMethod],
    "MonthlyCharges":[MonthlyCharges],
    "TotalCharges":[tenure * MonthlyCharges]
})

# =============================
# ENFORCE DTYPE
# =============================
categoricals = [
    "gender","Partner","Dependents","PhoneService","MultipleLines",
    "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies","Contract",
    "PaperlessBilling","PaymentMethod"
]
for c in categoricals:
    data[c] = data[c].astype(str)

data["SeniorCitizen"] = data["SeniorCitizen"].astype(int)
data["tenure"] = data["tenure"].astype(int)
data["MonthlyCharges"] = data["MonthlyCharges"].astype(float)
data["TotalCharges"] = data["TotalCharges"].astype(float)

# =============================
# PREDICT BUTTON
# =============================
st.write("")
predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

if predict_btn:
    with st.spinner("Calculating churn probability…"):
        prob = pipe.predict_proba(data)[:,1][0]
        threshold = 0.2
        churn = int(prob >= threshold)

    # =============================
    # GAUGE PROBABILITY
    # =============================
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "red" if churn else "green"},
                 'steps' : [
                     {'range': [0, 35], 'color': "green"},
                     {'range': [35, 70], 'color': "yellow"},
                     {'range': [70, 100], 'color': "red"}]}))
    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # RISK MESSAGE
    # =============================
    if churn:
        st.error(f"⚠️ High Risk! This customer is likely to churn ({prob*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk. This customer is unlikely to churn ({prob*100:.1f}%)")

# =========================================================
# 1. KPI CARDS
# =========================================================
st.write("---")
st.subheader("Model Metrics")

# Example metrics (replace with your actual calculations)
accuracy = 0.85   # if you have calculated
auc = 0.8375      # ROC-AUC
high_risk_count = (prob >= 0.2) if predict_btn else 0  # Customers likely to churn

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col2.metric("ROC-AUC", f"{auc:.2f}")
col3.metric("High-Risk Customers", f"{high_risk_count}")
# =============================
# SHAP EXPLAINER - Stylish Version
# =============================
st.write("---")
st.subheader("Top Features Influencing Prediction")

try:
    # Transform input data
    X_trans = pipe.named_steps["prep"].transform(data)

    # Get feature names
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(pipe.named_steps["prep"].transformers_[0][2])
    num_cols = pipe.named_steps["prep"].transformers_[1][2]
    feature_names = np.array(list(cat_features) + list(num_cols))

    # SHAP explainer
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_trans)

    # Plot with style
    shap.initjs()
    fig, ax = plt.subplots(figsize=(6,6))
    shap.summary_plot(
        shap_values, 
        X_trans, 
        feature_names=feature_names, 
        plot_type="dot",        # beeswarm dot plot
        color_bar=True,         # gradient color bar
        max_display=10,         # show top 10 features
        show=False
    )

    # Customize plot aesthetics
    ax.set_facecolor("#f5f5f5")   # light background
    st.pyplot(fig)

except Exception as e:
    st.write("SHAP plot could not be rendered:", e)



st.write("---")
st.caption("Made for churn modeling — clean, fast & aesthetic 💙")
