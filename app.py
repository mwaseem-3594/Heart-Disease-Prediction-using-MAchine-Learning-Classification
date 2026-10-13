import streamlit as st
import pickle
import numpy as np

# Configure page layout
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Classification Dashboard")
st.write("Input patient clinical descriptors below to evaluate the probability/risk of heart disease presence.")

# Load the saved model and scaler assets safely
@st.cache_resource
def load_assets():
    with open('heart_disease_prediction_model.pkl', 'rb') as f:
      data = pickle.load(f)
    return data

try:
    assets = load_assets()
    
    # Handle both structured dictionary files or standalone model fallbacks safely
    if isinstance(assets, dict):
        model = assets.get('model')
        scaler = assets.get('scaler')
    else:
        model = assets
        scaler = None
        
except FileNotFoundError:
    st.error("🚨 'heart_model_assets.pkl' not found. Please run the training script ('python train_model.py') first in your VS Code terminal!")
    st.stop()

st.divider()
st.subheader("Patient Demographics & Vitals")

# Arrange inputs in clean columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age (years)", min_value=1, max_value=100, value=54)
    sex = st.selectbox("Biological Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)

with col2:
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], 
                      format_func=lambda x: f"Type {x}")
    chol = st.slider("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False (0)" if x == 0 else "True (1)")

with col3:
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")

st.divider()
st.subheader("Advanced Electrocardiogram Metrics")
col4, col5, col6, col7 = st.columns(4)

with col4:
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
with col5:
    slope = st.selectbox("Peak Exercise ST Segment Slope", options=[0, 1, 2])
with col6:
    ca = st.selectbox("Major Vessels Colored by Flourosopy", options=[0, 1, 2, 3, 4])
with col7:
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# Run inference on click
st.markdown("<br>", unsafe_allow_html=True)
if st.button("📊 Calculate Diagnostic Prediction", type="primary", use_container_width=True):
    
    # Construct raw data vector matching the original feature layout order
    raw_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale inputs dynamically using original scaler calculations if available
    if scaler is not None:
        processed_features = scaler.transform(raw_features)
    else:
        processed_features = raw_features
    
    # Compute prediction probabilities
    prediction = model.predict(processed_features)[0]
    probabilities = model.predict_proba(processed_features)[0]
    
    st.divider()
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected:** The model predicts **Heart Disease presence** (Confidence: {probabilities[1]:.1%})")
    else:
        st.success(f"✅ **Low Risk Detected:** The model predicts **No Heart Disease presence** (Confidence: {probabilities[0]:.1%})")
        
    st.write("### Prediction Certainty Profile")
    st.progress(float(probabilities[1]), text=f"Heart Disease Risk Probability: {probabilities[1]:.1%}")