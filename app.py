import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "heart_disease_prediction_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# Category encodings used in the training dataset
CHEST_PAIN = {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}
REST_ECG = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
ST_SLOPE = {"Undefined / not recorded": 0, "Upsloping": 1, "Flat": 2, "Downsloping": 3}

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details below to estimate the likelihood of heart disease.")

with st.form("patient_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 1, 120, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest pain type", list(CHEST_PAIN))
        resting_bp = st.number_input("Resting blood pressure (mm Hg)", 0, 250, 130)
        cholesterol = st.number_input("Serum cholesterol (mg/dl)", 0, 700, 220)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
    with c2:
        resting_ecg = st.selectbox("Resting ECG", list(REST_ECG))
        max_hr = st.number_input("Max heart rate achieved (bpm)", 50, 250, 140)
        ex_angina = st.selectbox("Exercise-induced angina", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST depression)", -3.0, 7.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST slope", list(ST_SLOPE), index=2)
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    # Column order must match the training data
    row = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "chest_pain_type": CHEST_PAIN[chest_pain],
        "resting_bp_s": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": 1 if fbs == "Yes" else 0,
        "resting_ecg": REST_ECG[resting_ecg],
        "max_heart_rate": max_hr,
        "exercise_angina": 1 if ex_angina == "Yes" else 0,
        "oldpeak": oldpeak,
        "st_slope": ST_SLOPE[st_slope],
    }])

    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Higher risk of heart disease detected (probability: {prob:.1%})")
    else:
        st.success(f"✅ Lower risk of heart disease (probability of disease: {prob:.1%})")
    st.progress(prob)

    with st.expander("Input summary"):
        st.dataframe(row, hide_index=True)

st.caption(
    "This tool is for educational purposes only and is not a medical diagnosis. "
    "Please consult a qualified doctor for any health concerns."
)
