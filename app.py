import streamlit as st
import numpy as np
import pandas as pd

from train import load_and_preprocess, train_models
from fuzzy import create_fuzzy_system, get_fuzzy_score

st.set_page_config(page_title="Mental Health Risk", layout="wide")

# =========================
# LOAD MODELS + ENCODERS
# =========================
@st.cache_resource
def load_all():
    X_train, X_test, y_train, y_test, scaler, encoders, feature_names = load_and_preprocess("final_depression_dataset_1.csv")
    xgb, cat = train_models(X_train, y_train)
    fuzzy_sim = create_fuzzy_system()
    return xgb, cat, fuzzy_sim, scaler, encoders, feature_names

xgb, cat, fuzzy_sim, scaler, encoders, feature_names = load_all()

# =========================
# UI
# =========================
st.title("🧠 Mental Health Risk Assessment")
st.markdown("### Hybrid Model (XGBoost + CatBoost + Fuzzy Logic)")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Name")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 15, 80, 25)
    city = st.text_input("City")
    academic_pressure = st.slider("Academic Pressure", 0, 10, 5)
    study_satisfaction = st.slider("Study Satisfaction", 0, 10, 5)
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

with col2:
    role = st.selectbox("Working Professional or Student", ["Student", "Working"])
    profession = st.text_input("Profession")
    work_pressure = st.slider("Work Pressure", 0, 10, 5)
    job_satisfaction = st.slider("Job Satisfaction", 0, 10, 5)
    degree = st.selectbox("Degree", ["UG", "PG", "PhD"])
    financial_stress = st.slider("Financial Stress", 0, 10, 5)

with col3:
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
    sleep = st.slider("Sleep Duration (hours)", 0, 12, 7)
    suicidal = st.radio("Have you ever had suicidal thoughts?", ["No", "Yes"])
    family_history = st.radio("Family History of Mental Illness", ["No", "Yes"])
    work_hours = st.slider("Work/Study Hours", 0, 16, 6)

# =========================
# HELPER: ENCODE FUNCTION
# =========================
def encode_value(col, val):
    if col in encoders:
        le = encoders[col]
        try:
            return le.transform([val])[0]
        except:
            return 0  # unseen value
    return val

# =========================
# PREPROCESS INPUT (FIXED)
# =========================
def preprocess_input():

    input_dict = {
        "Name": name,
        "Gender": gender,
        "Age": age,
        "City": city,
        "Working Professional or Student": role,
        "Profession": profession,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal,
        "Work/Study Hours": work_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Encode categorical columns using SAME encoders
    for col in input_df.columns:
        input_df[col] = input_df[col].apply(lambda x: encode_value(col, x))

    # Ensure correct column order
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)
    

    return input_scaled

# =========================
# PREDICT
# =========================
if st.button("🔍 Predict Risk"):

    input_array = preprocess_input()

    xgb_pred = xgb.predict(input_array)[0]
    cat_pred = cat.predict(input_array)[0]

    xgb_prob = max(xgb.predict_proba(input_array)[0])
    cat_prob = max(cat.predict_proba(input_array)[0])

    fuzzy_score = get_fuzzy_score(
        fuzzy_sim,
        sleep=min(sleep, 10),
        stress=max(work_pressure, academic_pressure),
        social=10 - financial_stress
    )

    # =========================
# INTERPRETATION FUNCTIONS
# =========================

def ml_label(pred):
    return "High Risk ⚠️" if pred == 1 else "Low Risk ✅"

def fuzzy_label(score):
    if score <= 3:
        return "Low Risk ✅"
    elif score <= 6:
        return "Moderate Risk ⚖️"
    else:
        return "High Risk ⚠️"

def final_decision(xgb_prob, cat_prob, fuzzy_score):
    avg_conf = (xgb_prob + cat_prob) / 2

    if avg_conf > 0.7 or fuzzy_score > 7:
        return "High Risk ⚠️", "Immediate attention recommended"
    elif avg_conf > 0.5 or fuzzy_score > 4:
        return "Moderate Risk ⚖️", "Monitor mental health regularly"
    else:
        return "Low Risk ✅", "No immediate concern"

# =========================
# DISPLAY RESULTS
# =========================

st.markdown("---")
st.subheader("📊 Assessment Results")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 🚀 XGBoost")
    st.metric("Prediction", ml_label(xgb_pred))
    st.progress(float(min(max(xgb_prob, 0), 1)))
    st.write(f"{float(xgb_prob)*100:.2f}% confidence")

with c2:
    st.markdown("### 🌲 CatBoost")
    st.metric("Prediction", ml_label(cat_pred))
    st.progress(float(min(max(cat_prob, 0), 1)))
    st.write(f"{float(cat_prob)*100:.2f}% confidence")

with c3:
    st.markdown("### 🧠 Fuzzy Logic")
    st.metric("Risk Level", fuzzy_label(fuzzy_score))
    st.write(f"Score: {round(fuzzy_score, 2)} / 10")

# =========================
# FINAL DECISION
# =========================

st.markdown("---")
st.subheader("🧾 Final Assessment")

decision, message = final_decision(xgb_prob, cat_prob, fuzzy_score)

if "High" in decision:
    st.error(f"{decision}\n\n{message}")
elif "Moderate" in decision:
    st.warning(f"{decision}\n\n{message}")
else:
    st.success(f"{decision}\n\n{message}")

st.write(f"Overall Confidence: {((xgb_prob + cat_prob)/2)*100:.2f}%")
# =========================
# 📊 GRAPH DASHBOARD
# =========================

st.markdown("---")
st.subheader("📊 Analytics Dashboard")

colA, colB = st.columns(2)

# =========================
# 1. MODEL CONFIDENCE COMPARISON
# =========================
with colA:
    st.markdown("### 🔍 Model Confidence Comparison")

    conf_df = pd.DataFrame({
        "Model": ["XGBoost", "CatBoost"],
        "Confidence (%)": [
            float(xgb_prob)*100,
            float(cat_prob)*100
        ]
    })

    st.bar_chart(conf_df.set_index("Model"))

# =========================
# 2. INPUT FEATURE DISTRIBUTION
# =========================
with colB:
    st.markdown("### 📥 Input Feature Distribution")

    input_values = input_array.flatten()

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": input_values
    })

    st.line_chart(feature_df.set_index("Feature"))

# =========================
# 3. RISK BREAKDOWN
# =========================

st.markdown("### 🧠 Risk Breakdown")

avg_conf = float((xgb_prob + cat_prob) / 2) * 100

risk_df = pd.DataFrame({
    "Category": ["Low Risk", "High Risk"],
    "Value": [100 - avg_conf, avg_conf]
})

st.bar_chart(risk_df.set_index("Category"))

# =========================
# 4. ML vs FUZZY COMPARISON
# =========================

st.markdown("### ⚖️ ML vs Fuzzy Comparison")

comparison_df = pd.DataFrame({
    "System": ["XGBoost", "CatBoost", "Fuzzy"],
    "Score": [
        float(xgb_prob)*100,
        float(cat_prob)*100,
        fuzzy_score * 10   # scale fuzzy (0–10 → 0–100)
    ]
})

st.bar_chart(comparison_df.set_index("System"))