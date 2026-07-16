# STUDENT SCORE PREDICTOR - STREAMLIT APP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📚",
    layout="wide"
)

# ============================================
# LOAD ONLY THE AVAILABLE PKL FILES
# ============================================

@st.cache_resource
def load_artifacts():
    models = {
        'CatBoost': joblib.load("models/catboost_model.pkl"),
        'Random Forest': joblib.load("models/random_forest_model.pkl"),
        'XGBoost': joblib.load("models/xgboost_model.pkl"),
        'Linear Regression': joblib.load("models/linear_model.pkl"),
        'Decision Tree': joblib.load("models/decision_tree_model.pkl"),
        'RF (n=300)': joblib.load("models/rf_estimators_model.pkl"),
        'RF (n=300, depth=100)': joblib.load("models/rf_depth_model.pkl")
    }
    feature_names = joblib.load("models/feature_names.pkl")
    cat_mapping = joblib.load("models/cat_mapping.pkl")
    return models, feature_names, cat_mapping

models, feature_names, cat_mapping = load_artifacts()

# ============================================
# TITLE
# ============================================

st.title("📚 Student Score Predictor")
st.markdown("Predict exam score based on study habits, attendance, and background.")

# ============================================
# SIDEBAR – MODEL SELECTION
# ============================================

st.sidebar.header("Choose Prediction Model")
selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)
model = models[selected_model_name]

# ============================================
# INPUT SECTION – with reasonable fixed ranges
# ============================================

st.header("Enter Student Information")

# ---- Numeric inputs ----
col1, col2 = st.columns(2)

with col1:
    hours = st.number_input(
        "Hours Studied",
        min_value=0.0,
        max_value=40.0,
        value=20.0,
        step=1.0,
        help="Number of hours spent studying per week"
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0,
        help="Percentage of classes attended"
    )

    sleep = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=1.0,
        help="Average hours of sleep per night"
    )

with col2:
    prev_score = st.number_input(
        "Previous Scores",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
        help="Score from previous exams (out of 100)"
    )

    physical = st.number_input(
        "Physical Activity (hours/week)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=1.0,
        help="Hours of physical activity per week"
    )

# ---- Categorical inputs ----
st.subheader("Background Information")

col3, col4 = st.columns(2)

with col3:
    motivation = st.selectbox(
        "Motivation Level",
        cat_mapping['Motivation_Level'],
        help="Student's self-reported motivation"
    )

with col4:
    parent_edu = st.selectbox(
        "Parental Education Level",
        cat_mapping['Parental_Education_Level'],
        help="Highest education level of parents"
    )

# ============================================
# ENCODE INPUT TO MODEL FORMAT
# ============================================

# Create a DataFrame with zeros for all 9 features
input_df = pd.DataFrame(0, index=[0], columns=feature_names)

# Fill numeric features
numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 
                'Previous_Scores', 'Physical_Activity']
values = {
    'Hours_Studied': hours,
    'Attendance': attendance,
    'Sleep_Hours': sleep,
    'Previous_Scores': prev_score,
    'Physical_Activity': physical
}
for col in numeric_cols:
    input_df[col] = values[col]

# ---- One‑hot encode categoricals ----
# Motivation: baseline = 'High' (both dummy columns = 0)
if motivation == 'Low':
    input_df['Motivation_Level_Low'] = 1
elif motivation == 'Medium':
    input_df['Motivation_Level_Medium'] = 1
# else 'High' → both remain 0

# Parental Education: baseline = 'College' (both dummy columns = 0)
if parent_edu == 'High School':
    input_df['Parental_Education_Level_High School'] = 1
elif parent_edu == 'Postgraduate':
    input_df['Parental_Education_Level_Postgraduate'] = 1
# else 'College' → both remain 0

# ============================================
# PREDICTION
# ============================================

st.header("Prediction")

if st.button("Predict Exam Score", type="primary"):
    pred = model.predict(input_df)[0]
    st.success(f"**Predicted Exam Score:** {pred:.2f} / 100")

    # Optional: show input values for verification
    with st.expander("Show input data"):
        st.dataframe(input_df)

# ============================================
# FOOTER (optional)
# ============================================

st.caption("Model and feature mapping loaded from the exported PKL files.")
