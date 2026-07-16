import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import base64
import plotly.express as px
import plotly.graph_objects as go

# ========== Page Config ==========
st.set_page_config(page_title="Student Score Prediction App", layout="wide", page_icon="🎓")

# ========== Load Scalers, Encoders & Model ==========
SCALER_PATH = Path("models")
MODEL_PATH = SCALER_PATH / "model_poly.pkl"
model = joblib.load(MODEL_PATH)
poly_transformer = joblib.load(SCALER_PATH / "poly_transformer.pkl")

attendance_scaler = joblib.load(SCALER_PATH / "Attendance_scaler.pkl")
hours_scaler = joblib.load(SCALER_PATH / "Hours_Studied_scaler.pkl")
previous_scaler = joblib.load(SCALER_PATH / "Previous_Scores_scaler.pkl")
tutoring_scaler = joblib.load(SCALER_PATH / "Tutoring_Sessions_scaler.pkl")
physical_scaler = joblib.load(SCALER_PATH / "Physical_Activity_scaler.pkl")
sleep_scaler = joblib.load(SCALER_PATH / "Sleep_Hours_scaler.pkl")

resources_encoder = joblib.load(SCALER_PATH / "Access_to_Resources_encoder.pkl")
parental_encoder = joblib.load(SCALER_PATH / "Parental_Involvement_encoder.pkl")
income_encoder = joblib.load(SCALER_PATH / "Family_Income_encoder.pkl")


# ========== Custom Background ==========
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: black;  /* make all text black */
    }}
    h1, h2, h3, h4, h5, h6, p, div, span, label {{
        color: black !important;  /* force text to black */
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Call it once to set background
set_background("assets/Background.jpg")

# ========== Sidebar Navigation ==========
st.sidebar.image("assets/student-banner.jpg", use_container_width=True)
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "📈 Data Insights", "🔮 Predict"])


# ========== Footer ==========
def footer():
    st.markdown("""<hr style='border:1px solid #ccc;'>""", unsafe_allow_html=True)
    st.markdown("""<center style='font-size:14px;'>© 2025 | Built with ❤️ by <a href='https://github.com/Mo7amed3bdelghany' target='_blank'>Mohamed Abdelghany</a></center>""", unsafe_allow_html=True)


# ========== Home Page ==========
if page == "🏠 Home":
    st.title("🎓 Student Score Prediction App")
    st.markdown("""
    > **AI-powered assistant for predicting student exam scores**

    ---

    📚 This application helps you **predict student performance** based on academic,
    social, and lifestyle factors. It leverages a **Polynomial Regression Model** trained on
    educational datasets.

    ---

    ### ✅ Features
    - 📊 **Interactive Dataset Insights**
    - 🔮 **Real-time Prediction**
    - 📈 **Gauge Chart Visualization**
    - 📝 **Personalized Recommendations**
    - 📥 **Downloadable Results**
    - 🎯 Designed for **students, teachers, and researchers**

    ---
    """)
    # st.image(background_image, use_container_width=True)
    footer()

# ========== Data Insights ==========
elif page == "📈 Data Insights":
    st.title("📈 Dataset Insights")
    df = pd.read_csv("data/cleaned_data.csv")

    tab1, tab2, tab3 = st.tabs(["🔍 Preview", "📊 Visualizations", "📌 Statistics"])

    with tab1:
        st.subheader("🔍 Preview of the Dataset")
        if df.empty:
            st.warning("No dataset available. Upload `student_scores_cleaned.csv` to enable dashboard features.")
        else:
            n_rows = st.slider('Choose number of rows to show', min_value=5, max_value=len(df), step=1)
            columns_to_show = st.multiselect('Select Columns to Show', df.columns.tolist(), default=df.columns.tolist())
            st.write(df.iloc[:n_rows][columns_to_show])

    with tab2:
        st.subheader("📊 Visualizations")
        vis_tabs = st.tabs([
            '🔸 Scatter Plot',
            '🔹 Histogram',
            '🔺 Pie Chart',
            '📦 Box Plot',
            '📊 Count Plot'
        ])

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = ["Access_to_Resources", "Parental_Involvement", "Family_Income"]

        # Scatter
        with vis_tabs[0]:
            x = st.selectbox("X-axis", num_cols)
            y = st.selectbox("Y-axis", num_cols)
            color = st.selectbox("Color by", df.columns)
            fig = px.scatter(df, x=x, y=y, color=color)
            st.plotly_chart(fig)

        # Histogram
        with vis_tabs[1]:
            feature = st.selectbox("Feature to plot:", num_cols, key="hist")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=feature, kde=True, ax=ax)
            ax.set_title(f'Distribution of {feature}')
            st.pyplot(fig)

        # Pie
        with vis_tabs[2]:
            feature = st.selectbox("Categorical Feature", cat_cols)
            value_counts = df[feature].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f"{feature} Distribution")
            st.plotly_chart(fig)

        # Box Plot
        with vis_tabs[3]:
            feature = st.selectbox("Numeric feature:", num_cols, key="box")
            fig = px.box(df, y=feature)
            st.plotly_chart(fig)

        # Count Plot
        with vis_tabs[4]:
            feature = st.selectbox("Categorical Feature", cat_cols, key="count")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=feature, palette="Set2", ax=ax)
            plt.xticks(rotation=30)
            st.pyplot(fig)

    with tab3:
        st.subheader("📌 Descriptive Statistics")
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"))

    footer()

# ========== Prediction Page ==========
elif page == "🔮 Predict":
    st.title("🔮 Student Score Prediction Form")
    st.markdown("""
    Please fill in the student's details below to predict their **exam score**.
    """)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        attendance = col1.slider("📌 Attendance (%)", 0, 100, 95)
        hours_studied = col2.slider("📖 Hours Studied per Week", 0, 40, 21)
        previous_scores = col1.slider("📝 Previous Exam Scores (%)", 0, 100, 63)
        tutoring = col2.slider("👨‍🏫 Tutoring Sessions", 0, 10, 1)
        access = col1.selectbox("📚 Access to Resources", ["High", "Medium", "Low"])
        parental = col2.selectbox("👪 Parental Involvement", ["Low", "Medium", "High"])
        physical = col1.slider("🏃 Physical Activity (hours/week)", 0, 20, 4)
        sleep = col2.slider("😴 Sleep Hours per Day", 0, 12, 7)
        income = col1.selectbox("💰 Family Income", ["Low", "Medium", "High"])

        submitted = st.form_submit_button("🔍 Predict")
        
        # st.markdown("</div>", unsafe_allow_html=True)  # إغلاق الصندوق

    if submitted:
        # Scale numeric
        attendance_scaled = attendance_scaler.transform([[attendance]])[0][0]
        hours_scaled = hours_scaler.transform([[hours_studied]])[0][0]
        prev_scaled = previous_scaler.transform([[previous_scores]])[0][0]
        tutoring_scaled = tutoring_scaler.transform([[tutoring]])[0][0]
        physical_scaled = physical_scaler.transform([[physical]])[0][0]
        sleep_scaled = sleep_scaler.transform([[sleep]])[0][0]

        # Encode categorical
        access_encoded = resources_encoder.transform([access])[0]
        parental_encoded = parental_encoder.transform([parental])[0]
        income_encoded = income_encoder.transform([income])[0]

        # Final input (before polynomial transform)
        input_array = np.array([[attendance_scaled, hours_scaled, prev_scaled,
                                tutoring_scaled, access_encoded, parental_encoded,
                                physical_scaled, sleep_scaled, income_encoded]])

        # 🔄 Apply polynomial transformation
        input_poly = poly_transformer.transform(input_array)

        # Predict using the transformed input
        pred = model.predict(input_poly)[0]

        st.subheader("🎯 Prediction Result")
        st.success(f"✅ The predicted student exam score is: **{pred:.2f}%**")

        # ------------------ 📊 Gauge Chart ------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Predicted Score"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig)

        # ------------------ 🎯 Recommendations ------------------
        st.write("### 🎯 Personalized Recommendations")

        recs = []
        if hours_studied < 10:
            recs.append("📘 Increase weekly study hours to strengthen understanding.")
        if sleep < 6:
            recs.append("😴 Improve sleep schedule for better focus and memory.")
        if attendance < 60:
            recs.append("🏫 Improve class attendance to catch important lessons.")
        if previous_scores < 50:
            recs.append("📊 Revise past material and focus on weak subjects.")
        if physical < 2:
            recs.append("🏃 Engage in regular physical activity to boost brain function.")
        if access == "Low":
            recs.append("💡 Utilize more free online learning resources.")
        if parental == "Low":
            recs.append("👪 Encourage parental involvement for better support.")
        if income == "Low":
            recs.append("💰 Explore scholarship or community support programs.")

        if not recs:
            recs.append("🌟 Excellent habits! Keep up the good work.")

        for r in recs:
            st.write(r)

        # ------------------ 📥 Download ------------------
        result_text = f"Predicted Score: {pred:.2f}%"
        result_bytes = result_text.encode()
        b64 = base64.b64encode(result_bytes).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="prediction_result.txt">📥 Download Result</a>'
        st.markdown(href, unsafe_allow_html=True)

        footer()
