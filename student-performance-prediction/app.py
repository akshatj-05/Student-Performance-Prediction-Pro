from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import FEATURES, TARGET, load_dataset
from train import MODEL_DIR, build_preprocessor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Prediction")
st.caption("Machine learning demo using the UCI Student Performance dataset")
st.info(
    "This educational tool estimates a final grade from historical patterns. "
    "It is not an official grade, a measure of ability, or a guarantee."
)

@st.cache_data
def get_data():
    return load_dataset()

@st.cache_resource
def get_model():
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        # Train on demand if the user starts Streamlit before running train.py.
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error
        from train import build_preprocessor
        from src.data import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        from sklearn.ensemble import RandomForestRegressor
        df = get_data()
        X_train, X_test, y_train, y_test = train_test_split(
            df[FEATURES], df[TARGET], test_size=0.2, random_state=42
        )
        model = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
        ])
        model.fit(X_train, y_train)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    return joblib.load(model_path)

try:
    df = get_data()
    model = get_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Student records", f"{len(df):,}")
c2.metric("Input features", len(FEATURES))
c3.metric("Grade scale", "0–20")

st.subheader("Explore the dataset")
left, right = st.columns(2)
with left:
    st.write("**Final grade distribution (G3)**")
    fig, ax = plt.subplots()
    ax.hist(df[TARGET], bins=15, edgecolor="black")
    ax.set_xlabel("Final grade (G3)")
    ax.set_ylabel("Number of students")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.write("**Average final grade by study time**")
    chart_data = df.groupby("studytime")[TARGET].mean().rename("Average G3")
    st.bar_chart(chart_data)

st.subheader("Enter student information")
st.write("Provide the details below to estimate a final grade. The app uses earlier-period grades G1 and G2.")
with st.form("prediction_form"):
    a, b, c = st.columns(3)
    with a:
        school = st.selectbox("School", ["GP", "MS"], help="Dataset school codes")
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home address type", ["U", "R"], help="U = urban, R = rural")
    with b:
        studytime = st.selectbox("Weekly study time", [1, 2, 3, 4],
            format_func=lambda x: {1:"Less than 2 hours",2:"2–5 hours",3:"5–10 hours",4:"More than 10 hours"}[x])
        failures = st.selectbox("Past class failures", [0, 1, 2, 3])
        absences = st.number_input("Number of absences", min_value=0, max_value=100, value=2)
    with c:
        schoolsup = st.selectbox("Extra school support", ["no", "yes"])
        famsup = st.selectbox("Family educational support", ["no", "yes"])
        internet = st.selectbox("Internet access at home", ["yes", "no"])
        g1 = st.slider("Earlier period grade (G1)", 0, 20, 10)
        g2 = st.slider("Second period grade (G2)", 0, 20, 10)

    submitted = st.form_submit_button("Predict final grade", type="primary")

if submitted:
    student = pd.DataFrame([{
        "school": school, "sex": sex, "age": age, "address": address,
        "studytime": studytime, "failures": failures, "schoolsup": schoolsup,
        "famsup": famsup, "internet": internet, "absences": absences,
        "G1": g1, "G2": g2,
    }], columns=FEATURES)
    prediction = float(model.predict(student)[0])
    prediction = max(0.0, min(20.0, prediction))
    st.success(f"Estimated final grade: **{prediction:.1f} / 20**")
    st.caption("This is a model estimate based on historical data, not a guaranteed result.")

st.subheader("Model evaluation")
metrics_path = MODEL_DIR / "model_metrics.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
else:
    st.caption("Run `python train.py` to generate the model comparison table.")

with st.expander("View sample dataset records"):
    st.dataframe(df.head(10), use_container_width=True)
with st.expander("About the data and limitations"):
    st.markdown(
        "- Source: UCI Machine Learning Repository, Student Performance dataset.\n"
        "- This app uses Portuguese-course records from two secondary schools.\n"
        "- Associations in the dataset do not prove cause and effect.\n"
        "- The model may be inaccurate for students from different schools or countries.\n"
        "- Do not use predictions to label or make high-stakes decisions about students."
    )
