import streamlit as st
import pandas as pd
from pathlib import Path

from src.train_model import train_attrition_model, MODEL_FEATURES

st.set_page_config(
    page_title="Enterprise Workforce Intelligence",
    page_icon="👥",
    layout="wide",
)

DATA_PATHS = [
    Path("data/raw/employee_attrition.csv"),
    Path("employee_attrition.csv"),
]

st.title("AI Workforce Intelligence Platform")
st.caption("Attrition prediction and workforce analytics MVP")


@st.cache_data
def load_data():
    for path in DATA_PATHS:
        if path.exists():
            return pd.read_csv(path), str(path)
    return None, None


@st.cache_resource
def get_model(df):
    return train_attrition_model(df)


df, loaded_from = load_data()

if df is None:
    st.error("Dataset not found.")
    st.write("For local/demo use, place `employee_attrition.csv` in `data/raw/`.")
    st.write("For Render, upload the permitted demo CSV as a repository file named `employee_attrition.csv` or provide an equivalent deployment data source.")
    st.stop()

model, metrics = get_model(df)

attrition_count = int((df["Attrition"] == "Yes").sum())
attrition_rate = attrition_count / len(df) * 100
avg_satisfaction = df["JobSatisfaction"].mean()

# Score all employees so the dashboard can show predicted high-risk employees.
all_probabilities = model.predict_proba(df[MODEL_FEATURES])[:, 1]

def risk_label(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"

risk_labels = pd.Series(all_probabilities).map(risk_label)
high_risk_count = int((risk_labels == "High").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{len(df):,}")
c2.metric("Predicted High Risk", f"{high_risk_count:,}")
c3.metric("Actual Attrition Rate", f"{attrition_rate:.1f}%")
c4.metric("Avg Job Satisfaction", f"{avg_satisfaction:.2f}/4")

st.caption(f"Data loaded from: `{loaded_from}`")
st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Actual Attrition Rate by Department")
    department_risk = (
        df.assign(AttritionFlag=(df["Attrition"] == "Yes").astype(int))
        .groupby("Department")["AttritionFlag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    st.bar_chart(department_risk)

with right:
    st.subheader("Actual Attrition Distribution")
    st.bar_chart(df["Attrition"].value_counts())

st.divider()

st.subheader("Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
m2.metric("Precision", f"{metrics['precision']:.3f}")
m3.metric("Recall", f"{metrics['recall']:.3f}")
m4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
st.caption("Metrics use a stratified 20% holdout test set. The model uses class-balanced Logistic Regression because attrition is imbalanced.")

st.divider()

st.subheader("Employee Risk Scoring")
identifier_column = "EmployeeNumber"
display_columns = [
    c for c in [
        identifier_column,
        "Age",
        "Department",
        "JobRole",
        "OverTime",
        "JobSatisfaction",
        "MonthlyIncome",
        "YearsAtCompany",
    ] if c in df.columns
]

employee_options = df[identifier_column].tolist()
selected_id = st.selectbox("Select EmployeeNumber", employee_options)
selected = df[df[identifier_column] == selected_id].iloc[[0]].copy()

risk_probability = float(model.predict_proba(selected[MODEL_FEATURES])[:, 1][0])
risk_percent = risk_probability * 100
risk = risk_label(risk_probability)

r1, r2, r3 = st.columns(3)
r1.metric("Predicted Attrition Risk", f"{risk_percent:.1f}%")
r2.metric("Risk Level", risk)
r3.metric("Actual Attrition", str(selected["Attrition"].iloc[0]))

st.dataframe(selected[display_columns], use_container_width=True, hide_index=True)

st.divider()

st.subheader("Current MVP Scope")
st.write(
    "This MVP implements validated IBM HR data, attrition analytics, a class-balanced Logistic Regression model, "
    "employee-level risk scoring, and an interactive dashboard. Skill-gap analysis, personalized upskilling, career intelligence, "
    "RAG, and agentic workflows are planned extensions and should be added only after their supporting data and business rules are validated."
)
