import streamlit as st
import pandas as pd
from pathlib import Path

from src.train_model import train_attrition_model, MODEL_FEATURES

st.set_page_config(page_title="Enterprise Workforce Intelligence", page_icon="👥", layout="wide")

DATA_PATH = Path("data/raw/employee_attrition.csv")

st.title("AI Workforce Intelligence Platform")
st.caption("Attrition prediction and workforce analytics MVP")

if not DATA_PATH.exists():
    st.error("Dataset not found: data/raw/employee_attrition.csv")
    st.info("Upload employee_attrition.csv to data/raw/ in the repository, then redeploy.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def get_model(df):
    return train_attrition_model(df)

df = load_data(DATA_PATH)
model, metrics = get_model(df)

attrition_count = int((df["Attrition"] == "Yes").sum())
attrition_rate = attrition_count / len(df) * 100
avg_satisfaction = df["JobSatisfaction"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{len(df):,}")
c2.metric("High Attrition Risk Cases", f"{attrition_count:,}")
c3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
c4.metric("Avg Job Satisfaction", f"{avg_satisfaction:.2f}/4")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Attrition Risk by Department")
    department_risk = (
        df.assign(AttritionFlag=(df["Attrition"] == "Yes").astype(int))
        .groupby("Department")["AttritionFlag"]
        .mean().mul(100).sort_values(ascending=False)
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
st.caption("Metrics are calculated on a stratified holdout test set. Accuracy is not used alone because the target is imbalanced.")

st.divider()

st.subheader("Employee Risk Scoring")

identifier_column = "EmployeeNumber"
display_columns = [c for c in [identifier_column, "Age", "Department", "JobRole", "OverTime", "JobSatisfaction", "MonthlyIncome", "YearsAtCompany"] if c in df.columns]

employee_options = df[identifier_column].tolist()
selected_id = st.selectbox("Select EmployeeNumber", employee_options)
selected = df[df[identifier_column] == selected_id].iloc[[0]].copy()

risk_probability = float(model.predict_proba(selected[MODEL_FEATURES])[:, 1][0])
risk_percent = risk_probability * 100

if risk_percent >= 70:
    risk_label = "High"
elif risk_percent >= 40:
    risk_label = "Medium"
else:
    risk_label = "Low"

r1, r2, r3 = st.columns(3)
r1.metric("Predicted Attrition Risk", f"{risk_percent:.1f}%")
r2.metric("Risk Level", risk_label)
r3.metric("Actual Attrition", str(selected["Attrition"].iloc[0]))

st.dataframe(selected[display_columns], use_container_width=True, hide_index=True)

st.divider()

st.subheader("Current Scope")
st.write(
    "This MVP currently implements validated IBM HR data, attrition analytics, a class-balanced logistic regression model, "
    "employee-level risk scoring, and an interactive dashboard. Skill-gap analysis, upskilling recommendations, and additional "
    "HR datasets should be added only after their join keys and semantics are validated."
)
