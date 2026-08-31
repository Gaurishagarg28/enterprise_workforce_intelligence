from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent_graph import build_workforce_graph
from src.agentic_workforce import WorkforceOrchestrator
from src.llm_agent import WorkforceLLMAgent
from src.train_model import MODEL_FEATURES, train_attrition_model
from src.transformer_matcher import semantic_skill_match

st.set_page_config(page_title="Enterprise Workforce Intelligence", page_icon="🧠", layout="wide")

DATA_PATH = Path("data/raw/employee_attrition.csv")

st.title("🧠 Enterprise Workforce Intelligence")
st.caption("From workforce risk prediction to explainable retain / reskill decisions")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def get_model(df):
    return train_attrition_model(df)

@st.cache_resource
def get_graph():
    try:
        return build_workforce_graph()
    except Exception:
        return None

if not DATA_PATH.exists():
    st.error("data/raw/employee_attrition.csv was not found.")
    st.stop()

df = load_data()
model, metrics = get_model(df)

probabilities = model.predict_proba(df[MODEL_FEATURES])[:, 1]
high_risk = int((probabilities >= 0.70).sum())
attrition_rate = float((df["Attrition"] == "Yes").mean() * 100)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{len(df):,}")
c2.metric("High Risk", f"{high_risk:,}")
c3.metric("Actual Attrition", f"{attrition_rate:.1f}%")
c4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Attrition by Department")
    rates = (df.assign(flag=(df["Attrition"] == "Yes").astype(int)).groupby("Department")["flag"].mean() * 100).sort_values(ascending=False)
    st.bar_chart(rates)
with right:
    st.subheader("Model Performance")
    st.write({k.upper(): round(v, 3) for k, v in metrics.items()})
    st.caption("Stratified 20% holdout; class-balanced Logistic Regression.")

st.divider()
st.header("Agentic Employee Intelligence")

selected_id = st.selectbox("Select EmployeeNumber", df["EmployeeNumber"].tolist())
employee = df[df["EmployeeNumber"] == selected_id].iloc[[0]].copy()
probability = float(model.predict_proba(employee[MODEL_FEATURES])[:, 1][0])
role = str(employee["JobRole"].iloc[0])

# Current skill inventory is deliberately derived from observed fields, not fabricated employee certifications.
current_skills = []
if role in {"Research Scientist", "Research Director"}:
    current_skills += ["Research", "Statistics"]
if "Sales" in role:
    current_skills += ["Communication", "CRM"]
if "Manager" in role or "Director" in role:
    current_skills += ["Leadership", "Communication"]
if "OverTime" in employee and employee["OverTime"].iloc[0] == "No":
    current_skills += ["Workload Management"]

profile = {
    "role": role,
    "current_skills": sorted(set(current_skills)),
}

# The deterministic core owns decisions; LangGraph is the orchestration layer.
graph = get_graph()
if graph:
    try:
        result = graph.invoke({"model": model, "employee": employee, "features": MODEL_FEATURES, "role": role, "current_skills": current_skills})["intelligence"]
    except Exception:
        result = WorkforceOrchestrator().run(model, employee, MODEL_FEATURES, role, current_skills)
else:
    result = WorkforceOrchestrator().run(model, employee, MODEL_FEATURES, role, current_skills)

r1, r2, r3 = st.columns(3)
r1.metric("Attrition Risk", f"{probability * 100:.1f}%")
r2.metric("Risk Level", result["attrition"]["risk_level"])
r3.metric("Recommended Action", result["recommendation"]["decision"])

st.subheader("Why this employee needs attention")
sc1, sc2 = st.columns(2)
with sc1:
    st.write("**Required skills**")
    st.write(", ".join(result["skills"]["required_skills"]) or "None")
    st.write("**Current inferred skills**")
    st.write(", ".join(result["skills"]["current_skills"]) or "No validated skill record")
with sc2:
    st.metric("Readiness", f"{result['skills']['readiness']:.1f}%")
    st.write("**Skill gap**")
    st.write(", ".join(result["skills"]["skill_gap"]) or "No gap identified")
    st.write("**Actions**")
    for action in result["recommendation"]["actions"]:
        st.write(f"• {action}")

st.subheader("Transformer Skill Matching")
query = st.text_input("Try a target skill", value=(result["skills"]["skill_gap"][0] if result["skills"]["skill_gap"] else "Machine Learning"))
catalog = sorted({s for values in __import__("src.skill_engine", fromlist=["DEFAULT_ROLE_SKILLS"]).DEFAULT_ROLE_SKILLS.values() for s in values})
if st.button("Run semantic match"):
    try:
        matches = semantic_skill_match(query, catalog)
        st.dataframe(pd.DataFrame(matches), hide_index=True, width="stretch")
    except Exception as exc:
        st.warning(f"Transformer model unavailable in this environment: {exc}")
        st.info("The business decision layer remains deterministic and usable without the transformer download.")

st.subheader("LLM Decision Explanation")
llm = WorkforceLLMAgent()
if llm.enabled:
    if st.button("Explain decision with LLM"):
        st.write(llm.explain_decision(result))
else:
    st.info("Set OPENAI_API_KEY to enable the LLM explanation agent. Core scoring does not depend on an LLM.")

with st.expander("Agent execution trace"):
    st.json(result)

st.caption("Human-in-the-loop: this prototype provides decision support, not automated employment decisions. Sensitive demographic attributes are not used by the recommendation layer.")
