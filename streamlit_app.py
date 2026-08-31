from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent_graph import build_workforce_graph
from src.llm_agent import WorkforceLLMAgent
from src.rag_engine import retrieve
from src.train_model import MODEL_FEATURES, train_attrition_model
from src.transformer_matcher import semantic_skill_match
from src.skill_engine import DEFAULT_ROLE_SKILLS

st.set_page_config(page_title="Enterprise Workforce Intelligence", page_icon="🧠", layout="wide")

DATA_PATH = Path("data/raw/employee_attrition.csv")

st.title("🧠 Enterprise Workforce Intelligence")
st.caption("Predict risk → identify capability gaps → recommend retain/reskill actions")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def get_model(df):
    return train_attrition_model(df)

@st.cache_resource
def get_graph():
    return build_workforce_graph()

if not DATA_PATH.exists():
    st.error("data/raw/employee_attrition.csv was not found.")
    st.stop()

df = load_data()
model, metrics = get_model(df)
graph = get_graph()

probabilities = model.predict_proba(df[MODEL_FEATURES])[:, 1]
high_risk = int((probabilities >= 0.70).sum())
attrition_rate = float((df["Attrition"] == "Yes").mean() * 100)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{len(df):,}")
c2.metric("Predicted High Risk", f"{high_risk:,}")
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
st.header("Agentic Workforce Intelligence")

selected_id = st.selectbox("Select EmployeeNumber", df["EmployeeNumber"].tolist())
employee = df[df["EmployeeNumber"] == selected_id].iloc[[0]].copy()
role = str(employee["JobRole"].iloc[0])

current_skills = []
if role in {"Research Scientist", "Research Director"}:
    current_skills += ["Research", "Statistics"]
if "Sales" in role:
    current_skills += ["Communication", "CRM"]
if "Manager" in role or "Director" in role:
    current_skills += ["Leadership", "Communication"]
if employee["OverTime"].iloc[0] == "No":
    current_skills += ["Workload Management"]

with st.spinner("Running specialist agents..."):
    result = graph.invoke({
        "model": model,
        "employee": employee,
        "features": MODEL_FEATURES,
        "role": role,
        "current_skills": current_skills,
    })["intelligence"]

r1, r2, r3, r4 = st.columns(4)
r1.metric("Attrition Risk", f"{result['attrition']['probability'] * 100:.1f}%")
r2.metric("Risk Level", result["attrition"]["risk_level"])
r3.metric("Readiness", f"{result['skills']['readiness']:.1f}%")
r4.metric("Recommended Action", result["recommendation"]["decision"])

st.subheader("Decision Evidence")
sc1, sc2 = st.columns(2)
with sc1:
    st.write("**Role:**", role)
    st.write("**Required capabilities:**", ", ".join(result["skills"]["required_skills"]))
    st.write("**Inferred capabilities:**", ", ".join(result["skills"]["current_skills"]) or "No validated skill record")
with sc2:
    st.write("**Capability gap:**", ", ".join(result["skills"]["skill_gap"]) or "No gap identified")
    st.write("**Agent actions:**")
    for action in result["recommendation"]["actions"]:
        st.write(f"• {action}")

st.subheader("Transformer Semantic Skill Matching")
query = st.text_input("Target capability", value=(result["skills"]["skill_gap"][0] if result["skills"]["skill_gap"] else "Machine Learning"))
catalog = sorted({skill for skills in DEFAULT_ROLE_SKILLS.values() for skill in skills})
if st.button("Run Transformer Match"):
    try:
        matches = semantic_skill_match(query, catalog)
        st.dataframe(pd.DataFrame(matches), hide_index=True, width="stretch")
    except Exception as exc:
        st.warning(f"Transformer model unavailable: {exc}")

st.subheader("Grounded Workforce Knowledge")
query = st.text_input("Ask about the workforce policy", value="reskilling high risk employee")
sources = retrieve(query)
if sources:
    for source in sources:
        st.info(f"**{source['source']}** — {source['text']}")
else:
    st.caption("No matching knowledge chunks found.")

st.subheader("LLM Decision Explanation")
llm = WorkforceLLMAgent()
if llm.enabled:
    if st.button("Explain with LLM"):
        st.write(llm.explain_decision(result))
else:
    st.info("Set OPENAI_API_KEY and OPENAI_MODEL to enable grounded LLM explanation. Core scoring remains deterministic.")

with st.expander("Agent execution trace"):
    st.json(result)

st.caption("Human-in-the-loop: this is decision support, not an autonomous employment decision system.")
