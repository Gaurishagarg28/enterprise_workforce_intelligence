from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent_graph import build_workforce_graph
from src.llm_agent import WorkforceLLMAgent
from src.rag_engine import retrieve
from src.skill_engine import DEFAULT_ROLE_SKILLS
from src.train_model import MODEL_FEATURES, train_attrition_model
from src.transformer_matcher import semantic_skill_match

st.set_page_config(page_title="Enterprise Workforce Intelligence", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

DATA_PATH = Path("data/raw/employee_attrition.csv")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 1rem 0 1.5rem 0;}
.hero h1 {margin-bottom: 0.25rem;}
.hero p {font-size: 1.05rem; margin-top: 0;}
</style>
""", unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>🧠 Enterprise Workforce Intelligence</h1><p>From workforce risk to capability preservation: retain, reskill, or hire.</p></div>", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error("Dataset not found at data/raw/employee_attrition.csv")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def get_model(df):
    return train_attrition_model(df)

@st.cache_resource
def get_graph():
    return build_workforce_graph()

df = load_data()
model, metrics = get_model(df)
graph = get_graph()
probabilities = model.predict_proba(df[MODEL_FEATURES])[:, 1]
high_risk_count = int((probabilities >= 0.70).sum())
medium_risk_count = int(((probabilities >= 0.40) & (probabilities < 0.70)).sum())
actual_attrition_rate = float((df["Attrition"] == "Yes").mean() * 100)
role_catalog = sorted(df["JobRole"].dropna().unique().tolist())

with st.sidebar:
    st.header("Workforce Controls")
    selected_department = st.selectbox("Department", ["All"] + sorted(df["Department"].dropna().unique().tolist()))
    selected_role = st.selectbox("Role", ["All"] + role_catalog)
    risk_threshold = st.slider("High-risk threshold", 0.50, 0.90, 0.70, 0.05)

filtered = df.copy()
if selected_department != "All": filtered = filtered[filtered["Department"] == selected_department]
if selected_role != "All": filtered = filtered[filtered["JobRole"] == selected_role]
filtered_high_risk = int((model.predict_proba(filtered[MODEL_FEATURES])[:, 1] >= risk_threshold).sum())

st.subheader("Executive Workforce Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{len(filtered):,}")
c2.metric("High-Risk Employees", f"{filtered_high_risk:,}")
c3.metric("Actual Attrition", f"{actual_attrition_rate:.1f}%")
c4.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}")
st.caption(f"Class-balanced Logistic Regression | Holdout ROC-AUC: {metrics['roc_auc']:.3f} | Global high-risk count at 70%: {high_risk_count}")
st.divider()

col1, col2 = st.columns([1.15, 0.85])
with col1:
    st.subheader("Where is workforce risk concentrated?")
    department_risk = (df.assign(AttritionFlag=(df["Attrition"] == "Yes").astype(int)).groupby("Department")["AttritionFlag"].mean().mul(100).sort_values(ascending=False))
    st.bar_chart(department_risk)
with col2:
    st.subheader("Risk mix")
    st.bar_chart(pd.Series({"High": high_risk_count, "Medium": medium_risk_count, "Low": len(df) - high_risk_count - medium_risk_count}))

st.divider()
st.header("Employee Decision Center")
employee_id = st.selectbox("EmployeeNumber", df["EmployeeNumber"].tolist())
employee = df[df["EmployeeNumber"] == employee_id].iloc[[0]].copy()
role = str(employee["JobRole"].iloc[0])

current_skills = []
if role in {"Research Scientist", "Research Director"}: current_skills += ["Research", "Statistics"]
if "Sales" in role: current_skills += ["Communication", "CRM"]
if "Manager" in role or "Director" in role: current_skills += ["Leadership", "Communication"]
if employee["OverTime"].iloc[0] == "No": current_skills += ["Workload Management"]

with st.spinner("Running workforce agents..."):
    result = graph.invoke({"model": model, "employee": employee, "features": MODEL_FEATURES, "role": role, "current_skills": current_skills})["intelligence"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Attrition Risk", f"{result['attrition']['probability'] * 100:.1f}%")
m2.metric("Risk Level", result["attrition"]["risk_level"])
m3.metric("Reskill Readiness", f"{result['skills']['readiness']:.1f}%")
m4.metric("Recommended Action", result["recommendation"]["decision"])

left, right = st.columns(2)
with left:
    st.subheader("Capability Profile")
    st.write(f"**Current role:** {role}")
    st.write("**Required capabilities:**")
    st.write(", ".join(result["skills"]["required_skills"]))
    st.write("**MVP inferred capabilities:**")
    st.write(", ".join(result["skills"]["current_skills"]) or "No validated employee skill record")
with right:
    st.subheader("Workforce Decision")
    st.write(f"**Capability gap:** {', '.join(result['skills']['skill_gap']) or 'No gap identified'}")
    st.write("**Suggested actions:**")
    for action in result["recommendation"]["actions"]: st.write(f"• {action}")

st.divider()
st.subheader("Transformer Skill Intelligence")
query = st.text_input("Search for a related capability", value=(result["skills"]["skill_gap"][0] if result["skills"]["skill_gap"] else "Machine Learning"))
catalog = sorted({skill for skills in DEFAULT_ROLE_SKILLS.values() for skill in skills})
if st.button("Find Semantically Related Skills"):
    try:
        matches = semantic_skill_match(query, catalog)
        if matches: st.dataframe(pd.DataFrame(matches), hide_index=True, width="stretch")
        else: st.info("No semantic matches exceeded the configured similarity threshold.")
    except Exception as exc:
        st.warning(f"Transformer model unavailable: {exc}")

st.subheader("Grounded HR Knowledge")
knowledge_query = st.text_input("Policy question", value="reskilling high risk employee")
knowledge = retrieve(knowledge_query)
if knowledge:
    for source in knowledge: st.info(f"**{source['source']}** — {source['text']}")
else: st.caption("No matching policy knowledge found.")

st.subheader("LLM Explanation")
llm = WorkforceLLMAgent()
if st.button("Generate Grounded Explanation"):
    with st.spinner("Generating explanation..."):
        st.markdown(llm.explain_decision(result))
    if not llm.enabled:
        st.caption("The local deterministic fallback was used because the OpenAI client is unavailable.")

with st.expander("Agent trace / audit evidence"):
    st.json(result)

st.caption("Responsible AI: recommendations are decision-support signals. Human HR review is required before action.")
