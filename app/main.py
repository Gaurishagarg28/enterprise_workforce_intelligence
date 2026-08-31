from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.agent_graph import build_workforce_graph
from src.rag_engine import retrieve
from src.train_model import MODEL_FEATURES, train_attrition_model

app = FastAPI(title="Enterprise Workforce Intelligence API", version="1.2.0")
DATA_PATH = "data/raw/employee_attrition.csv"
df = pd.read_csv(DATA_PATH)
model, metrics = train_attrition_model(df)
graph = build_workforce_graph()


class EmployeeRequest(BaseModel):
    employee_id: int = Field(..., gt=0)
    target_role: str | None = None
    current_skills: list[str] = Field(default_factory=list)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "employees": len(df),
        "model": "class-balanced-logistic-regression",
        "orchestration": "langgraph",
        "agents": ["attrition_agent", "skill_gap_agent", "recommendation_agent"],
    }


@app.get("/dashboard/summary")
def dashboard_summary():
    probabilities = model.predict_proba(df[MODEL_FEATURES])[:, 1]
    return {
        "total_employees": len(df),
        "actual_attrition_rate": round((df["Attrition"] == "Yes").mean() * 100, 2),
        "predicted_high_risk": int((probabilities >= 0.70).sum()),
        "model_metrics": metrics,
    }


@app.get("/knowledge")
def knowledge(query: str):
    return {"query": query, "sources": retrieve(query)}


@app.post("/employee/intelligence")
def employee_intelligence(request: EmployeeRequest):
    rows = df[df["EmployeeNumber"] == request.employee_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Employee not found")

    employee = rows.iloc[[0]]
    role = request.target_role or str(employee["JobRole"].iloc[0])
    result = graph.invoke({
        "model": model,
        "employee": employee,
        "features": MODEL_FEATURES,
        "role": role,
        "current_skills": request.current_skills,
    })
    return result["intelligence"]
