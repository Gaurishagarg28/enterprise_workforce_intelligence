from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.agentic_workforce import WorkforceOrchestrator
from src.rag_engine import retrieve
from src.train_model import MODEL_FEATURES, train_attrition_model

app = FastAPI(title="Enterprise Workforce Intelligence API", version="1.1.0")
DATA_PATH = "data/raw/employee_attrition.csv"
df = pd.read_csv(DATA_PATH)
model, metrics = train_attrition_model(df)
orchestrator = WorkforceOrchestrator()

class EmployeeRequest(BaseModel):
    employee_id: int = Field(..., gt=0)
    target_role: str | None = None
    current_skills: list[str] = Field(default_factory=list)

@app.get("/health")
def health():
    return {"status": "ok", "employees": len(df), "model": "class-balanced-logistic-regression", "agentic": True}

@app.get("/dashboard/summary")
def dashboard_summary():
    probabilities = model.predict_proba(df[MODEL_FEATURES])[:, 1]
    return {"total_employees": len(df), "actual_attrition_rate": round((df["Attrition"] == "Yes").mean() * 100, 2), "predicted_high_risk": int((probabilities >= 0.70).sum()), "model_metrics": metrics}

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
    return orchestrator.run(model, employee, MODEL_FEATURES, role, request.current_skills)
