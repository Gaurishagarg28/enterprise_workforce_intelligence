from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.agentic_workforce import WorkforceOrchestrator
from src.train_model import MODEL_FEATURES, train_attrition_model

app = FastAPI(title="Enterprise Workforce Intelligence API", version="1.0.0")

DATA_PATH = "data/raw/employee_attrition.csv"
df = pd.read_csv(DATA_PATH)
model, metrics = train_attrition_model(df)
orchestrator = WorkforceOrchestrator()


class EmployeeRequest(BaseModel):
    employee_id: int = Field(..., gt=0)
    target_role: str = Field(..., min_length=1)
    current_skills: list[str] = Field(default_factory=list)


@app.get("/health")
def health():
    return {"status": "ok", "employees": len(df), "model": "class-balanced-logistic-regression"}


@app.get("/dashboard/summary")
def dashboard_summary():
    return {
        "total_employees": len(df),
        "actual_attrition_rate": round((df["Attrition"] == "Yes").mean() * 100, 2),
        "model_metrics": metrics,
    }


@app.post("/employee/intelligence")
def employee_intelligence(request: EmployeeRequest):
    rows = df[df["EmployeeNumber"] == request.employee_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Employee not found")
    return orchestrator.run(model, rows.iloc[[0]], MODEL_FEATURES, request.target_role, request.current_skills)
