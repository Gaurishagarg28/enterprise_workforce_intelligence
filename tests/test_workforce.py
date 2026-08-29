import pandas as pd

from src.agentic_workforce import WorkforceOrchestrator
from src.skill_engine import calculate_skill_gap, readiness_score
from src.train_model import MODEL_FEATURES, train_attrition_model


def test_skill_gap():
    assert calculate_skill_gap(["Python", "MLOps"], ["Python"]) == ["MLOps"]


def test_readiness():
    assert readiness_score(["Python", "MLOps"], ["Python"]) == 50.0


def test_model_probability():
    df = pd.read_csv("data/raw/employee_attrition.csv")
    model, metrics = train_attrition_model(df)
    probability = float(model.predict_proba(df.iloc[[0]][MODEL_FEATURES])[:, 1][0])
    assert 0.0 <= probability <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_orchestrator_output():
    df = pd.read_csv("data/raw/employee_attrition.csv")
    model, _ = train_attrition_model(df)
    result = WorkforceOrchestrator().run(
        model,
        df.iloc[[0]],
        MODEL_FEATURES,
        "Manager",
        ["Communication"],
    )
    assert "attrition" in result
    assert "skills" in result
    assert "recommendation" in result
