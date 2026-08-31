from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .skill_engine import calculate_skill_gap, readiness_score, required_skills_for_role


@dataclass
class AgentResult:
    agent: str
    status: str
    data: dict[str, Any]


class AttritionAgent:
    name = "attrition_agent"

    def run(self, model, employee: pd.DataFrame, features: list[str]) -> AgentResult:
        probability = float(model.predict_proba(employee[features])[:, 1][0])
        level = "HIGH" if probability >= 0.70 else "MEDIUM" if probability >= 0.40 else "LOW"
        return AgentResult(self.name, "success", {
            "probability": probability,
            "risk_level": level,
            "evidence": "Class-balanced Logistic Regression probability on validated HR features.",
        })


class SkillGapAgent:
    name = "skill_gap_agent"

    def run(self, role: str, current_skills: list[str]) -> AgentResult:
        required = required_skills_for_role(role)
        gap = calculate_skill_gap(required, current_skills)
        readiness = readiness_score(required, current_skills)
        return AgentResult(self.name, "success", {
            "role": role,
            "required_skills": sorted(required),
            "current_skills": sorted(set(current_skills)),
            "skill_gap": gap,
            "readiness": readiness,
            "evidence": "Role requirements from the controlled MVP capability taxonomy; current skills are explicitly labelled inferred unless validated employee-skill data is supplied.",
        })


class RecommendationAgent:
    name = "recommendation_agent"

    def run(self, skill_gap: list[str], attrition_level: str, readiness: float) -> AgentResult:
        actions = []
        if skill_gap:
            actions.append(f"Upskill in {skill_gap[0]}")
            if len(skill_gap) > 1:
                actions.append(f"Build capability in {skill_gap[1]}")
        if attrition_level == "HIGH":
            actions.append("Prioritize retention conversation")

        if attrition_level == "HIGH" and readiness >= 70:
            decision = "RETAIN_AND_RESKILL"
        elif readiness >= 50:
            decision = "RESKILL"
        else:
            decision = "DEVELOP_OR_HIRE"

        return AgentResult(self.name, "success", {
            "decision": decision,
            "actions": actions,
            "evidence": "Transparent rule-based combination of attrition level, skill gap and readiness; requires human review.",
        })


class WorkforceOrchestrator:
    """Deterministic specialist agents; LangGraph provides workflow orchestration."""

    def __init__(self):
        self.attrition = AttritionAgent()
        self.skill_gap = SkillGapAgent()
        self.recommendation = RecommendationAgent()

    def run(self, model, employee: pd.DataFrame, features: list[str], role: str, current_skills: list[str]) -> dict[str, Any]:
        attrition = self.attrition.run(model, employee, features)
        skills = self.skill_gap.run(role, current_skills)
        recommendation = self.recommendation.run(
            skills.data["skill_gap"], attrition.data["risk_level"], skills.data["readiness"]
        )
        return {
            "employee_id": int(employee["EmployeeNumber"].iloc[0]),
            "attrition": attrition.data,
            "skills": skills.data,
            "recommendation": recommendation.data,
            "agents_executed": [attrition.agent, skills.agent, recommendation.agent],
        }
