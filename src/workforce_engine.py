from __future__ import annotations

from typing import Any

import pandas as pd

from src.skill_engine import calculate_skill_gap, readiness_score, required_skills_for_role


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_employee_profile(row: pd.Series) -> dict[str, Any]:
    role = str(row.get("JobRole", row.get("Title", "Unknown")))
    department = str(row.get("Department", row.get("DepartmentType", "Unknown")))

    required = required_skills_for_role(role)
    # Controlled MVP skill inventory: transparently inferred only from fields present in the employee records.
    current: set[str] = set()
    if row.get("Training Hours", row.get("TrainingHours")) not in (None, ""):
        current.add("Training")
    if _safe_float(row.get("KPI Score")) is not None or _safe_float(row.get("CustomerSatisfaction")) is not None:
        current.add("Analytics")
    if str(row.get("JobRole", "")) in {"Research Scientist", "Research Director"}:
        current.update({"Research", "Statistics"})
    if "Sales" in department or "Sales" in role:
        current.update({"Communication", "CRM"})
    if "Manager" in role or "Director" in role:
        current.update({"Leadership", "Communication"})

    gap = calculate_skill_gap(required, current)
    readiness = readiness_score(required, current)

    return {
        "employee_id": row.get("EmployeeID", row.get("EmployeeNumber")),
        "role": role,
        "department": department,
        "attrition": row.get("Attrition"),
        "job_satisfaction": _safe_float(row.get("JobSatisfaction", row.get("Satisfaction Score"))),
        "engagement": _safe_float(row.get("Engagement Score", row.get("EngagementScore"))),
        "performance": row.get("Performance Score", row.get("PerformanceRating")),
        "current_skills": sorted(current),
        "required_skills": sorted(required),
        "skill_gap": gap,
        "readiness": readiness,
    }


def decision_from_intelligence(profile: dict[str, Any], attrition_probability: float) -> dict[str, Any]:
    readiness = float(profile["readiness"])
    gap_size = len(profile["skill_gap"])

    if attrition_probability >= 0.70 and readiness >= 60:
        decision = "RESKILL"
        reason = "High retention risk with sufficient reskilling readiness."
    elif attrition_probability >= 0.70:
        decision = "RETAIN"
        reason = "High retention risk; prioritize retention before capability transition."
    elif gap_size > 0 and readiness >= 70:
        decision = "RESKILL"
        reason = "The employee has a material skill gap but strong readiness for development."
    else:
        decision = "RETAIN"
        reason = "Current signals do not justify an immediate external hiring decision."

    return {
        "decision": decision,
        "reason": reason,
        "attrition_probability": round(attrition_probability, 4),
        "skill_gap_count": gap_size,
        "readiness": readiness,
    }
