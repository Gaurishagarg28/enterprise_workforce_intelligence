from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


RAW = Path("data/raw")
EXTERNAL = Path("data/external")

# Fallback only: used when supplied O*NET source files are not present locally.
DEFAULT_ROLE_SKILLS = {
    "Sales Executive": {"Communication", "CRM", "Negotiation", "Analytics"},
    "Research Scientist": {"Python", "Statistics", "Machine Learning", "Experimentation"},
    "Laboratory Technician": {"Quality Control", "Documentation", "Data Analysis"},
    "Manufacturing Director": {"Operations", "Leadership", "Quality", "Analytics"},
    "Healthcare Representative": {"Communication", "Domain Knowledge", "CRM"},
    "Manager": {"Leadership", "Communication", "Analytics"},
    "Research Director": {"Leadership", "Research", "Statistics", "Machine Learning"},
    "Human Resources": {"HR Operations", "Communication", "Analytics", "Employee Relations"},
    "default": {"Communication", "Analytics", "Problem Solving"},
}


def _norm(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _load_onet() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    occ_path = RAW / "occupation_data.csv"
    skill_path = RAW / "essential_skills.csv"
    software_path = RAW / "software_skills.csv"
    if not (occ_path.exists() and skill_path.exists() and software_path.exists()):
        return None, None, None
    return pd.read_csv(occ_path), pd.read_csv(skill_path), pd.read_csv(software_path)


def _onet_role_match(role: str, occupations: pd.DataFrame) -> pd.DataFrame:
    target = _norm(role)
    exact = occupations[occupations["Title"].map(_norm) == target]
    if not exact.empty:
        return exact
    # Conservative fallback: title containment, never semantic guessing.
    return occupations[occupations["Title"].map(_norm).str.contains(target, regex=False)]


def _onet_required_skills(role: str) -> set[str]:
    occupations, essential, _ = _load_onet()
    if occupations is None or essential is None:
        return set(DEFAULT_ROLE_SKILLS.get(role, DEFAULT_ROLE_SKILLS["default"]))

    matches = _onet_role_match(role, occupations)
    if matches.empty:
        return set(DEFAULT_ROLE_SKILLS.get(role, DEFAULT_ROLE_SKILLS["default"]))

    codes = set(matches["O*NET-SOC Code"].astype(str))
    rows = essential[essential["O*NET-SOC Code"].astype(str).isin(codes)].copy()
    rows = rows[rows["Scale ID"].eq("IM")]
    if rows.empty:
        return set(DEFAULT_ROLE_SKILLS.get(role, DEFAULT_ROLE_SKILLS["default"]))

    # Top 8 by importance gives a stable, explainable role benchmark.
    rows = rows.sort_values("Data Value", ascending=False)
    return set(rows["Element Name"].dropna().astype(str).head(8))


def required_skills_for_role(role: str) -> set[str]:
    return _onet_required_skills(role)


def calculate_skill_gap(required: Iterable[str], current: Iterable[str]) -> list[str]:
    current_norm = {_norm(x) for x in current}
    return sorted([x for x in set(required) if _norm(x) not in current_norm])


def readiness_score(required: Iterable[str], current: Iterable[str]) -> float:
    required_set = set(required)
    if not required_set:
        return 100.0
    current_norm = {_norm(x) for x in current}
    matched = sum(_norm(x) in current_norm for x in required_set)
    return round(matched / len(required_set) * 100, 1)


def organization_skill_gaps(records: Iterable[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for record in records:
        for skill in record.get("skill_gap", []):
            counts[skill] = counts.get(skill, 0) + 1

    return [
        {
            "skill": skill,
            "employees_missing": count,
            "severity": "HIGH" if count >= 100 else "MEDIUM" if count >= 50 else "LOW",
        }
        for skill, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]
