from __future__ import annotations

from typing import Iterable


# Controlled MVP role taxonomy. These are role-level requirements, not claims about an
# individual's certified skills. Employee skill inventory should replace inferred skills
# when validated employee-level skill data is available.
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


def required_skills_for_role(role: str) -> set[str]:
    return set(DEFAULT_ROLE_SKILLS.get(role, DEFAULT_ROLE_SKILLS["default"]))


def calculate_skill_gap(required: Iterable[str], current: Iterable[str]) -> list[str]:
    return sorted(set(required) - set(current))


def readiness_score(required: Iterable[str], current: Iterable[str]) -> float:
    required_set = set(required)
    if not required_set:
        return 100.0
    current_set = set(current)
    matched = len(required_set.intersection(current_set))
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
