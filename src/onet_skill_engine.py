from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


RAW = Path("data/raw")


def load_occupation_data(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or RAW / "occupation_data.csv")


def load_software_skills(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or RAW / "software_skills.csv")


def load_essential_skills(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or RAW / "essential_skills.csv")


def _normalise(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def find_role(title: str, occupations: pd.DataFrame) -> pd.DataFrame:
    target = _normalise(title)
    exact = occupations[occupations["Title"].map(_normalise) == target]
    if not exact.empty:
        return exact
    contains = occupations[occupations["Title"].map(_normalise).str.contains(target, regex=False)]
    return contains


def role_software_skills(title: str, software: pd.DataFrame, top_k: int = 15) -> list[dict]:
    target = _normalise(title)
    rows = software[software["Title"].map(_normalise) == target].copy()
    if rows.empty:
        rows = software[software["Title"].map(_normalise).str.contains(target, regex=False)].copy()
    if rows.empty:
        return []

    rows["hot"] = rows["Hot Technology"].eq("Y")
    rows["demand"] = rows["In Demand"].eq("Y")
    rows = rows.sort_values(["demand", "hot", "Workplace Example"], ascending=[False, False, True])
    return rows[["Workplace Example", "Element Name", "Hot Technology", "In Demand"]].head(top_k).to_dict("records")


def role_essential_skills(title: str, essential: pd.DataFrame, top_k: int = 15) -> list[dict]:
    target = _normalise(title)
    rows = essential[essential["Title"].map(_normalise) == target].copy()
    if rows.empty:
        rows = essential[essential["Title"].map(_normalise).str.contains(target, regex=False)].copy()
    if rows.empty:
        return []

    # Keep the importance score where available. LV is a level score; IM is importance.
    importance = rows[rows["Scale ID"].eq("IM")].copy()
    if importance.empty:
        importance = rows.copy()
    importance = importance.sort_values("Data Value", ascending=False)
    return importance[["Element ID", "Element Name", "Scale ID", "Scale Name", "Data Value"]].head(top_k).to_dict("records")


def semantic_role_text(title: str, occupations: pd.DataFrame, software: pd.DataFrame, essential: pd.DataFrame) -> str:
    occ = find_role(title, occupations)
    description = occ["Description"].iloc[0] if not occ.empty else ""
    software_names = [x["Workplace Example"] for x in role_software_skills(title, software, 10)]
    skill_names = [x["Element Name"] for x in role_essential_skills(title, essential, 10)]
    return " | ".join([
        f"Role: {title}",
        f"Description: {description}",
        f"Technologies: {', '.join(software_names)}",
        f"Essential skills: {', '.join(skill_names)}",
    ])


def skill_gap_from_taxonomy(required: Iterable[str], current: Iterable[str]) -> list[str]:
    current_norm = {_normalise(x) for x in current}
    return sorted([skill for skill in required if _normalise(skill) not in current_norm])
