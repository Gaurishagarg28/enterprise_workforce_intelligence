from __future__ import annotations

from typing import Any, TypedDict

from .agentic_workforce import AttritionAgent, RecommendationAgent, SkillGapAgent


class WorkforceState(TypedDict, total=False):
    model: Any
    employee: Any
    features: list[str]
    role: str
    current_skills: list[str]
    attrition: dict[str, Any]
    skills: dict[str, Any]
    recommendation: dict[str, Any]
    intelligence: dict[str, Any]


def build_workforce_graph():
    """Build an explicit specialist-agent graph with deterministic business rules."""
    from langgraph.graph import END, START, StateGraph

    attrition_agent = AttritionAgent()
    skill_agent = SkillGapAgent()
    recommendation_agent = RecommendationAgent()

    def attrition_node(state: WorkforceState) -> WorkforceState:
        result = attrition_agent.run(state["model"], state["employee"], state["features"])
        return {"attrition": result.data}

    def skill_node(state: WorkforceState) -> WorkforceState:
        result = skill_agent.run(state["role"], state.get("current_skills", []))
        return {"skills": result.data}

    def recommendation_node(state: WorkforceState) -> WorkforceState:
        result = recommendation_agent.run(
            state["skills"]["skill_gap"],
            state["attrition"]["risk_level"],
            state["skills"]["readiness"],
        )
        return {"recommendation": result.data}

    def finalize_node(state: WorkforceState) -> WorkforceState:
        return {
            "intelligence": {
                "employee_id": int(state["employee"]["EmployeeNumber"].iloc[0]),
                "attrition": state["attrition"],
                "skills": state["skills"],
                "recommendation": state["recommendation"],
                "agents_executed": [
                    "attrition_agent",
                    "skill_gap_agent",
                    "recommendation_agent",
                ],
            }
        }

    graph = StateGraph(WorkforceState)
    graph.add_node("attrition_agent", attrition_node)
    graph.add_node("skill_gap_agent", skill_node)
    graph.add_node("recommendation_agent", recommendation_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "attrition_agent")
    graph.add_edge("attrition_agent", "skill_gap_agent")
    graph.add_edge("skill_gap_agent", "recommendation_agent")
    graph.add_edge("recommendation_agent", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
