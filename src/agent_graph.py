from __future__ import annotations

from typing import Any, TypedDict

from .agentic_workforce import WorkforceOrchestrator


class WorkforceState(TypedDict, total=False):
    model: Any
    employee: Any
    features: list[str]
    role: str
    current_skills: list[str]
    intelligence: dict[str, Any]


def build_workforce_graph():
    """LangGraph orchestration wrapper around deterministic specialist agents."""
    from langgraph.graph import END, START, StateGraph

    orchestrator = WorkforceOrchestrator()

    def run_agents(state: WorkforceState) -> WorkforceState:
        result = orchestrator.run(
            state["model"],
            state["employee"],
            state["features"],
            state["role"],
            state.get("current_skills", []),
        )
        return {"intelligence": result}

    graph = StateGraph(WorkforceState)
    graph.add_node("workforce_agents", run_agents)
    graph.add_edge(START, "workforce_agents")
    graph.add_edge("workforce_agents", END)
    return graph.compile()
