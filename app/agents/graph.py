"""
Wires researcher -> writer -> reviewer into a LangGraph StateGraph, with a
conditional edge that loops writer<->reviewer until approved (or the
revision cap in nodes.py kicks in).

    researcher -> writer -> reviewer --(approved)--> END
                     ^                     |
                     +----(revise)---------+
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.nodes import researcher_node, reviewer_node, route_after_review, writer_node
from app.agents.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_review, {"writer": "writer", "end": END})

    return graph.compile()


def run_agentic_query(query: str) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {"query": query, "revision_count": 0, "approved": False}
    return app.invoke(initial_state)
