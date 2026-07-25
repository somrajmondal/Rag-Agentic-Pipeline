"""
The three nodes of the graph. Each takes AgentState in, returns a partial
state update (langgraph merges it). Researcher does retrieval, Writer drafts
an answer grounded in that context, Reviewer checks it and either approves
or sends it back for revision (max 2 loops, then force-approve so it can't
hang forever).
"""
from __future__ import annotations

from app.agents.state import AgentState
from app.agents.tools import call_ai, rag_search

MAX_REVISIONS = 2


def researcher_node(state: AgentState) -> dict:
    context, sources = rag_search(state["query"])
    return {"context": context, "sources": sources}


def writer_node(state: AgentState) -> dict:
    revision_note = ""
    if state.get("review"):
        revision_note = f"\n\nPrevious reviewer feedback to address:\n{state['review']}"

    system = (
        "You are a precise technical writer. Answer the user's question using ONLY the "
        "provided context. If the context doesn't contain the answer, say so explicitly "
        "instead of guessing. Cite sources inline like [source: filename]."
    )
    user = (
        f"Question: {state['query']}\n\n"
        f"Context:\n{state['context']}"
        f"{revision_note}"
    )
    draft = call_ai(system, user)
    return {"draft": draft, "revision_count": state.get("revision_count", 0) + 1}


def reviewer_node(state: AgentState) -> dict:
    system = (
        "You are a strict fact-checking reviewer. Given a question, the source context, "
        "and a draft answer, check whether the draft is fully supported by the context "
        "and actually answers the question. Reply with 'APPROVED' on the first line if "
        "it's good. Otherwise reply with 'REVISE' on the first line followed by specific, "
        "actionable feedback."
    )
    user = (
        f"Question: {state['query']}\n\n"
        f"Context:\n{state['context']}\n\n"
        f"Draft answer:\n{state['draft']}"
    )
    verdict = call_ai(system, user, max_tokens=512)
    approved = verdict.strip().upper().startswith("APPROVED")

    if approved or state.get("revision_count", 0) >= MAX_REVISIONS:
        return {"approved": True, "final_answer": state["draft"], "review": verdict}
    return {"approved": False, "review": verdict}


def route_after_review(state: AgentState) -> str:
    """Conditional edge: loop back to writer, or finish."""
    return "end" if state.get("approved") else "writer"
