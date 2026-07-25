"""
Shared state passed between graph nodes. Same researcher/writer/reviewer
shape as the standalone LangGraph project, plus a `context` field populated
by the RAG retrieval tool.
"""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str              # the user's original question / task
    context: str             # retrieved RAG context (filled by researcher)
    sources: list[str]       # source filenames used, for citation
    draft: str                # writer's answer draft
    review: str               # reviewer's feedback
    final_answer: str         # approved final output
    revision_count: int       # loop guard
    approved: bool
