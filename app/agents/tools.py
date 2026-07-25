"""
Tool-ish helpers the agent nodes call. Kept as plain functions rather than
LangChain @tool decorators — the graph nodes call these directly, and it's
one less abstraction layer to debug.
"""
from __future__ import annotations

from anthropic import Anthropic
from openai import OpenAI

from app.config import settings
from app.rag.pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None
_openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def rag_search(query: str, top_k: int | None = None) -> tuple[str, list[str]]:
    """Runs retrieval and returns (context_block, source_names)."""
    pipeline = get_pipeline()
    chunks = pipeline.retrieve(query, top_k=top_k)
    if not chunks:
        return "No relevant context found in the knowledge base.", []
    context = "\n\n---\n\n".join(
        f"[source: {c.metadata.get('source', 'unknown')}]\n{c.text}" for c in chunks
    )
    sources = sorted({c.metadata.get("source", "unknown") for c in chunks})
    return context, sources


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    provider = settings.llm_provider.lower()

    if provider == "openai":
        if _openai_client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        response = _openai_client.chat.completions.create(
            model=settings.openai_model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    raise ValueError(f"Unsupported LLM provider: {provider}")


def call_ai(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    return call_llm(system_prompt, user_prompt, max_tokens=max_tokens)
