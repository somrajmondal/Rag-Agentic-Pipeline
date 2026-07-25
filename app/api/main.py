"""
FastAPI surface:
  POST /ingest/file       - ingest a single file already on disk (path)
  POST /ingest/directory  - ingest every supported file in a directory
  POST /query             - plain RAG: retrieve + one-shot LLM answer
  POST /agent/query       - full agentic loop: researcher -> writer -> reviewer
  GET  /health            - liveness + chunk count
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app.agents.graph import run_agentic_query
from app.agents.tools import call_claude
from app.config import settings
from app.rag.pipeline import RAGPipeline

app = FastAPI(title="RAG + Agentic Pipeline", version="1.0.0")
_pipeline: RAGPipeline | None = None
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


class IngestFileRequest(BaseModel):
    path: str


class IngestDirRequest(BaseModel):
    directory: str


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


@app.get("/health")
def health():
    pipeline = get_pipeline()
    return {"status": "ok", "chunks_indexed": pipeline.store.count()}


@app.post("/ingest/file")
def ingest_file(
    req: IngestFileRequest,
    api_key: str = Depends(verify_api_key),
):
    pipeline = get_pipeline()
    try:
        n = pipeline.ingest_file(req.path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"chunks_added": n}


@app.post("/ingest/directory")
def ingest_directory(
    req: IngestDirRequest,
    api_key: str = Depends(verify_api_key),
):
    pipeline = get_pipeline()
    n = pipeline.ingest_directory(req.directory)
    return {"chunks_added": n}


@app.post("/query")
def query(
    req: QueryRequest,
    api_key: str = Depends(verify_api_key),
):
    """Simple RAG: retrieve context, one LLM call, no review loop."""
    pipeline = get_pipeline()
    context = pipeline.retrieve_as_context(req.query, top_k=req.top_k)
    system = (
        "Answer the user's question using ONLY the provided context. "
        "Cite sources inline like [source: filename]. If unsure, say so."
    )
    answer = call_claude(system, f"Question: {req.query}\n\nContext:\n{context}")
    return {"answer": answer, "context": context}


@app.post("/agent/query")
def agent_query(
    req: QueryRequest,
    api_key: str = Depends(verify_api_key),
):
    """Full agentic loop: researcher retrieves, writer drafts, reviewer checks/loops."""
    result = run_agentic_query(req.query)
    return {
        "answer": result.get("final_answer"),
        "sources": result.get("sources"),
        "revision_count": result.get("revision_count"),
        "review": result.get("review"),
    }
