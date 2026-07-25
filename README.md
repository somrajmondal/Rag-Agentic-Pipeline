# RAG + Agentic Pipeline

A production-shaped RAG pipeline (FAISS + sentence-transformers + FastAPI)
with a LangGraph agentic layer (researcher → writer → reviewer) sitting on
top of it. The agents use RAG retrieval as a tool, so answers are grounded
and self-checked instead of one-shot.

## Architecture

`
                         ┌─────────────────────────────────────────┐
                         │              Ingestion                  │
  documents (.pdf/.md/   │  loader.py → chunker.py → embedder.py    │
  .txt) ────────────────►│  (load)      (recursive     (sentence-   │
                         │               split+overlap) transformers)│
                         └───────────────────┬───────────────────────┘
                                             ▼
                                   ┌───────────────────┐
                                   │   FAISS store       │
                                   │  (cosine similarity, │
                                   │   on-disk index)      │
                                   └─────────┬─────────────┘
                                             ▲
                                    query embedding
                                             │
        ┌────────────────────────────────────┼───────────────────────┐
        │                    Agentic Layer (LangGraph)                │
        │                                                              │
        │   ┌────────────┐     ┌──────────┐     ┌────────────┐        │
        │   │ Researcher │────►│  Writer  │────►│  Reviewer  │        │
        │   │ (RAG tool) │     │ (drafts, │     │ (checks vs │        │
        │   │            │     │  cites)  │     │  context)  │        │
        │   └────────────┘     └────▲─────┘     └─────┬──────┘        │
        │                            └── revise ───────┘              │
        │                                (max 2 loops)  │              │
        │                                                ▼ approved     │
        │                                          final_answer          │
        └──────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                                    FastAPI (/query, /agent/query)
`

**Why this split:** the RAG half (pp/ingestion, pp/embeddings,
pp/vectorstore, pp/rag) has no idea agents exist — it's just
load → chunk → embed → store → retrieve. The agent half (pp/agents)
depends on the RAG half through one function, 
ag_search(). You can use
either layer standalone: hit /query for plain RAG, or /agent/query for
the full self-checking loop.

> This project currently uses OpenAI for embeddings and LLM calls. Anthropic is not used in the current codebase.

## Project layout

`
app/
  config.py              # single source of truth for all settings (.env)
  ingestion/
    loader.py             # .txt/.md/.pdf -> raw text + metadata
    chunker.py             # recursive splitter with overlap
  embeddings/
    embedder.py             # sentence-transformers wrapper, cached singleton
  vectorstore/
    faiss_store.py         # on-disk FAISS index, upsert, similarity search
  rag/
    pipeline.py               # ties ingestion + retrieval together
  agents/
    state.py                  # AgentState TypedDict shared across nodes
    tools.py                    # rag_search() + call_llm() helpers
    nodes.py                     # researcher / writer / reviewer node fns
    graph.py                      # LangGraph StateGraph wiring + revision loop
  api/
    main.py                        # FastAPI: /ingest, /query, /agent/query
cli.py                              # no-server quick usage
tests/test_chunker.py               # chunker unit tests (no external deps)
docker-compose.yml                  # postgres+pgvector, and the api itself
`

## Quickstart

1. **Create your local env file:**
   `ash
   copy .env.example .env
   `
   Fill in OPENAI_API_KEY and API_KEY at minimum.

2. **Install deps locally (for CLI/dev use):**
   `ash
   pip install -r requirements.txt
   `

3. **Ingest documents:**
   `ash
   python cli.py ingest ./data/documents
   `

4. **Ask a question:**
   `ash
   # plain RAG - one retrieval + one LLM call
   python cli.py query "What are the payment terms?"

   # agentic - researcher retrieves, writer drafts, reviewer fact-checks and
   # can send it back for a rewrite (capped at 2 revisions)
   python cli.py agent "What are the payment terms?"
   `

5. **Or run the API locally:**
   `ash
   uvicorn app.api.main:app --reload
   `
   Then call:
   `ash
   curl -X POST http://localhost:8000/ingest/directory -d '{"directory": "./data/documents"}' -H 'Content-Type: application/json' -H 'X-API-Key: your_api_key_here'
   curl -X POST http://localhost:8000/agent/query -d '{"query": "What are the payment terms?"}' -H 'Content-Type: application/json' -H 'X-API-Key: your_api_key_here'
   `

## Notes on the agentic loop

- 
esearcher_node calls 
ag_search() once and stashes context + sources in state.
- writer_node drafts an answer grounded in that context only, citing [source: filename].
- 
eviewer_node checks the draft against the context and either approves or returns
  actionable feedback, which routes back to writer_node via 
oute_after_review.
- MAX_REVISIONS = 2 in 
odes.py prevents infinite loops — after 2 rewrites it
  force-approves so the graph always terminates.

## Swapping pieces

- **Different embedding model:** change EMBEDDING_MODEL / EMBEDDING_DIM in .env
  (dim must match what you set at table-creation time — drop/recreate document_chunks if you change it).
- **Different vector store:** implement the same interface as FaissVectorStore
  (init_schema, upsert, similarity_search, count) and swap it into RAGPipeline.__init__.
- **More agents:** add a node function + wire it into graph.py. State fields
  are additive — add whatever new key a node needs to AgentState.
