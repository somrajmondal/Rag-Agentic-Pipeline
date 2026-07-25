"""
Orchestrates ingestion (load -> chunk -> embed -> upsert) and retrieval
(embed query -> similarity search -> format context).
Agents call `RAGPipeline.retrieve()` as a tool; the API exposes both halves.
"""
from __future__ import annotations

from app.config import settings
from app.embeddings.embedder import get_embedder
from app.ingestion.chunker import chunk_text
from app.ingestion.loader import load_directory, load_document
from app.vectorstore.faiss_store import FaissVectorStore, RetrievedChunk


class RAGPipeline:
    def __init__(self, store: FaissVectorStore | None = None):
        self.store = store or FaissVectorStore()
        self.embedder = get_embedder()
        self.store.init_schema()

    # ---------- ingestion ----------

    def ingest_file(self, path: str) -> int:
        doc = load_document(path)
        chunks = chunk_text(doc.text, metadata=doc.metadata)
        return self._embed_and_store(chunks)

    def ingest_directory(self, dir_path: str) -> int:
        docs = load_directory(dir_path)
        total = 0
        for doc in docs:
            chunks = chunk_text(doc.text, metadata=doc.metadata)
            total += self._embed_and_store(chunks)
        return total

    def _embed_and_store(self, chunks) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        return self.store.upsert(texts, embeddings, metadatas)

    # ---------- retrieval ----------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)
        return self.store.similarity_search(query_embedding, top_k=top_k or settings.top_k)

    def retrieve_as_context(self, query: str, top_k: int | None = None) -> str:
        """Formats retrieved chunks into a single context block for LLM prompting."""
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return "No relevant context found."
        blocks = []
        for c in chunks:
            source = c.metadata.get("source", "unknown")
            blocks.append(f"[source: {source} | score: {c.score:.3f}]\n{c.text}")
        return "\n\n---\n\n".join(blocks)
