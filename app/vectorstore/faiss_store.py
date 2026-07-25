"""
Local FAISS-backed vector store.

This keeps the ingest/retrieve API identical to the previous Postgres-backed
implementation, but stores the index on disk so the project runs locally
without requiring PostgreSQL or the pgvector extension.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import settings


@dataclass
class RetrievedChunk:
    id: int
    text: str
    metadata: dict[str, Any]
    score: float


class FaissVectorStore:
    def __init__(self, index_path: str | None = None, dim: int | None = None):
        self.index_path = Path(index_path or settings.vector_index_path)
        self.metadata_path = self.index_path.with_suffix(".json")
        self.dim = dim or settings.embedding_dim
        self._index: faiss.Index | None = None
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    def init_schema(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            if self.metadata_path.exists():
                payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                self._texts = payload.get("texts", [])
                self._metadatas = payload.get("metadatas", [])
        else:
            self._index = faiss.IndexFlatIP(self.dim)
            self._texts = []
            self._metadatas = []
            self._write_index()
            self._write_metadata()

        if self._index is None:
            raise RuntimeError("FAISS index failed to initialize")

    def _write_index(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.index_path))

    def _write_metadata(self) -> None:
        payload = {
            "texts": self._texts,
            "metadatas": self._metadatas,
        }
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def upsert(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> int:
        assert len(texts) == len(embeddings) == len(metadatas)
        if not texts:
            return 0

        if self._index is None:
            self.init_schema()

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")

        self._index.add(vectors)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._write_index()
        self._write_metadata()
        return len(texts)

    def similarity_search(self, query_embedding: list[float], top_k: int | None = None) -> list[RetrievedChunk]:
        if self._index is None:
            self.init_schema()

        top_k = top_k or settings.top_k
        if top_k <= 0:
            return []

        query_vector = np.asarray([query_embedding], dtype=np.float32)
        if query_vector.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {query_vector.shape[1]}")

        if self._index.ntotal == 0:
            return []

        distances, indices = self._index.search(query_vector, min(top_k, self._index.ntotal))
        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            text = self._texts[int(idx)]
            metadata = self._metadatas[int(idx)]
            score = float(distances[0][rank])
            results.append(RetrievedChunk(id=int(idx), text=text, metadata=metadata, score=score))
        return results

    def count(self) -> int:
        if self._index is None:
            self.init_schema()
        return self._index.ntotal
