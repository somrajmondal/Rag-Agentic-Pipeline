"""
pgvector-backed store. Uses raw SQL via psycopg for full control over the
IVFFlat index and cosine-distance operator (<=>), same pattern as the
production RAG pipeline this project is based on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


@dataclass
class RetrievedChunk:
    id: int
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity, higher = more similar


class PgVectorStore:
    def __init__(self, dsn: str | None = None, dim: int | None = None):
        # psycopg wants a plain postgresql:// dsn, not the sqlalchemy-style URL
        raw = dsn or settings.database_url
        self.dsn = raw.replace("postgresql+psycopg://", "postgresql://")
        self.dim = dim or settings.embedding_dim

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}',
                    embedding VECTOR({self.dim}) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )
            # IVFFlat index for approximate nearest-neighbour search.
            # Needs data present to train well; safe to (re)build after bulk loads.
            conn.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes WHERE indexname = 'document_chunks_embedding_idx'
                    ) THEN
                        CREATE INDEX document_chunks_embedding_idx
                        ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    END IF;
                END $$;
                """
            )

    def upsert(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> int:
        assert len(texts) == len(embeddings) == len(metadatas)
        if not texts:
            return 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO document_chunks (text, metadata, embedding)
                    VALUES (%s, %s, %s)
                    """,
                    [(t, json.dumps(m), e) for t, e, m in zip(texts, embeddings, metadatas)],
                )
        return len(texts)

    def similarity_search(self, query_embedding: list[float], top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, text, metadata, 1 - (embedding <=> %s) AS score
                FROM document_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            ).fetchall()
        return [RetrievedChunk(id=r[0], text=r[1], metadata=r[2], score=float(r[3])) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
