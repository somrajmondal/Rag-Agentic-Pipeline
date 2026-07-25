"""
Thin wrapper around sentence-transformers so the rest of the codebase
depends on `Embedder`, not a specific library. Swap the model in .env
without touching any other file.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import settings


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._openai_client = None
        self._model = None

        if settings.embedding_provider.lower() == "openai" and settings.openai_api_key:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            pass
            # from sentence_transformers import SentenceTransformer

            # self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._openai_client is not None:
            response = self._openai_client.embeddings.create(input=texts, model=self.model_name)
            return [item.embedding for item in response.data]

        if self._model is None:
            raise RuntimeError("Embedding model is not initialized")

        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Cached singleton — loading the model is the expensive part."""
    return Embedder()
