"""
Central settings, loaded from env / .env.
Everything downstream (embedder, vectorstore, agents) reads from here
so there's exactly one place to change models / dims / chunk sizes.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    agent_model: str = "gpt-4o-mini"

    # Server auth
    api_key: str = ""

    # Vector store
    vector_index_path: str = "./data/faiss.index"

    # Embeddings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 5


settings = Settings()
