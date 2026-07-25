from pathlib import Path

from app.vectorstore.faiss_store import FaissVectorStore, RetrievedChunk


def test_faiss_store_upsert_and_similarity_search(tmp_path: Path):
    store = FaissVectorStore(index_path=str(tmp_path / "faiss.index"), dim=3)
    store.init_schema()

    texts = ["alpha", "beta", "gamma"]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    metadata = [
        {"source": "a.txt"},
        {"source": "b.txt"},
        {"source": "c.txt"},
    ]

    added = store.upsert(texts, embeddings, metadata)
    assert added == 3

    results = store.similarity_search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].text == "alpha"
    assert results[0].metadata["source"] == "a.txt"
    assert store.count() == 3
