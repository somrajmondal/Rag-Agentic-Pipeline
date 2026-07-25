from app.ingestion.chunker import chunk_text


def test_short_text_single_chunk():
    chunks = chunk_text("hello world", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_long_text_splits_and_overlaps():
    text = ("paragraph one. " * 20) + "\n\n" + ("paragraph two. " * 20)
    chunks = chunk_text(text, chunk_size=150, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 150 + 20  # allows for stitched overlap
        assert c.metadata["total_chunks"] == len(chunks)


def test_metadata_propagates():
    chunks = chunk_text("a" * 500, metadata={"source": "test.txt"}, chunk_size=100, chunk_overlap=0)
    assert all(c.metadata["source"] == "test.txt" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
