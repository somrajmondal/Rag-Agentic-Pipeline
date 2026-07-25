"""
Recursive character-based chunker (same idea as LangChain's RecursiveCharacterTextSplitter,
but dependency-free). Splits on paragraph -> line -> sentence -> char boundaries,
keeping chunk_overlap characters of context between chunks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import settings

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_text(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size or not separators:
        return [text]

    sep, *rest_separators = separators
    parts = text.split(sep) if sep else list(text)

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, rest_separators, chunk_size))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    metadata = metadata or {}

    raw_chunks = _split_text(text.strip(), SEPARATORS, chunk_size)

    # stitch overlap back in so each chunk keeps a tail of the previous one
    overlapped: list[str] = []
    for i, c in enumerate(raw_chunks):
        if i == 0 or chunk_overlap == 0:
            overlapped.append(c)
        else:
            tail = raw_chunks[i - 1][-chunk_overlap:]
            overlapped.append(tail + c)

    return [
        Chunk(text=c, metadata={**metadata, "chunk_index": i, "total_chunks": len(overlapped)})
        for i, c in enumerate(overlapped)
        if c.strip()
    ]
