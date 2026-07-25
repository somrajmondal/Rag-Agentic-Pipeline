"""
Loads raw text out of files. Supports .txt, .md, .pdf.
Returns a list of (text, metadata) tuples — one per file, page-joined for PDFs
so the chunker can decide how to split rather than the loader.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_txt(path: str) -> RawDocument:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return RawDocument(text=text, metadata={"source": os.path.basename(path), "path": path})


def load_pdf(path: str) -> RawDocument:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        content = page.extract_text() or ""
        pages.append(f"[page {i + 1}]\n{content}")
    text = "\n\n".join(pages)
    return RawDocument(
        text=text,
        metadata={"source": os.path.basename(path), "path": path, "num_pages": len(reader.pages)},
    )


LOADERS = {
    ".txt": load_txt,
    ".md": load_txt,
    ".pdf": load_pdf,
}


def load_document(path: str) -> RawDocument:
    ext = os.path.splitext(path)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS)}")
    return loader(path)


def load_directory(dir_path: str) -> list[RawDocument]:
    docs = []
    for name in sorted(os.listdir(dir_path)):
        full_path = os.path.join(dir_path, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(full_path) and ext in LOADERS:
            docs.append(load_document(full_path))
    return docs
