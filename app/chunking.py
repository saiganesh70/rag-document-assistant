"""Chunking: fixed-size with overlap, or semantic (split where adjacent sentences diverge)."""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion import PageContent


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _make_chunk(page: PageContent, idx: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{page.doc_id}_p{page.page_number}_c{idx}",
        text=text,
        metadata={
            "doc_id": page.doc_id,
            "filename": page.filename,
            "page_number": page.page_number,
            "total_pages": page.total_pages,
            "chunk_index": idx,
        },
    )


def fixed_chunk(pages: List[PageContent], chunk_size: int = 500, chunk_overlap: int = 100) -> List[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[DocumentChunk] = []
    for page in pages:
        for idx, piece in enumerate(splitter.split_text(page.text)):
            if piece.strip():
                chunks.append(_make_chunk(page, idx, piece.strip()))
    return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def semantic_chunk(
    pages: List[PageContent],
    embed_fn: Callable[[List[str]], List[List[float]]],
    distance_threshold: float = 0.35,
    max_chunk_chars: int = 900,
) -> List[DocumentChunk]:
    """Start a new chunk when cosine distance between neighbouring sentences exceeds the threshold."""
    chunks: List[DocumentChunk] = []
    for page in pages:
        sentences = split_sentences(page.text)
        if not sentences:
            continue
        if len(sentences) == 1:
            chunks.append(_make_chunk(page, 0, sentences[0]))
            continue
        vecs = np.array(embed_fn(sentences), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        groups: List[List[str]] = [[sentences[0]]]
        size = len(sentences[0])
        for i in range(1, len(sentences)):
            distance = 1.0 - float(np.dot(vecs[i - 1], vecs[i]))
            if distance > distance_threshold or size + len(sentences[i]) > max_chunk_chars:
                groups.append([sentences[i]])
                size = len(sentences[i])
            else:
                groups[-1].append(sentences[i])
                size += len(sentences[i])
        for idx, group in enumerate(groups):
            chunks.append(_make_chunk(page, idx, " ".join(group)))
    return chunks


def chunk_documents(
    pages: List[PageContent],
    strategy: str = "fixed",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    semantic_threshold: float = 0.35,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> List[DocumentChunk]:
    strategy = strategy.lower()
    if strategy == "fixed":
        return fixed_chunk(pages, chunk_size, chunk_overlap)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking needs an embed_fn")
        return semantic_chunk(pages, embed_fn, semantic_threshold)
    raise ValueError(f"Unknown chunking strategy '{strategy}' (use 'fixed' or 'semantic')")
