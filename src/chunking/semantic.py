"""Semantic chunking strategy.

Splits text at points where cosine similarity between adjacent sentences
drops below a threshold, respecting topic boundaries rather than fixed sizes.

Uses PubMedBERT sentence embeddings to detect semantic shifts.
"""

import re
import numpy as np
from src.chunking.fixed import chunk_text
from src.shared.embeddings import get_embeddings

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Naively split text into sentences."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0.0:
        return 0.0
    return float(np.dot(va, vb) / norm)


def _semantic_split(sentences: list[str], embeddings: list[list[float]], threshold: float) -> list[str]:
    """Group sentences into chunks by detecting similarity drops below threshold."""
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    chunks: list[str] = []
    current: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_corpus_semantic(
    corpus: list[dict],
    similarity_threshold: float = 0.5,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[dict]:
    """Chunk all abstracts using semantic boundary detection.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.
        similarity_threshold: Cosine similarity below which a split occurs.
        chunk_size: Approximate max tokens per chunk.
        overlap: Approximate token overlap between consecutive chunks.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
    """
    chunked: list[dict] = []
    for article in corpus:
        abstract = article.get("abstract", "") or ""
        sentences = _split_sentences(abstract)

        if not sentences:
            continue

        embeddings = get_embeddings(sentences)
        semantic_chunks = _semantic_split(sentences, embeddings, similarity_threshold)

        chunks: list[str] = []
        for semantic_chunk in semantic_chunks:
            chunks.extend(chunk_text(semantic_chunk, chunk_size=chunk_size, overlap=overlap))

        for i, chunk in enumerate(chunks):
            chunked.append({
                "pmid": article["pmid"],
                "title": article["title"],
                "chunk_index": i,
                "text": chunk,
            })
    return chunked
