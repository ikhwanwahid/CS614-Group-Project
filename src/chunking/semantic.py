"""Semantic chunking strategy."""

import math
import re

from src.chunking.fixed import chunk_text
from src.shared.chunking_utils import abstract_to_text, build_chunk_record
from src.shared.embeddings import get_embeddings

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Naively split text into sentences."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0

    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    norm = norm_a * norm_b
    if norm == 0.0:
        return 0.0
    return dot / norm


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
    """Chunk abstracts using semantic boundary detection plus size control."""
    chunked: list[dict] = []

    for article in corpus:
        abstract = abstract_to_text(article.get("abstract"))
        sentences = _split_sentences(abstract)

        if not sentences:
            continue

        embeddings = get_embeddings(sentences)
        semantic_chunks = _semantic_split(sentences, embeddings, similarity_threshold)

        chunks: list[str] = []
        for semantic_chunk in semantic_chunks:
            chunks.extend(chunk_text(semantic_chunk, chunk_size=chunk_size, overlap=overlap))

        for i, chunk in enumerate(chunks):
            chunked.append(build_chunk_record(article, i, chunk))

    return chunked
