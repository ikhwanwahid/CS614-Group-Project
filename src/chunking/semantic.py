"""Semantic chunking strategy.

Splits text at points where cosine similarity between adjacent sentences
drops below a threshold, respecting topic boundaries rather than fixed sizes.

Uses PubMedBERT sentence embeddings to detect semantic shifts.
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer

from src.shared.embeddings import DEFAULT_MODEL, get_model

# Minimum tokens per chunk — merge tiny chunks with their neighbour
MIN_CHUNK_TOKENS = 30


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _chunk_text_semantic(
    text: str,
    model: SentenceTransformer,
    similarity_threshold: float,
) -> list[str]:
    """Split text into chunks at semantic boundaries."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    # Embed all sentences at once
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Compute cosine similarity between adjacent sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        norm_a = np.linalg.norm(embeddings[i])
        norm_b = np.linalg.norm(embeddings[i + 1])
        if norm_a == 0 or norm_b == 0:
            similarities.append(0.0)
        else:
            sim = float(np.dot(embeddings[i], embeddings[i + 1]) / (norm_a * norm_b))
            similarities.append(sim)

    # Split where similarity drops below threshold
    chunks = []
    current = [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))

    # Merge tiny chunks with their neighbour
    merged = []
    for chunk in chunks:
        token_est = len(chunk.split())
        if merged and token_est < MIN_CHUNK_TOKENS:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged


def chunk_corpus_semantic(
    corpus: list[dict],
    similarity_threshold: float = 0.5,
) -> list[dict]:
    """Chunk all abstracts using semantic boundary detection.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.
        similarity_threshold: Cosine similarity below which a split occurs.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
    """
    model = get_model(DEFAULT_MODEL)
    all_chunks = []

    for article in corpus:
        abstract = article.get("abstract", "")
        if not abstract:
            continue

        texts = _chunk_text_semantic(abstract, model, similarity_threshold)

        for i, text in enumerate(texts):
            all_chunks.append({
                "pmid": article["pmid"],
                "title": article["title"],
                "chunk_index": i,
                "text": text,
            })

    return all_chunks
