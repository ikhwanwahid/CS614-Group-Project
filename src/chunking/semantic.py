"""Semantic chunking strategy.

Splits text at points where cosine similarity between adjacent sentences
drops below a threshold, respecting topic boundaries rather than fixed sizes.

Uses PubMedBERT sentence embeddings to detect semantic shifts.
"""

import json
import os
import re

import numpy as np

from src.chunking.fixed import chunk_text
from src.shared.chunking_utils import build_chunk_record
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
    """Chunk abstracts using semantic boundary detection plus size control."""
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
            chunked.append(build_chunk_record(article, i, chunk))

    return chunked


if __name__ == "__main__":
    # Load corpus from data/corpus.json
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # For testing, use only first 5 articles to speed up
    corpus = corpus[:5]
    print(f"Loaded corpus with {len(corpus)} articles (limited for testing).")

    # Chunk the corpus using semantic chunking
    chunked_corpus = chunk_corpus_semantic(corpus, similarity_threshold=0.5, chunk_size=200, overlap=50)

    print(f"Total chunks generated: {len(chunked_corpus)}")

    # Print sample output
    print("\nSample chunks:")
    for i, chunk in enumerate(chunked_corpus[:5]):  # First 5 chunks
        print(f"Chunk {i+1}: PMID {chunk['pmid']}, Index {chunk['chunk_index']}")
        print(f"Text: {chunk['text'][:100]}...")  # First 100 chars
        print("-" * 50)
