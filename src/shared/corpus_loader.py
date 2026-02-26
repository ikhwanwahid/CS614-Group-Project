"""Load and chunk the retrieval corpus."""

import json
from pathlib import Path


def load_corpus(corpus_path: str = "data/corpus.json") -> list[dict]:
    """Load raw corpus from JSON file."""
    with open(corpus_path) as f:
        return json.load(f)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens with overlap.

    Uses a simple word-based approximation (1 token ≈ 0.75 words).
    """
    words = text.split()
    words_per_chunk = int(chunk_size * 0.75)
    words_overlap = int(overlap * 0.75)

    chunks = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += words_per_chunk - words_overlap

    return chunks


def chunk_corpus(corpus: list[dict], chunk_size: int = 200, overlap: int = 50) -> list[dict]:
    """Chunk all abstracts in the corpus.

    Returns list of dicts with fields: pmid, title, chunk_index, text.
    """
    chunked = []
    for article in corpus:
        chunks = chunk_text(article["abstract"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunked.append({
                "pmid": article["pmid"],
                "title": article["title"],
                "chunk_index": i,
                "text": chunk,
            })
    return chunked


def save_processed_corpus(chunks: list[dict], output_path: str = "data/corpus/processed/chunks.json"):
    """Save chunked corpus to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
