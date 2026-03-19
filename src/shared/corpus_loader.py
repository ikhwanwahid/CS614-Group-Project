"""Load and chunk the retrieval corpus.

Chunking strategies are implemented in src/chunking/. This module provides
backward-compatible wrappers and the corpus loading/saving utilities.
"""

import json
from pathlib import Path

from src.chunking.fixed import chunk_corpus_fixed


def load_corpus(corpus_path: str = "data/corpus.json") -> list[dict]:
    """Load the local SciFact corpus from JSON."""
    with open(corpus_path) as f:
        return json.load(f)


def chunk_corpus(corpus: list[dict], chunk_size: int = 200, overlap: int = 50) -> list[dict]:
    """Chunk all abstracts in the corpus (backward-compatible, uses fixed strategy).

    For other chunking strategies, use src.chunking.chunk_corpus(corpus, strategy=...).
    """
    return chunk_corpus_fixed(corpus, chunk_size, overlap)


def save_processed_corpus(chunks: list[dict], output_path: str = "data/corpus/processed/chunks.json"):
    """Save chunked corpus to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
