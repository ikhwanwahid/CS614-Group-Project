"""Fixed-size chunking strategy (baseline).

Splits text into chunks of approximately chunk_size tokens with overlap.
Uses a simple word-based approximation (1 token ~ 0.75 words).
"""

import json
import os


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
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


def chunk_corpus_fixed(corpus: list[dict], chunk_size: int = 200, overlap: int = 50) -> list[dict]:
    """Chunk all abstracts using fixed-size windowing.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.
        chunk_size: Approximate number of tokens per chunk.
        overlap: Approximate token overlap between consecutive chunks.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
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


if __name__ == "__main__":
    # Load corpus from data/corpus.json
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Loaded corpus with {len(corpus)} articles.")

    # Chunk the corpus
    chunked_corpus = chunk_corpus_fixed(corpus, chunk_size=200, overlap=50)

    print(f"Total chunks generated: {len(chunked_corpus)}")

    # Print sample output
    print("\nSample chunks:")
    for i, chunk in enumerate(chunked_corpus[:5]):  # First 5 chunks
        print(f"Chunk {i+1}: PMID {chunk['pmid']}, Index {chunk['chunk_index']}")
        print(f"Text: {chunk['text'][:100]}...")  # First 100 chars
        print("-" * 50)
