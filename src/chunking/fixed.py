"""Fixed-size chunking strategy."""

from src.shared.chunking_utils import abstract_to_text, build_chunk_record


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
        corpus: List of article dicts with 'doc_id', 'title', 'abstract'.
        chunk_size: Approximate number of tokens per chunk.
        overlap: Approximate token overlap between consecutive chunks.

    Returns:
        List of chunk dicts with 'doc_id', 'title', 'chunk_index', 'text'.
    """
    chunked = []
    for article in corpus:
        chunks = chunk_text(abstract_to_text(article.get("abstract")), chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunked.append(build_chunk_record(article, i, chunk))
    return chunked
