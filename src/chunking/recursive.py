"""Recursive chunking strategy.

Uses LangChain's RecursiveCharacterTextSplitter with paragraph -> sentence ->
character fallback and keeps the same corpus-backed metadata as the other
chunking strategies for fair comparison.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.shared.chunking_utils import build_chunk_record

# Separators: paragraph break, sentence end, space, then individual characters
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def chunk_corpus_recursive(
    corpus: list[dict], chunk_size: int = 200, overlap: int = 50
) -> list[dict]:
    """Chunk all abstracts using recursive splitting.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract',
                and optionally 'year'.
        chunk_size: Approximate token budget per chunk.
        overlap: Approximate token overlap between consecutive chunks.

    Returns:
        List of dicts with 'pmid', 'title', 'chunk_index', 'text', 'metadata'.
    """

    # Keep API in token-like units for consistency with other chunkers.
    char_chunk_size = int(chunk_size * 4)
    char_overlap = int(overlap * 4)

    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=char_chunk_size,
        chunk_overlap=char_overlap,
        length_function=len,
    )

    chunks: list[dict] = []
    for article in corpus:
        abstract = article.get("abstract", "") or ""
        raw_chunks = splitter.split_text(abstract)
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(build_chunk_record(article, i, chunk_text))
    return chunks
