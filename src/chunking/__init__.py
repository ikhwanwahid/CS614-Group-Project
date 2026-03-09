"""Chunking strategies for corpus preparation.

Each strategy takes a list of corpus articles and returns chunked documents
in a uniform format: list[dict] with keys (pmid, title, chunk_index, text, metadata).
"""

from src.chunking.fixed import chunk_corpus_fixed
from src.chunking.semantic import chunk_corpus_semantic
from src.chunking.section_aware import chunk_corpus_section_aware
from src.chunking.recursive import chunk_corpus_recursive

STRATEGIES = {
    "fixed": chunk_corpus_fixed,
    "semantic": chunk_corpus_semantic,
    "section_aware": chunk_corpus_section_aware,
    "recursive": chunk_corpus_recursive,
}


def chunk_corpus(corpus: list[dict], strategy: str = "fixed", **kwargs) -> list[dict]:
    """Chunk corpus using the specified strategy.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.
        strategy: One of 'fixed', 'semantic', 'section_aware', 'recursive'.
        **kwargs: Strategy-specific parameters.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy](corpus, **kwargs)
