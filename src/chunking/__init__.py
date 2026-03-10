"""Chunking strategies for corpus preparation.

Each strategy takes a list of corpus articles and returns chunked documents
in a uniform format: list[dict] with keys (pmid, title, chunk_index, text, metadata).

Imports are lazy to avoid pulling in heavy dependencies (sentence-transformers,
langchain) when only one strategy is needed.
"""

STRATEGY_NAMES = ("fixed", "semantic", "section_aware", "recursive")


def _get_strategy(name: str):
    """Lazy-import and return the chunking function for a strategy."""
    if name == "fixed":
        from src.chunking.fixed import chunk_corpus_fixed
        return chunk_corpus_fixed
    elif name == "semantic":
        from src.chunking.semantic import chunk_corpus_semantic
        return chunk_corpus_semantic
    elif name == "section_aware":
        from src.chunking.section_aware import chunk_corpus_section_aware
        return chunk_corpus_section_aware
    elif name == "recursive":
        from src.chunking.recursive import chunk_corpus_recursive
        return chunk_corpus_recursive
    else:
        raise ValueError(f"Unknown chunking strategy: {name}. Choose from: {list(STRATEGY_NAMES)}")


def chunk_corpus(corpus: list[dict], strategy: str = "fixed", **kwargs) -> list[dict]:
    """Chunk corpus using the specified strategy.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.
        strategy: One of 'fixed', 'semantic', 'section_aware', 'recursive'.
        **kwargs: Strategy-specific parameters.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
    """
    fn = _get_strategy(strategy)
    return fn(corpus, **kwargs)
