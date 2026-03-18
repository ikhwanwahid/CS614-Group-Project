"""Chunking strategy dispatcher.

Strategies are imported lazily so fixed/section-aware runs do not need to load
semantic or recursive chunking dependencies unless selected.
"""

from __future__ import annotations

from collections.abc import Callable

StrategyFn = Callable[..., list[dict]]


def _load_strategy(strategy: str) -> StrategyFn:
    if strategy == "fixed":
        from src.chunking.fixed import chunk_corpus_fixed

        return chunk_corpus_fixed
    if strategy == "semantic":
        from src.chunking.semantic import chunk_corpus_semantic

        return chunk_corpus_semantic
    if strategy == "section_aware":
        from src.chunking.section_aware import chunk_corpus_section_aware

        return chunk_corpus_section_aware
    if strategy == "recursive":
        from src.chunking.recursive import chunk_corpus_recursive

        return chunk_corpus_recursive
    raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from: {list_strategies()}")


def list_strategies() -> tuple[str, ...]:
    """Return supported chunking strategy names."""
    return ("fixed", "semantic", "section_aware", "recursive")


def chunk_corpus(corpus: list[dict], strategy: str = "fixed", **kwargs) -> list[dict]:
    """Chunk corpus using the specified strategy.

    Args:
        corpus: List of article dicts with 'doc_id', 'title', 'abstract'.
        strategy: One of 'fixed', 'semantic', 'section_aware', 'recursive'.
        **kwargs: Strategy-specific parameters.

    Returns:
        List of chunk dicts with 'doc_id', 'title', 'chunk_index', 'text'.
    """
    return _load_strategy(strategy)(corpus, **kwargs)
