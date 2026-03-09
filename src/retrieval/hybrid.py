"""Hybrid retrieval — BM25 (lexical) + dense (PubMedBERT) with reciprocal rank fusion."""


def retrieve_hybrid(query: str, collection, corpus_texts: list[str] | None = None, top_k: int = 10, bm25_weight: float = 0.5) -> list[dict]:
    """Retrieve top-k passages using BM25 + dense retrieval with rank fusion.

    Args:
        query: Search query string.
        collection: ChromaDB collection (for dense search).
        corpus_texts: List of all corpus texts for BM25 indexing.
        top_k: Number of results to return after fusion.
        bm25_weight: Weight for BM25 scores in fusion (0-1).

    Returns:
        List of hit dicts with 'id', 'text', 'metadata', 'score'.
    """
    raise NotImplementedError(
        "Hybrid retrieval not yet implemented — RAG pair (Members 2 & 3).\n"
        "Approach: run BM25 (via rank-bm25) and ChromaDB dense search in parallel, "
        "combine using reciprocal rank fusion (RRF)."
    )
