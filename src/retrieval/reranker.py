"""Cross-encoder re-ranking of retrieved passages."""


def rerank(query: str, passages: list[dict], top_k: int = 5, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[dict]:
    """Re-rank passages using a cross-encoder model.

    Args:
        query: The original search query.
        passages: List of passage dicts from initial retrieval.
        top_k: Number of top passages to return after re-ranking.
        model_name: HuggingFace cross-encoder model name.

    Returns:
        Re-ranked list of passage dicts with updated scores.
    """
    raise NotImplementedError(
        "Cross-encoder reranker not yet implemented — RAG pair (Members 2 & 3).\n"
        "Approach: use sentence-transformers CrossEncoder to score (query, passage) "
        "pairs, sort by score, return top-k."
    )
