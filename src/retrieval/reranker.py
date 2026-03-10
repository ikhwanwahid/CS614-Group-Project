"""Cross-encoder re-ranking of retrieved passages."""

from sentence_transformers import CrossEncoder

# Module-level cache for the cross-encoder model
_model_cache: dict[str, CrossEncoder] = {}

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model(model_name: str = DEFAULT_RERANKER) -> CrossEncoder:
    """Get or cache a CrossEncoder model."""
    if model_name not in _model_cache:
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def rerank(
    query: str,
    passages: list[dict],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER,
) -> list[dict]:
    """Re-rank passages using a cross-encoder model.

    Args:
        query: The original search query.
        passages: List of passage dicts with at least 'text' key.
        top_k: Number of top passages to return after re-ranking.
        model_name: HuggingFace cross-encoder model name.

    Returns:
        Re-ranked list of passage dicts with 'rerank_score' added.
    """
    if not passages:
        return []

    model = _get_model(model_name)

    # Score each (query, passage) pair
    pairs = [(query, p.get("text", "")) for p in passages]
    scores = model.predict(pairs)

    # Add scores to passages
    for i, p in enumerate(passages):
        p["rerank_score"] = float(scores[i])

    # Sort by rerank score descending, return top-k
    ranked = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]
