"""Cross-encoder re-ranking of retrieved passages.

Loads a CrossEncoder once per process (cached by model name) and scores every
(query, passage) pair jointly, which captures interactions that bi-encoder
embeddings miss.
"""

from sentence_transformers import CrossEncoder

# Module-level cache so the model weights are loaded only once per process
_MODEL_CACHE: dict[str, CrossEncoder] = {}


def _get_model(model_name: str) -> CrossEncoder:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _MODEL_CACHE[model_name]


def rerank(
    query: str,
    passages: list[dict],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict]:
    """Re-rank passages using a cross-encoder model.

    Scores every (query, passage) pair jointly with a cross-encoder, which
    captures fine-grained token interactions missed by bi-encoder cosine search.
    Mutates each passage dict in-place to add a ``rerank_score`` field.

    Args:
        query: The original search query / claim text.
        passages: List of passage dicts from initial retrieval, each must have
            a ``'text'`` key.  Other keys (id, metadata, score) are preserved.
        top_k: Number of top passages to return after re-ranking.
        model_name: HuggingFace cross-encoder model identifier.  The default
            ``ms-marco-MiniLM-L-6-v2`` (~80 MB) runs locally, no API key needed.

    Returns:
        Re-ranked list of passage dicts sorted by ``rerank_score`` (descending),
        truncated to ``top_k``.
    """
    if not passages:
        return []

    model = _get_model(model_name)

    pairs = [(query, p["text"]) for p in passages]
    scores = model.predict(pairs)  # returns numpy array of logit scores

    for i, passage in enumerate(passages):
        passage["rerank_score"] = float(scores[i])

    ranked = sorted(passages, key=lambda p: p["rerank_score"], reverse=True)
    return ranked[:top_k]
