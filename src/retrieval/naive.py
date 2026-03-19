"""Naive retrieval — cosine similarity search."""

from src.shared.vector_store import search


def retrieve(
    query: str,
    collection=None,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve top-k passages using cosine similarity.

    Args:
        query: Search query string.
        collection: Prepared Chroma collection to query.
        top_k: Number of hits to return.
    """
    if collection is None:
        raise ValueError("A Chroma collection is required for naive retrieval.")

    return search(collection, query, top_k=top_k)
