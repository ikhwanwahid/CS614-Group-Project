"""Naive retrieval — cosine similarity search."""

from src.shared.vector_store import get_chroma_client, get_or_create_collection, search


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve top-k passages using cosine similarity."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    return search(collection, query, top_k=top_k)
