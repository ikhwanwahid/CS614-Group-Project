"""ChromaDB vector store setup and search."""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from src.shared.embeddings import DEFAULT_MODEL

# ChromaDB has built-in support for sentence-transformers
_CHROMA_EF_CACHE: dict[str, embedding_functions.SentenceTransformerEmbeddingFunction] = {}


def _get_embedding_function(
    model_name: str = DEFAULT_MODEL,
) -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Get or cache a ChromaDB-compatible sentence-transformer embedding function."""
    if model_name not in _CHROMA_EF_CACHE:
        _CHROMA_EF_CACHE[model_name] = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
        )
    return _CHROMA_EF_CACHE[model_name]


def get_chroma_client(persist_dir: str = "data/corpus/embeddings/chroma_db") -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(
    client: chromadb.ClientAPI,
    collection_name: str = "health_corpus",
    embedding_model: str = DEFAULT_MODEL,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection with local HuggingFace embeddings."""
    ef = _get_embedding_function(embedding_model)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
    )


def reset_collection(client: chromadb.ClientAPI, collection_name: str) -> None:
    """Delete a collection if it exists so it can be rebuilt cleanly."""
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass


def add_documents(collection: chromadb.Collection, chunks: list[dict]):
    """Add chunked documents to the collection in batches."""
    BATCH_SIZE = 500
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[f"{c['doc_id']}_{c['chunk_index']}" for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "chunk_index": c["chunk_index"],
                    **c.get("metadata", {}),
                }
                for c in batch
            ],
        )


def search(collection: chromadb.Collection, query: str, top_k: int = 5) -> list[dict]:
    """Search the collection and return top-k results."""
    results = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits
