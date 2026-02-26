"""Embedding utilities for vector search using local HuggingFace models."""

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"

_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Get or cache a SentenceTransformer model."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def get_embeddings(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """Get embeddings for a list of texts using a local HuggingFace model."""
    model = get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 10)
    return embeddings.tolist()


def get_embedding(text: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    """Get embedding for a single text."""
    return get_embeddings([text], model_name)[0]
