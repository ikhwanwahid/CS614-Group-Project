"""Hybrid retrieval — BM25 (lexical) + dense (PubMedBERT) with reciprocal rank fusion."""

from rank_bm25 import BM25Okapi

from src.shared.vector_store import search as dense_search

# RRF constant (standard value from the literature)
RRF_K = 60

# Module-level BM25 cache to avoid rebuilding on every call
_bm25_cache: dict[str, tuple[BM25Okapi, list[str]]] = {}


def _build_bm25_index(collection) -> tuple[BM25Okapi, list[str], list[dict]]:
    """Build a BM25 index from all documents in a ChromaDB collection."""
    cache_key = collection.name

    # Get all documents from the collection
    all_docs = collection.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    # Tokenize for BM25
    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)

    # Build doc list for lookup
    doc_list = []
    for i in range(len(ids)):
        doc_list.append({
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
        })

    return bm25, documents, doc_list


def retrieve_hybrid(
    query: str,
    collection,
    top_k: int = 10,
    bm25_weight: float = 0.5,
) -> list[dict]:
    """Retrieve top-k passages using BM25 + dense retrieval with rank fusion.

    Args:
        query: Search query string.
        collection: ChromaDB collection (for dense search).
        top_k: Number of results to return after fusion.
        bm25_weight: Weight for BM25 in fusion (0-1). Dense weight = 1 - bm25_weight.

    Returns:
        List of hit dicts with 'id', 'text', 'metadata', 'score'.
    """
    n_candidates = top_k * 3  # fetch more candidates for fusion

    # 1. Dense retrieval from ChromaDB
    dense_hits = dense_search(collection, query, top_k=n_candidates)
    dense_ranking = {h["id"]: rank for rank, h in enumerate(dense_hits)}

    # 2. BM25 retrieval
    bm25, documents, doc_list = _build_bm25_index(collection)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Sort by BM25 score, take top candidates
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_candidates]
    bm25_ranking = {}
    for rank, idx in enumerate(bm25_ranked_indices):
        bm25_ranking[doc_list[idx]["id"]] = rank

    # 3. Reciprocal rank fusion
    all_doc_ids = set(dense_ranking.keys()) | set(bm25_ranking.keys())
    fused_scores = {}
    doc_lookup = {h["id"]: h for h in dense_hits}
    for doc in doc_list:
        doc_lookup.setdefault(doc["id"], doc)

    dense_weight = 1.0 - bm25_weight

    for doc_id in all_doc_ids:
        score = 0.0
        if doc_id in dense_ranking:
            score += dense_weight * (1.0 / (RRF_K + dense_ranking[doc_id]))
        if doc_id in bm25_ranking:
            score += bm25_weight * (1.0 / (RRF_K + bm25_ranking[doc_id]))
        fused_scores[doc_id] = score

    # 4. Sort by fused score, return top-k
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    results = []
    for doc_id in sorted_ids:
        doc = doc_lookup.get(doc_id, {})
        results.append({
            "id": doc_id,
            "text": doc.get("text", ""),
            "metadata": doc.get("metadata", {}),
            "score": fused_scores[doc_id],
        })

    return results
