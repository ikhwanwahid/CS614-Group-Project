"""Hybrid retrieval — BM25 (lexical) + dense (PubMedBERT) with reciprocal rank fusion.

Combines keyword matching (BM25) with semantic similarity (ChromaDB dense search)
using Reciprocal Rank Fusion (RRF) so neither method dominates.
"""

from rank_bm25 import BM25Okapi

_RRF_K = 60  # Standard RRF constant — penalises very low-ranked docs less aggressively


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """Compute RRF contribution for a document at 0-based `rank`."""
    return 1.0 / (k + rank + 1)


def retrieve_hybrid(
    query: str,
    collection,
    corpus_texts: list[str] | None = None,
    top_k: int = 10,
    bm25_weight: float = 0.5,  # kept for API compatibility; RRF treats both sources equally
) -> list[dict]:
    """Retrieve top-k passages using BM25 + dense retrieval with RRF fusion.

    Pulls all documents from the ChromaDB collection to build a BM25 index, runs
    a dense vector search in parallel, then fuses the two ranked lists via
    Reciprocal Rank Fusion.

    Args:
        query: Search query string.
        collection: ChromaDB collection (used for both dense search and doc fetch).
        corpus_texts: Optional pre-fetched list of texts parallel to
            ``collection.get()["ids"]``. When None (default) texts are fetched
            directly from the collection.
        top_k: Number of results to return after fusion.
        bm25_weight: Unused by RRF; retained for backward-compatible call sites.

    Returns:
        List of hit dicts with 'id', 'text', 'metadata', 'score' (RRF score,
        higher = more relevant), sorted descending.
    """
    # --- Fetch all indexed documents from ChromaDB ---
    all_docs = collection.get(include=["documents", "metadatas"])
    ids: list[str] = all_docs["ids"]
    texts: list[str] = corpus_texts if corpus_texts is not None else all_docs["documents"]
    metadatas: list[dict] = all_docs["metadatas"]

    if not ids:
        return []

    id_to_doc = {
        doc_id: {"text": texts[i], "metadata": metadatas[i]}
        for i, doc_id in enumerate(ids)
    }

    # --- BM25 retrieval (lexical) ---
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_raw_scores = bm25.get_scores(query.lower().split())

    # Build rank lookup: id → 0-based rank (0 = highest BM25 score)
    bm25_order = sorted(range(len(bm25_raw_scores)), key=lambda i: bm25_raw_scores[i], reverse=True)
    bm25_rank_of: dict[str, int] = {ids[i]: rank for rank, i in enumerate(bm25_order)}

    # --- Dense retrieval (ChromaDB cosine similarity) ---
    n_dense = min(top_k * 3, len(ids))
    dense_results = collection.query(query_texts=[query], n_results=n_dense)
    dense_ids: list[str] = dense_results["ids"][0]
    dense_rank_of: dict[str, int] = {doc_id: rank for rank, doc_id in enumerate(dense_ids)}

    # --- RRF fusion over union of top candidates from both lists ---
    bm25_top_ids = {ids[i] for i in bm25_order[: top_k * 3]}
    candidate_ids: set[str] = set(dense_ids) | bm25_top_ids

    scored: list[tuple[str, float]] = []
    for doc_id in candidate_ids:
        rrf = 0.0
        if doc_id in dense_rank_of:
            rrf += _rrf_score(dense_rank_of[doc_id])
        if doc_id in bm25_rank_of:
            rrf += _rrf_score(bm25_rank_of[doc_id])
        scored.append((doc_id, rrf))

    scored.sort(key=lambda x: x[1], reverse=True)

    hits: list[dict] = []
    for doc_id, score in scored[:top_k]:
        doc = id_to_doc[doc_id]
        hits.append({
            "id": doc_id,
            "text": doc["text"],
            "metadata": doc["metadata"],
            "score": score,
        })
    return hits
