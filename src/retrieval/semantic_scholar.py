"""Semantic Scholar API retrieval for external evidence fallback.

Used by the rerouting pipeline when local ChromaDB evidence is weak.
Returns results in the same format as vector_store.search() for easy merging.
"""

from __future__ import annotations

import os
import time
import requests

_API_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,year,citationCount"
_LAST_CALL_TIME: float = 0.0  # enforce 1 req/s rate limit


def _get_api_key() -> str:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if not key:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    return key


def _rate_limit():
    """Enforce 1 request per second."""
    global _LAST_CALL_TIME
    elapsed = time.time() - _LAST_CALL_TIME
    if elapsed < 1.5:
        time.sleep(1.5 - elapsed)
    _LAST_CALL_TIME = time.time()


def search(query: str, limit: int = 5, min_citation_count: int = 0) -> list[dict]:
    """Search Semantic Scholar and return results in ChromaDB-compatible format.

    Returns list of dicts with keys: id, text, metadata, distance
    (distance is always 0.0 since S2 uses relevance ranking, not embedding distance)
    """
    api_key = _get_api_key()
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": limit,
        "fields": _FIELDS,
    }
    if min_citation_count > 0:
        params["minCitationCount"] = min_citation_count

    for attempt in range(3):
        _rate_limit()
        try:
            resp = requests.get(_API_BASE, params=params, headers=headers, timeout=10)
            if resp.status_code == 429:
                wait = 3 * (attempt + 1)
                print(f"[SemanticScholar] Rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            print(f"[SemanticScholar] Search failed for '{query[:50]}': {e}")
            return []
    else:
        print(f"[SemanticScholar] Gave up after 3 rate-limit retries for '{query[:50]}'")
        return []

    hits = []
    for paper in data.get("data", []):
        abstract = paper.get("abstract")
        if not abstract:
            continue  # skip papers without abstracts

        hits.append({
            "id": paper.get("paperId", ""),
            "text": abstract,
            "metadata": {
                "doc_id": paper.get("paperId", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "citation_count": paper.get("citationCount", 0),
                "source": "semantic_scholar",
            },
            "distance": 0.0,  # S2 returns by relevance, not distance
        })

    return hits


def search_multiple(queries: list[str], limit_per_query: int = 3) -> dict[str, list[dict]]:
    """Search multiple queries, respecting rate limits. Returns {query: hits}."""
    results = {}
    for query in queries:
        results[query] = search(query, limit=limit_per_query)
    return results
