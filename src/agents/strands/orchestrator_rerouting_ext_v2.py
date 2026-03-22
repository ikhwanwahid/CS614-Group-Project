"""Revamped Strands orchestrator — original-claim-first retrieval + external fallback.

Key improvements over v1 (orchestrator_rerouting_ext.py):
  1. Always search the ORIGINAL CLAIM first (what RAG does), sub-claims supplement
  2. Smart gating: simple claims skip decomposition entirely
  3. Semantic Scholar fallback triggers on moderate evidence too (not just weak)
  4. Fewer agent calls for simple claims → lower latency and error accumulation

Flow (simple claim, high local evidence):
    claim → [Direct ChromaDB search] → [Verdict LLM call] → done  (2 steps, ~20s)

Flow (simple claim, weak local evidence):
    claim → [Direct ChromaDB search] → [S2 fallback] → [Verdict LLM call] → done  (~30s)

Flow (complex claim):
    claim → [Claim Parser] → [Original + sub-claim retrieval] → [Evidence Reviewer]
        ├─ SUFFICIENT → [Verdict Agent] → done                               (~60s)
        └─ WEAK → [S2 fallback] → [Re-review] → [Verdict Agent] → done      (~80s)
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

from src.agents.strands.claim_parser import parse_claim
from src.agents.strands.retrieval_agent import (
    EvidencePassage,
    RetrievalOutput,
    SubClaimEvidence,
    retrieve_evidence,
)
from src.agents.strands.evidence_reviewer import review_evidence
from src.agents.strands.verdict_agent import generate_verdict
from src.retrieval.semantic_scholar import search as s2_search
from src.shared.vector_store import get_chroma_client, get_or_create_collection, search as chroma_search

load_dotenv()

# Evidence strengths that trigger external search (more aggressive than v1)
_WEAK_STRENGTHS = {"weak", "insufficient", "mixed", "moderate"}

# Heuristics for simple-claim detection
_SIMPLE_CLAIM_MAX_WORDS = 25
_SIMPLE_CLAIM_NO_CONJUNCTIONS = re.compile(
    r"\b(and|while|whereas|furthermore|additionally|moreover|both|not only)\b",
    re.IGNORECASE,
)


def _is_simple_claim(claim: str) -> bool:
    """Determine if a claim is simple enough to skip decomposition.

    Simple claims are short, single-assertion statements that don't need
    to be broken into sub-claims. Decomposing them just dilutes the query.
    """
    words = claim.split()
    if len(words) > _SIMPLE_CLAIM_MAX_WORDS:
        return False
    if _SIMPLE_CLAIM_NO_CONJUNCTIONS.search(claim):
        return False
    # Multiple sentences suggest complexity
    sentences = [s.strip() for s in claim.split(".") if s.strip()]
    if len(sentences) > 1:
        return False
    return True


def _search_local(claim: str, top_k: int = 5) -> list[dict]:
    """Direct ChromaDB search for a single query string."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    return chroma_search(collection, claim, top_k=top_k)


def _hits_to_evidence_passages(hits: list[dict], source_prefix: str = "") -> list[EvidencePassage]:
    """Convert raw ChromaDB hits to EvidencePassage objects."""
    passages = []
    for h in hits:
        src = h["metadata"].get("doc_id", "N/A")
        if source_prefix:
            src = f"{source_prefix}:{src}"
        passages.append(EvidencePassage(
            source=src,
            title=h["metadata"].get("title", ""),
            passage=h["text"][:800],
            relevance_score=round(1.0 - h.get("distance", 0.5), 3),
        ))
    return passages


def _search_external(queries: list[str], limit_per_query: int = 3) -> list[dict]:
    """Search Semantic Scholar for multiple queries, return flat list of hits."""
    all_hits = []
    seen_ids = set()
    for query in queries:
        hits = s2_search(query, limit=limit_per_query)
        for h in hits:
            pid = h.get("id", "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_hits.append(h)
    return all_hits


def _format_evidence(retrieval_output: RetrievalOutput) -> str:
    """Serialize retrieval output to JSON for downstream agents."""
    return json.dumps(
        [
            {
                "sub_claim": se.sub_claim,
                "evidence": [e.model_dump() for e in se.evidence],
            }
            for se in retrieval_output.all_evidence
        ],
        indent=2,
    )


def _format_review(review, retrieval_output: RetrievalOutput) -> str:
    """Combine review + evidence into the JSON blob expected by generate_verdict."""
    return json.dumps(
        {
            "summary": review.summary,
            "flags": [f.model_dump() for f in review.flags],
            "evidence_strength": review.evidence_strength,
            "key_findings": review.key_findings,
            "recommendation": review.recommendation,
            "evidence_by_sub_claim": json.loads(_format_evidence(retrieval_output)),
        },
        indent=2,
    )


def _identify_weak_subclaims(review) -> list[str]:
    """Identify sub-claims flagged as having weak/missing evidence."""
    weak = set()
    for flag in review.flags:
        flag_type = (flag.flag_type or "").upper()
        if flag_type in ("GAP", "WEAK_EVIDENCE", "INSUFFICIENT"):
            for sc in (flag.affected_sub_claims or []):
                weak.add(sc)
    return list(weak)


def _merge_external_into_retrieval(
    retrieval_output: RetrievalOutput,
    external_hits: list[dict],
    target_subclaims: list[str] | None = None,
) -> tuple[RetrievalOutput, int]:
    """Merge S2 hits into retrieval output. Returns (merged_output, papers_added)."""
    papers_added = 0
    ext_passages = _hits_to_evidence_passages(
        [{"text": h["text"], "metadata": h["metadata"], "distance": 0.0} for h in external_hits],
        source_prefix="S2",
    )

    merged_evidence = []
    for se in retrieval_output.all_evidence:
        existing = list(se.evidence)

        # Add external evidence to targeted sub-claims (or all if none specified)
        if target_subclaims is None or se.sub_claim in target_subclaims:
            existing.extend(ext_passages)
            papers_added += len(ext_passages)

        merged_evidence.append(SubClaimEvidence(
            sub_claim=se.sub_claim,
            evidence=existing,
        ))

    return RetrievalOutput(all_evidence=merged_evidence), papers_added


# ── Simple claim path ────────────────────────────────────────────────────────

def _run_simple_path(claim: str) -> dict:
    """Fast path for simple claims: direct search + single verdict call.

    Skips Claim Parser, Retrieval Agent, and Evidence Reviewer entirely.
    Uses direct ChromaDB search (like RAG) + optional S2 fallback.
    """
    # Direct local search with the original claim
    local_hits = _search_local(claim, top_k=5)
    local_passages = _hits_to_evidence_passages(local_hits)

    # Check if local evidence is strong enough
    relevant_hits = [h for h in local_hits if h.get("distance", 1.0) < 0.45]
    needs_external = len(relevant_hits) < 2

    external_search_used = False
    external_papers_added = 0

    if needs_external:
        ext_hits = _search_external([claim], limit_per_query=5)
        if ext_hits:
            external_search_used = True
            external_papers_added = len(ext_hits)
            ext_passages = _hits_to_evidence_passages(
                [{"text": h["text"], "metadata": h["metadata"], "distance": 0.0} for h in ext_hits],
                source_prefix="S2",
            )
            local_passages.extend(ext_passages)

    # Build a minimal retrieval output for the verdict agent
    retrieval_output = RetrievalOutput(
        all_evidence=[SubClaimEvidence(sub_claim=claim, evidence=local_passages)]
    )

    # Skip Evidence Reviewer — go straight to verdict
    evidence_json = _format_evidence(retrieval_output)
    # Build a minimal review for the verdict agent
    review_json = json.dumps({
        "summary": f"Direct evidence retrieval for claim. {len(local_passages)} passages found.",
        "flags": [],
        "evidence_strength": "STRONG" if not needs_external else "MODERATE",
        "key_findings": [f"Retrieved {len(local_passages)} passages for direct claim search."],
        "recommendation": "Evaluate based on retrieved evidence.",
        "evidence_by_sub_claim": json.loads(evidence_json),
    }, indent=2)

    verdict = generate_verdict(claim, review_json)

    return {
        "parsed_claims": {"main_claim": claim, "sub_claims": [{"sub_claim": claim, "query": claim}]},
        "retrieval": retrieval_output.model_dump(),
        "review": {"summary": "Simple claim — direct retrieval path", "flags": [], "evidence_strength": "MODERATE", "key_findings": [], "recommendation": ""},
        "verdict": verdict.model_dump(),
        "path": "simple",
        "rerouting_loops": 1,
        "external_search_used": external_search_used,
        "external_papers_added": external_papers_added,
    }


# ── Complex claim path ───────────────────────────────────────────────────────

def _run_complex_path(claim: str) -> dict:
    """Full path for complex claims: decompose + original-claim-first retrieval + review + verdict.

    Key difference from v1: always includes original claim as a retrieval query
    alongside sub-claim queries, preventing query dilution.
    """
    # ── Agent 1: Decompose claim ─────────────────────────────────────────────
    parsed = parse_claim(claim)
    sub_claims = [
        {"sub_claim": sc.sub_claim, "query": sc.query}
        for sc in parsed.sub_claims
    ]

    # ── Retrieve: original claim FIRST, then sub-claims ──────────────────────
    # This is the key fix: always include the original claim as a retrieval query
    original_hits = _search_local(claim, top_k=5)
    original_evidence = SubClaimEvidence(
        sub_claim=claim,
        evidence=_hits_to_evidence_passages(original_hits),
    )

    # Agent 2: Retrieve evidence for sub-claims (via Strands agent)
    sub_retrieval = retrieve_evidence(sub_claims)

    # Combine: original claim evidence + sub-claim evidence
    all_evidence = [original_evidence] + list(sub_retrieval.all_evidence)
    retrieval_output = RetrievalOutput(all_evidence=all_evidence)

    evidence_json = _format_evidence(retrieval_output)

    # ── Agent 3: Review evidence ─────────────────────────────────────────────
    review = review_evidence(claim, evidence_json)

    strength = (review.evidence_strength or "").lower()
    external_search_used = False
    external_papers_added = 0

    if strength in _WEAK_STRENGTHS:
        # ── Reroute: Semantic Scholar fallback ───────────────────────────────
        weak_subclaims = _identify_weak_subclaims(review)

        # Build search queries: always include original claim + weak sub-claims
        search_queries = [claim]  # original claim always searched
        for sc in sub_claims:
            if not weak_subclaims or sc["sub_claim"] in weak_subclaims:
                search_queries.append(sc.get("query", sc["sub_claim"]))

        # Deduplicate queries
        seen = set()
        unique_queries = []
        for q in search_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        ext_hits = _search_external(unique_queries, limit_per_query=3)

        if ext_hits:
            external_search_used = True
            external_papers_added = len(ext_hits)

            # Merge external evidence into all sub-claims (including original)
            target = weak_subclaims if weak_subclaims else None
            retrieval_output, _ = _merge_external_into_retrieval(
                retrieval_output, ext_hits, target_subclaims=target,
            )
            evidence_json = _format_evidence(retrieval_output)

            # Re-review with enriched evidence
            review = review_evidence(claim, evidence_json)

    # ── Agent 4: Generate verdict ────────────────────────────────────────────
    review_json = _format_review(review, retrieval_output)
    verdict = generate_verdict(claim, review_json)

    return {
        "parsed_claims": parsed.model_dump(),
        "retrieval": retrieval_output.model_dump(),
        "review": review.model_dump(),
        "verdict": verdict.model_dump(),
        "path": "complex",
        "rerouting_loops": 2 if external_search_used else 1,
        "external_search_used": external_search_used,
        "external_papers_added": external_papers_added,
    }


# ── Main entry point ─────────────────────────────────────────────────────────

def run_pipeline_rerouting_ext_v2(claim: str) -> dict:
    """Run the revamped multi-agent pipeline with smart gating and external fallback.

    Args:
        claim: The health claim to fact-check.

    Returns:
        Dict with keys: parsed_claims, retrieval, review, verdict,
        path (simple|complex), rerouting_loops, external_search_used,
        external_papers_added.
    """
    if _is_simple_claim(claim):
        return _run_simple_path(claim)
    else:
        return _run_complex_path(claim)
