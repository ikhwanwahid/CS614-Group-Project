"""Strands orchestrator with adaptive rerouting + Semantic Scholar fallback.

Flow:
    claim
      → Claim Parser          (once)
      → Retrieval Agent       (local ChromaDB)
      → Evidence Reviewer
          ├─ evidence SUFFICIENT  → Verdict Agent → done
          └─ evidence WEAK/INSUFF → Semantic Scholar API search
                                      → merge external + local evidence
                                      → Evidence Reviewer (round 2)
                                      → Verdict Agent → done

Key difference from orchestrator_rerouting.py:
  * Instead of re-searching the same local corpus (pointless), the reroute
    queries Semantic Scholar for fresh external abstracts.
  * Only 1 reroute loop (external search), not 3 identical local retries.
  * Tracks which evidence came from local vs external sources.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from src.agents.strands.claim_parser import parse_claim
from src.agents.strands.retrieval_agent import retrieve_evidence, RetrievalOutput, SubClaimEvidence, EvidencePassage
from src.agents.strands.evidence_reviewer import review_evidence
from src.agents.strands.verdict_agent import generate_verdict
from src.retrieval.semantic_scholar import search as s2_search

load_dotenv()

# Evidence strengths that trigger external search fallback
_WEAK_STRENGTHS = {"weak", "insufficient", "mixed"}


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


def _search_external_for_subclaims(
    sub_claims: list[dict],
    weak_subclaim_texts: list[str],
) -> dict[str, list[dict]]:
    """Query Semantic Scholar for sub-claims with weak local evidence.

    Args:
        sub_claims: All sub-claims with 'sub_claim' and 'query' keys.
        weak_subclaim_texts: Sub-claim texts flagged as weak by reviewer.

    Returns:
        Dict mapping sub-claim text to list of S2 hits.
    """
    results = {}
    for sc in sub_claims:
        sc_text = sc["sub_claim"]
        # Search for all sub-claims if reviewer flagged specific ones,
        # or all if no specific flags (general weakness)
        if weak_subclaim_texts and sc_text not in weak_subclaim_texts:
            continue
        query = sc.get("query", sc_text)
        hits = s2_search(query, limit=3)
        if hits:
            results[sc_text] = hits
    return results


def _merge_external_evidence(
    retrieval_output: RetrievalOutput,
    external_results: dict[str, list[dict]],
) -> RetrievalOutput:
    """Merge Semantic Scholar results into existing retrieval output.

    External papers are appended to the relevant sub-claim's evidence list,
    tagged with source='semantic_scholar' for traceability.
    """
    merged_evidence = []

    for se in retrieval_output.all_evidence:
        existing = list(se.evidence)

        ext_hits = external_results.get(se.sub_claim, [])
        for hit in ext_hits:
            existing.append(EvidencePassage(
                source=f"S2:{hit['metadata'].get('doc_id', 'unknown')[:12]}",
                title=hit["metadata"].get("title", ""),
                passage=hit["text"][:800],  # cap length
                relevance_score=0.8,  # S2 relevance-ranked, assign reasonable score
            ))

        merged_evidence.append(SubClaimEvidence(
            sub_claim=se.sub_claim,
            evidence=existing,
        ))

    return RetrievalOutput(all_evidence=merged_evidence)


def run_pipeline_rerouting_ext(claim: str) -> dict:
    """Run the multi-agent pipeline with Semantic Scholar external fallback.

    Args:
        claim: The health claim to fact-check.

    Returns:
        Dict with keys: parsed_claims, retrieval, review, verdict,
        rerouting_loops, external_search_used, external_papers_added.
    """
    # ── Agent 1: Decompose claim (done once) ──────────────────────────────────
    parsed = parse_claim(claim)
    sub_claims = [
        {"sub_claim": sc.sub_claim, "query": sc.query}
        for sc in parsed.sub_claims
    ]

    # ── Agent 2: Retrieve local evidence ──────────────────────────────────────
    retrieval_output = retrieve_evidence(sub_claims)
    evidence_json = _format_evidence(retrieval_output)

    # ── Agent 3: Review evidence (round 1) ────────────────────────────────────
    review = review_evidence(claim, evidence_json)

    strength = (review.evidence_strength or "").lower()
    external_search_used = False
    external_papers_added = 0

    if strength in _WEAK_STRENGTHS:
        # ── Reroute: search Semantic Scholar for weak sub-claims ──────────
        weak_subclaims = _identify_weak_subclaims(review)

        # If no specific sub-claims flagged, search all of them
        if not weak_subclaims:
            weak_subclaims = [sc["sub_claim"] for sc in sub_claims]

        external_results = _search_external_for_subclaims(sub_claims, weak_subclaims)
        external_papers_added = sum(len(hits) for hits in external_results.values())

        if external_papers_added > 0:
            external_search_used = True

            # Merge external evidence into retrieval output
            retrieval_output = _merge_external_evidence(retrieval_output, external_results)
            evidence_json = _format_evidence(retrieval_output)

            # Re-review with enriched evidence
            review = review_evidence(claim, evidence_json)

    # ── Agent 4: Generate verdict ─────────────────────────────────────────────
    review_json = _format_review(review, retrieval_output)
    verdict = generate_verdict(claim, review_json)

    return {
        "parsed_claims": parsed.model_dump(),
        "retrieval": retrieval_output.model_dump(),
        "review": review.model_dump(),
        "verdict": verdict.model_dump(),
        "rerouting_loops": 2 if external_search_used else 1,
        "external_search_used": external_search_used,
        "external_papers_added": external_papers_added,
    }
