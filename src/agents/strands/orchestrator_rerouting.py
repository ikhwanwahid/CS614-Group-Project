"""Strands orchestrator with adaptive rerouting.

Flow:
    claim
      → Claim Parser          (once)
      → Retrieval Agent       ←──────────────────────────┐
      → Evidence Reviewer                                 │
          ├─ evidence SUFFICIENT  → Verdict Agent → done  │
          └─ evidence WEAK/INSUFF → refine queries ───────┘
                                    (max MAX_RETRIEVAL_LOOPS times)

The key difference from the plain orchestrator:
  * When the Evidence Reviewer rates strength as "weak" or signals gaps,
    we ask the LLM to rewrite the search queries to be more targeted, then
    loop back to the Retrieval Agent.
  * We cap loops at MAX_RETRIEVAL_LOOPS to prevent runaway calls.
  * The final output includes a "rerouting_loops" count for experiment tracking.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from src.agents.strands.claim_parser import parse_claim
from src.agents.strands.retrieval_agent import retrieve_evidence
from src.agents.strands.evidence_reviewer import review_evidence
from src.agents.strands.verdict_agent import generate_verdict
from src.shared.llm import call_llm

load_dotenv()

MAX_RETRIEVAL_LOOPS = 3

# Evidence strengths that trigger a reroute (more retrieval needed)
_WEAK_STRENGTHS = {"weak", "insufficient"}

_QUERY_REFINER_SYSTEM = """You are a PubMed search query specialist.
Given a set of sub-claims, their original search queries, and a reviewer's
critique, rewrite the queries to be more specific and likely to surface
better evidence.

Rules:
- Keep sub-claim text unchanged; only rewrite the "query" field
- Use MeSH terms and Boolean operators where helpful (AND, OR, NOT)
- Make queries more specific — avoid overly broad terms
- If the reviewer flagged a GAP for a sub-claim, try a completely different angle

Respond ONLY with valid JSON — an array matching this exact shape:
[
  {"sub_claim": "<unchanged sub-claim text>", "query": "<new PubMed query>"}
]"""


def _format_evidence(retrieval_output) -> str:
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


def _format_review(review, retrieval_output) -> str:
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


def _refine_queries(sub_claims: list[dict], review) -> list[dict]:
    """Ask the LLM to rewrite search queries based on the reviewer's flags.

    Args:
        sub_claims: Current list of {"sub_claim": str, "query": str} dicts.
        review:     ReviewedEvidence object from evidence_reviewer.

    Returns:
        Refined list of {"sub_claim": str, "query": str} dicts.
        Falls back to original sub_claims if parsing fails.
    """
    flags_json = json.dumps([f.model_dump() for f in review.flags], indent=2)
    sub_claims_json = json.dumps(sub_claims, indent=2)

    prompt = (
        f"Original sub-claims and queries:\n{sub_claims_json}\n\n"
        f"Reviewer critique (flags):\n{flags_json}\n\n"
        f"Overall evidence strength: {review.evidence_strength}\n\n"
        "Rewrite the search queries to surface better evidence for the weak areas."
    )

    try:
        response = call_llm(prompt, system=_QUERY_REFINER_SYSTEM)
        from src.pipelines.configurable import parse_json_response
        refined = parse_json_response(response["content"])

        # parse_json_response always returns a dict; handle list wrapped in dict
        if isinstance(refined, list):
            candidates = refined
        elif isinstance(refined, dict):
            # Model may have returned {"sub_claims": [...]}
            candidates = refined.get("sub_claims", refined.get("queries", []))
        else:
            candidates = []

        if candidates and len(candidates) == len(sub_claims):
            # Guarantee original sub_claim text is preserved
            return [
                {
                    "sub_claim": orig["sub_claim"],
                    "query": cand.get("query", orig["query"]),
                }
                for orig, cand in zip(sub_claims, candidates)
            ]
    except Exception:
        pass  # Fallback: keep original queries

    return sub_claims


def run_pipeline_rerouting(claim: str) -> dict:
    """Run the multi-agent pipeline with adaptive rerouting.

    Args:
        claim: The health claim to fact-check.

    Returns:
        Dict with keys: parsed_claims, retrieval, review, verdict,
        rerouting_loops (int — how many retrieval rounds were needed).
    """
    # ── Agent 1: Decompose claim (done once) ──────────────────────────────────
    parsed = parse_claim(claim)
    sub_claims = [
        {"sub_claim": sc.sub_claim, "query": sc.query}
        for sc in parsed.sub_claims
    ]

    retrieval_output = None
    review = None

    for loop in range(MAX_RETRIEVAL_LOOPS):
        # ── Agent 2: Retrieve evidence ────────────────────────────────────────
        retrieval_output = retrieve_evidence(sub_claims)
        evidence_json = _format_evidence(retrieval_output)

        # ── Agent 3: Review evidence ──────────────────────────────────────────
        review = review_evidence(claim, evidence_json)

        strength = (review.evidence_strength or "").lower()
        is_last_loop = loop == MAX_RETRIEVAL_LOOPS - 1
        evidence_sufficient = strength not in _WEAK_STRENGTHS

        if evidence_sufficient or is_last_loop:
            # Proceed to verdict
            break

        # ── Reroute: refine queries and loop back ─────────────────────────────
        sub_claims = _refine_queries(sub_claims, review)

    # ── Agent 4: Generate verdict ─────────────────────────────────────────────
    review_json = _format_review(review, retrieval_output)
    verdict = generate_verdict(claim, review_json)

    return {
        "parsed_claims": parsed.model_dump(),
        "retrieval": retrieval_output.model_dump(),
        "review": review.model_dump(),
        "verdict": verdict.model_dump(),
        "rerouting_loops": loop + 1,
    }
