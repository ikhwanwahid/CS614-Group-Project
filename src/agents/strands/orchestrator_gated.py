"""Gated orchestrator — skips Retrieval + Review agents when local evidence is strong."""

from __future__ import annotations

import json
from dataclasses import asdict

from src.agents.strands.claim_parser import parse_claim
from src.agents.strands.retrieval_agent import retrieve_evidence
from src.agents.strands.evidence_reviewer import review_evidence
from src.agents.strands.verdict_agent import generate_verdict
from src.agents.strands.confidence_gate import assess_local_confidence, ConfidenceAssessment


def _format_local_evidence(local_hits: dict[str, list[dict]]) -> str:
    """Format raw ChromaDB hits into the same JSON shape that the Retrieval Agent produces.

    The Verdict Agent expects evidence grouped by sub-claim, each with
    ``source``, ``title``, ``passage``, and ``relevance_score`` fields.
    """
    evidence_by_sub = []
    for sub_claim, hits in local_hits.items():
        passages = [
            {
                "source": h["metadata"].get("doc_id", "N/A"),
                "title": h["metadata"]["title"],
                "passage": h["text"],
                "relevance_score": round(1.0 - h["distance"], 3),
            }
            for h in hits
        ]
        evidence_by_sub.append({"sub_claim": sub_claim, "evidence": passages})
    return json.dumps(evidence_by_sub, indent=2)


def _build_synthetic_review(
    main_claim: str,
    evidence_json: str,
    assessment: ConfidenceAssessment,
) -> str:
    """Build a minimal review JSON for the short-circuit path.

    Instead of calling the Evidence Reviewer agent (~20s), we synthesise a
    lightweight review from the confidence-gate metrics.  The Verdict Agent
    still sees evidence + review context, just without LLM-generated flags.
    """
    review = {
        "summary": (
            f"Local corpus provides strong evidence (gate score {assessment.score:.2f}, "
            f"coverage {assessment.coverage_ratio:.0%}).  "
            "Evidence review was skipped via confidence gate."
        ),
        "flags": [],
        "evidence_strength": "STRONG",
        "key_findings": [
            f"Sub-claim '{sc.sub_claim}': {sc.relevant_hits} relevant local hits "
            f"(avg distance {sc.avg_distance_top_n:.3f})"
            for sc in assessment.sub_claim_scores
        ],
        "recommendation": "Proceed to verdict based on local evidence.",
        "evidence_by_sub_claim": json.loads(evidence_json),
    }
    return json.dumps(review, indent=2)


def run_pipeline_with_gating(claim: str) -> dict:
    """Run the P6 pipeline with a confidence gate after claim parsing.

    Flow:
        claim → Claim Parser → Local ChromaDB search → Confidence Gate
            HIGH → Verdict Agent (with local evidence)     ~50 s, 2 agent calls
            LOW  → Retrieval + Review + Verdict (full)     ~90 s, 4 agent calls

    Returns the same dict shape as ``orchestrator.run_pipeline`` plus a
    ``gating`` key with gate decision metadata.
    """
    # ------------------------------------------------------------------
    # Agent 1: Decompose claim into sub-claims (always runs)
    # ------------------------------------------------------------------
    parsed = parse_claim(claim)
    sub_claims = [
        {"sub_claim": sc.sub_claim, "query": sc.query}
        for sc in parsed.sub_claims
    ]

    # ------------------------------------------------------------------
    # Confidence gate: search ChromaDB directly and score
    # ------------------------------------------------------------------
    assessment, local_hits = assess_local_confidence(sub_claims)

    gating_info = {
        **asdict(assessment),
        "path": "SHORT_CIRCUIT" if assessment.is_high_confidence else "FULL_PIPELINE",
    }

    if assessment.is_high_confidence:
        # ----- SHORT-CIRCUIT path: skip Retrieval Agent + Evidence Reviewer -----
        evidence_json = _format_local_evidence(local_hits)
        review_json = _build_synthetic_review(claim, evidence_json, assessment)

        verdict = generate_verdict(claim, review_json)

        return {
            "parsed_claims": parsed.model_dump(),
            "retrieval": None,       # skipped
            "review": None,          # skipped (synthetic review used internally)
            "verdict": verdict.model_dump(),
            "gating": gating_info,
        }

    # ----- FULL PIPELINE path: standard 4-agent flow -----
    retrieval_output = retrieve_evidence(sub_claims)
    evidence_json = json.dumps(
        [
            {
                "sub_claim": se.sub_claim,
                "evidence": [e.model_dump() for e in se.evidence],
            }
            for se in retrieval_output.all_evidence
        ],
        indent=2,
    )

    review = review_evidence(claim, evidence_json)
    review_json = json.dumps(
        {
            "summary": review.summary,
            "flags": [f.model_dump() for f in review.flags],
            "evidence_strength": review.evidence_strength,
            "key_findings": review.key_findings,
            "recommendation": review.recommendation,
            "evidence_by_sub_claim": json.loads(evidence_json),
        },
        indent=2,
    )

    verdict = generate_verdict(claim, review_json)

    return {
        "parsed_claims": parsed.model_dump(),
        "retrieval": retrieval_output.model_dump(),
        "review": review.model_dump(),
        "verdict": verdict.model_dump(),
        "gating": gating_info,
    }
