"""Strands orchestrator — runs the 4-agent pipeline sequentially."""

import json

from src.agents.strands.claim_parser import parse_claim
from src.agents.strands.retrieval_agent import retrieve_evidence
from src.agents.strands.evidence_reviewer import review_evidence
from src.agents.strands.verdict_agent import generate_verdict


def run_pipeline(claim: str) -> dict:
    """Run the full P6 multi-agent pipeline.

    Flow: claim → Claim Parser → Retrieval Agent → Evidence Reviewer → Verdict Agent → output
    """
    # Agent 1: Decompose claim into sub-claims
    parsed = parse_claim(claim)
    sub_claims = [{"sub_claim": sc.sub_claim, "query": sc.query} for sc in parsed.sub_claims]

    # Agent 2: Retrieve evidence for each sub-claim
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

    # Agent 3: Review evidence — flag contradictions, gaps, quality
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

    # Agent 4: Generate final verdict
    verdict = generate_verdict(claim, review_json)

    return {
        "parsed_claims": parsed.model_dump(),
        "retrieval": retrieval_output.model_dump(),
        "review": review.model_dump(),
        "verdict": verdict.model_dump(),
    }
