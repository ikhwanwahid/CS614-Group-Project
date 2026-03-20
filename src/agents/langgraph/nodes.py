"""LangGraph nodes — one function per pipeline stage.

Each node receives the full PipelineState, reads what it needs,
and returns a dict with only the keys it wants to add/update.
LangGraph merges these partial dicts into the shared state automatically.

Design principle: reuse the same system prompts as the Strands agents so
that any quality difference is attributable to the orchestration framework
(LangGraph vs Strands), not to different prompts or models.
"""

from __future__ import annotations

import json

from src.agents.langgraph.state import PipelineState
from src.shared.llm import call_llm

# ── Copied verbatim from the Strands agents so prompts are identical ──────────

_CLAIM_PARSER_SYSTEM = """You are a medical claim decomposition specialist. Your task is to break down
health claims into specific, verifiable sub-claims. For each sub-claim, generate a targeted
PubMed search query that would find relevant evidence.

Guidelines:
- Decompose into 2-4 sub-claims depending on complexity
- Each sub-claim should be independently verifiable
- Search queries should use medical terminology and be specific
- Consider: mechanism claims, population scope, strength of effect, and temporal claims

Respond ONLY with valid JSON:
{
    "main_claim": "<original claim>",
    "sub_claims": [
        {"sub_claim": "<specific verifiable assertion>", "query": "<PubMed search query>"}
    ]
}"""

_EVIDENCE_REVIEWER_SYSTEM = """You are a medical evidence reviewer. Your task is to critically review
retrieved evidence across all sub-claims and provide a structured assessment.

You must:
1. Flag contradictions between evidence passages
2. Identify gaps — sub-claims with weak or missing evidence
3. Note evidence quality (study type: systematic review, RCT, observational, expert opinion)
4. Assess whether the evidence collectively supports, refutes, or partially supports the main claim
5. Highlight if the claim overstates what the evidence actually shows

Respond ONLY with valid JSON:
{
    "summary": "<overall assessment>",
    "flags": [
        {
            "flag_type": "CONTRADICTION | GAP | WEAK_EVIDENCE | QUALITY_NOTE",
            "description": "<description>",
            "affected_sub_claims": ["<sub-claim text>"]
        }
    ],
    "evidence_strength": "STRONG | MODERATE | WEAK | MIXED",
    "key_findings": ["<finding 1>", "<finding 2>"],
    "recommendation": "<preliminary direction for verdict>"
}"""

_VERDICT_SYSTEM = """You are a health claim verdict generator. Based on the evidence review provided,
generate a final verdict with a clear explanation.

Verdict options:
- SUPPORTED: Well-supported by strong, consistent evidence
- UNSUPPORTED: Contradicts available evidence or has no supporting evidence
- OVERSTATED: Contains a kernel of truth but exaggerates the evidence
- INSUFFICIENT_EVIDENCE: Not enough quality evidence to determine

Your explanation must:
1. Address each sub-claim and what the evidence shows
2. Cite specific studies (by PMID) when possible
3. Acknowledge limitations and nuance
4. Be 3-5 sentences long

Respond ONLY with valid JSON:
{
    "verdict": "SUPPORTED | UNSUPPORTED | OVERSTATED | INSUFFICIENT_EVIDENCE",
    "explanation": "<3-5 sentence explanation>",
    "evidence": [
        {"source": "<PMID or author>", "passage": "<key passage>", "relevance_score": 0.0}
    ]
}"""


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_json(content: str) -> dict:
    """Reuse the robust JSON parser already in configurable.py."""
    from src.pipelines.configurable import parse_json_response
    return parse_json_response(content)


def _resolve_model(state: PipelineState) -> str:
    """Pull the model name injected into state by the graph runner (optional)."""
    return state.get("_model", "claude-sonnet-4-20250514")  # type: ignore[arg-type]


# ── Node functions ────────────────────────────────────────────────────────────

def parse_claim_node(state: PipelineState) -> dict:
    """Node 1 — decompose the claim into sub-claims with PubMed queries."""
    claim = state["claim"]
    model = _resolve_model(state)

    prompt = f"Decompose this health claim into verifiable sub-claims: {claim}"
    response = call_llm(prompt, system=_CLAIM_PARSER_SYSTEM, model=model)
    parsed = _parse_json(response["content"])

    sub_claims: list[dict] = parsed.get("sub_claims", [])
    # Normalise — guarantee both keys exist even if model drifts
    sub_claims = [
        {"sub_claim": sc.get("sub_claim", ""), "query": sc.get("query", claim)}
        for sc in sub_claims
    ]
    if not sub_claims:
        # Fallback: treat the whole claim as a single sub-claim
        sub_claims = [{"sub_claim": claim, "query": claim}]

    return {"sub_claims": sub_claims}


def retrieve_evidence_node(state: PipelineState) -> dict:
    """Node 2 — retrieve evidence passages for every sub-claim."""
    sub_claims = state["sub_claims"]

    # Use the shared retrieval agent; it already handles the vector store
    from src.agents.strands.retrieval_agent import retrieve_evidence
    retrieval_output = retrieve_evidence(sub_claims)

    evidence: list[dict] = [
        {
            "sub_claim": se.sub_claim,
            "evidence": [e.model_dump() for e in se.evidence],
        }
        for se in retrieval_output.all_evidence
    ]
    return {"evidence": evidence}


def review_evidence_node(state: PipelineState) -> dict:
    """Node 3 — review evidence quality, flag gaps and contradictions."""
    claim = state["claim"]
    evidence = state["evidence"]
    model = _resolve_model(state)

    evidence_json = json.dumps(evidence, indent=2)
    prompt = (
        f"Review the following evidence for the claim: \"{claim}\"\n\n"
        f"Evidence by sub-claim:\n{evidence_json}\n\n"
        "Provide your structured review."
    )
    response = call_llm(prompt, system=_EVIDENCE_REVIEWER_SYSTEM, model=model)
    review = _parse_json(response["content"])

    # Guarantee expected keys are present
    review.setdefault("summary", "")
    review.setdefault("flags", [])
    review.setdefault("evidence_strength", "WEAK")
    review.setdefault("key_findings", [])
    review.setdefault("recommendation", "")

    return {"review": review}


def generate_verdict_node(state: PipelineState) -> dict:
    """Node 4 — produce the final verdict from the reviewed evidence."""
    claim = state["claim"]
    review = state["review"]
    evidence = state["evidence"]
    model = _resolve_model(state)

    review_with_evidence = {**review, "evidence_by_sub_claim": evidence}
    review_json = json.dumps(review_with_evidence, indent=2)

    prompt = (
        f"Generate a verdict for the claim: \"{claim}\"\n\n"
        f"Evidence review:\n{review_json}\n\n"
        "Provide your verdict, explanation, and cited evidence."
    )
    response = call_llm(prompt, system=_VERDICT_SYSTEM, model=model)
    verdict = _parse_json(response["content"])

    verdict.setdefault("verdict", "INSUFFICIENT_EVIDENCE")
    verdict.setdefault("explanation", "")
    verdict.setdefault("evidence", [])

    return {"verdict": verdict}
