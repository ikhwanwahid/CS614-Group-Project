"""LangGraph node functions for the 4-agent fact-checking pipeline.

Each node mirrors the Strands agent's prompt and logic but uses the shared
call_llm() client so it can run with any model/provider.
"""

import json
import re

from src.shared.llm import call_llm
from src.shared.vector_store import get_chroma_client, get_or_create_collection, search
from src.agents.langgraph.state import PipelineState


# ── Prompts (mirrored from Strands agents) ────────────────────────────────

CLAIM_PARSER_SYSTEM = """You are a medical claim decomposition specialist. Break down
health claims into specific, verifiable sub-claims. For each sub-claim, generate a targeted
PubMed search query that would find relevant evidence.

Guidelines:
- Decompose into 2-4 sub-claims depending on complexity
- Each sub-claim should be independently verifiable
- Search queries should use medical terminology and be specific
- Consider: mechanism claims, population scope, strength of effect, and temporal claims

Respond ONLY with valid JSON:
[
    {"sub_claim": "specific verifiable statement", "query": "PubMed search query"}
]"""

EVIDENCE_REVIEWER_SYSTEM = """You are a medical evidence reviewer. Critically review
retrieved evidence across all sub-claims and provide a structured assessment.

You must:
1. Flag contradictions between evidence passages
2. Identify gaps — sub-claims with weak or missing evidence
3. Note evidence quality (study type: systematic review, RCT, observational, expert opinion)
4. Assess whether the evidence collectively supports, refutes, or partially supports the main claim
5. Highlight if the claim overstates what the evidence actually shows

Respond ONLY with valid JSON:
{
    "summary": "Overall assessment",
    "flags": [{"flag_type": "CONTRADICTION|GAP|WEAK_EVIDENCE|QUALITY_NOTE", "description": "...", "affected_sub_claims": ["..."]}],
    "evidence_strength": "STRONG|MODERATE|WEAK|MIXED",
    "key_findings": ["finding 1", "finding 2"],
    "recommendation": "preliminary verdict direction"
}"""

VERDICT_SYSTEM = """You are a health claim verdict generator. Based on the evidence review,
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
    "verdict": "SUPPORTED|UNSUPPORTED|OVERSTATED|INSUFFICIENT_EVIDENCE",
    "explanation": "3-5 sentence explanation",
    "evidence": [{"source": "PMID:...", "passage": "key passage", "relevance_score": 0.0-1.0}]
}"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_json(content: str, fallback=None):
    """Parse JSON from LLM response with markdown fallback."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return fallback


def _call(prompt: str, system: str, state: PipelineState, max_tokens: int = 2048) -> dict:
    """Convenience wrapper around call_llm using state's model/provider."""
    model = state.get("model")
    provider = state.get("provider", "anthropic")
    return call_llm(prompt, system=system, model=model, provider=provider, max_tokens=max_tokens)


# ── Node 1: Claim Parser ──────────────────────────────────────────────────

def parse_claim_node(state: PipelineState) -> dict:
    """Decompose the claim into sub-claims with search queries."""
    claim = state["claim"]
    response = _call(
        f"Decompose this health claim into verifiable sub-claims:\n\n{claim}",
        CLAIM_PARSER_SYSTEM,
        state,
        max_tokens=1024,
    )
    sub_claims = _parse_json(response["content"], fallback=[{"sub_claim": claim, "query": claim}])

    # Normalise: ensure list of dicts with sub_claim + query
    if isinstance(sub_claims, dict):
        sub_claims = sub_claims.get("sub_claims", [{"sub_claim": claim, "query": claim}])

    return {"sub_claims": sub_claims}


# ── Node 2: Retrieval Agent ───────────────────────────────────────────────

def retrieve_evidence_node(state: PipelineState) -> dict:
    """Retrieve evidence for each sub-claim from ChromaDB."""
    sub_claims = state["sub_claims"]

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    all_evidence = []
    for sc in sub_claims:
        query = sc.get("query", sc.get("sub_claim", ""))
        hits = search(collection, query, top_k=5)

        passages = [
            {
                "source": f"PMID:{h['metadata']['pmid']}",
                "title": h["metadata"]["title"],
                "passage": h["text"],
                "relevance_score": round(1.0 - h["distance"], 3),
            }
            for h in hits
        ]

        all_evidence.append({
            "sub_claim": sc.get("sub_claim", query),
            "evidence": passages[:3],  # top-3 per sub-claim
        })

    evidence_json = json.dumps(all_evidence, indent=2)
    return {"evidence": all_evidence, "evidence_json": evidence_json}


# ── Node 3: Evidence Reviewer ─────────────────────────────────────────────

def review_evidence_node(state: PipelineState) -> dict:
    """Review evidence quality, flag contradictions and gaps."""
    claim = state["claim"]
    evidence_json = state["evidence_json"]

    response = _call(
        f'Review the following evidence for the claim: "{claim}"\n\n'
        f"Evidence by sub-claim:\n{evidence_json}\n\n"
        "Provide your structured review.",
        EVIDENCE_REVIEWER_SYSTEM,
        state,
    )

    review = _parse_json(response["content"], fallback={
        "summary": "Unable to parse review",
        "flags": [],
        "evidence_strength": "UNKNOWN",
        "key_findings": [],
        "recommendation": "",
    })

    review_data = {
        **review,
        "evidence_by_sub_claim": json.loads(evidence_json),
    }
    review_json = json.dumps(review_data, indent=2)

    return {"review": review, "review_json": review_json}


# ── Node 4: Verdict Agent ─────────────────────────────────────────────────

def generate_verdict_node(state: PipelineState) -> dict:
    """Generate final verdict from reviewed evidence."""
    claim = state["claim"]
    review_json = state["review_json"]

    response = _call(
        f'Generate a verdict for the claim: "{claim}"\n\n'
        f"Evidence review:\n{review_json}\n\n"
        "Provide your verdict, explanation, and cited evidence.",
        VERDICT_SYSTEM,
        state,
    )

    verdict = _parse_json(response["content"], fallback={
        "verdict": "INSUFFICIENT_EVIDENCE",
        "explanation": response["content"],
        "evidence": [],
    })

    return {"verdict": verdict}
