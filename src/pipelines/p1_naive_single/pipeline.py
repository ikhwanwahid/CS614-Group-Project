"""P1: Naive RAG + Single-Pass pipeline."""

import json
import re
import time

from src.shared.llm import call_llm
from src.shared.schema import FactCheckResult
from src.shared.vector_store import get_chroma_client, get_or_create_collection, search

SYSTEM_PROMPT = """You are a health claim fact-checker. Given the following evidence passages and a health claim, provide:
1. A verdict: SUPPORTED, UNSUPPORTED, OVERSTATED, or INSUFFICIENT_EVIDENCE
2. An explanation justifying your verdict (2-3 sentences)
3. Which evidence passages you relied on

Respond ONLY with valid JSON matching this schema:
{
    "verdict": "SUPPORTED | UNSUPPORTED | OVERSTATED | INSUFFICIENT_EVIDENCE",
    "explanation": "Your explanation here",
    "evidence": [
        {"source": "PMID or author reference", "passage": "key passage text", "relevance_score": 0.0-1.0}
    ]
}"""

# Claude Sonnet pricing (per token)
INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


def _parse_json_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "verdict": "INSUFFICIENT_EVIDENCE",
        "explanation": content,
        "evidence": [],
    }


def run(claim: str) -> dict:
    """Run P1 pipeline on a claim. Returns output matching shared schema."""
    start_time = time.time()

    # 1. Retrieve top-5 chunks via cosine similarity
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    hits = search(collection, claim, top_k=5)

    # 2. Format evidence passages
    passages = "\n\n".join(
        f"[{i+1}] (PMID: {h['metadata']['pmid']}) {h['text']}" for i, h in enumerate(hits)
    )

    # 3. Single LLM call
    prompt = f"Claim: {claim}\n\nEvidence:\n{passages}"
    response = call_llm(prompt, system=SYSTEM_PROMPT)

    input_tokens = response["input_tokens"]
    output_tokens = response["output_tokens"]
    total_tokens = input_tokens + output_tokens
    estimated_cost = (input_tokens * INPUT_COST_PER_TOKEN) + (output_tokens * OUTPUT_COST_PER_TOKEN)

    # 4. Parse response
    result_data = _parse_json_response(response["content"])

    latency = time.time() - start_time

    result = FactCheckResult(
        claim=claim,
        verdict=result_data.get("verdict", "INSUFFICIENT_EVIDENCE"),
        explanation=result_data.get("explanation", ""),
        evidence=result_data.get("evidence", []),
        metadata={
            "latency_seconds": round(latency, 2),
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "pipeline": "P1",
            "retrieval_method": "naive_rag",
            "agent_type": "single_pass",
        },
    )
    return result.model_dump()
