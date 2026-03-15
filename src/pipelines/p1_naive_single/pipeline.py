"""P1: Naive RAG + Single-Pass pipeline."""

import time

from src.shared.llm import call_llm
from src.shared.schema import FactCheckResult
from src.shared.vector_store import get_chroma_client, get_or_create_collection, search
from src.pipelines.configurable import SYSTEM_PROMPT, parse_json_response

# Claude Sonnet pricing (per token)
INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


def run(claim: str, model: str | None = None) -> dict:
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
    llm_kwargs = {"system": SYSTEM_PROMPT}
    if model:
        llm_kwargs["model"] = model
    response = call_llm(prompt, **llm_kwargs)

    input_tokens = response["input_tokens"]
    output_tokens = response["output_tokens"]
    total_tokens = input_tokens + output_tokens
    estimated_cost = (input_tokens * INPUT_COST_PER_TOKEN) + (output_tokens * OUTPUT_COST_PER_TOKEN)

    # 4. Parse response
    result_data = parse_json_response(response["content"])

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
