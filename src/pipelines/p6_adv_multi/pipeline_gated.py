"""P6-gated: Advanced RAG + Multi-Agent pipeline with confidence gating."""

import time

from src.agents.strands.orchestrator_gated import run_pipeline_with_gating
from src.shared.schema import FactCheckResult

# Bedrock Claude Sonnet pricing (per token)
INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


def run(claim: str) -> dict:
    """Run the P6-gated pipeline on a claim.

    Returns a dict matching the shared ``FactCheckResult`` schema with an
    extra ``gating_info`` key containing gate decision metadata.
    """
    start_time = time.time()

    raw = run_pipeline_with_gating(claim)

    latency = time.time() - start_time
    verdict_data = raw["verdict"]
    gating = raw.get("gating", {})

    if "verdict" not in verdict_data:
        raise KeyError(f"Orchestrator response missing 'verdict' key: {verdict_data}")

    estimated_tokens = len(str(raw)) // 4
    estimated_cost = estimated_tokens * (INPUT_COST_PER_TOKEN + OUTPUT_COST_PER_TOKEN) / 2

    path = gating.get("path", "FULL_PIPELINE")
    agent_type = "multi_agent_gated_short" if path == "SHORT_CIRCUIT" else "multi_agent_gated_full"

    result = FactCheckResult(
        claim=claim,
        verdict=verdict_data["verdict"],
        explanation=verdict_data.get("explanation", ""),
        evidence=verdict_data.get("evidence", []),
        metadata={
            "latency_seconds": round(latency, 2),
            "total_tokens": estimated_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "pipeline": "P6-gated",
            "retrieval_method": "advanced_rag",
            "agent_type": agent_type,
        },
    )

    output = result.model_dump()
    output["gating_info"] = gating
    return output
