"""P6: Advanced RAG + Multi-Agent pipeline."""

import time

from src.agents.strands.orchestrator import run_pipeline
from src.shared.schema import FactCheckResult

# Bedrock Claude Sonnet pricing (per token)
INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


def run(claim: str) -> dict:
    """Run P6 pipeline on a claim. Returns output matching shared schema."""
    start_time = time.time()

    raw = run_pipeline(claim)

    latency = time.time() - start_time
    verdict_data = raw["verdict"]

    if "verdict" not in verdict_data:
        raise KeyError(f"Orchestrator response missing 'verdict' key: {verdict_data}")

    estimated_tokens = len(str(raw)) // 4
    estimated_cost = estimated_tokens * (INPUT_COST_PER_TOKEN + OUTPUT_COST_PER_TOKEN) / 2

    result = FactCheckResult(
        claim=claim,
        verdict=verdict_data["verdict"],
        explanation=verdict_data.get("explanation", ""),
        evidence=verdict_data.get("evidence", []),
        metadata={
            "latency_seconds": round(latency, 2),
            "total_tokens": estimated_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "pipeline": "P6",
            "retrieval_method": "advanced_rag",
            "agent_type": "multi_agent",
        },
    )
    return result.model_dump()
