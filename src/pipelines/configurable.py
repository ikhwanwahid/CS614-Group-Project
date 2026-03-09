"""Configurable pipeline — single entry point for all experiment configurations.

Dispatches to the appropriate chunking strategy, retrieval method, agent
architecture, and model based on the experiment config.
"""

import time

from src.shared.schema import FactCheckResult

# Supported values for each axis
CHUNKING_STRATEGIES = ("fixed", "semantic", "section_aware", "recursive")
RETRIEVAL_METHODS = ("naive", "hybrid", "hybrid_reranked")
AGENT_ARCHITECTURES = ("single_pass", "strands_multi", "langgraph_multi", "strands_rerouting")
MODELS = ("claude-sonnet-4", "gpt-4o-mini", "claude-haiku", "llama-3.1-8b", "llama-3.1-8b-ft", "llama-3.1-70b")


def run_experiment(
    claim: str,
    chunking_strategy: str = "fixed",
    retrieval_method: str = "naive",
    agent_architecture: str = "single_pass",
    model: str = "claude-sonnet-4",
) -> dict:
    """Run a single claim through a configured pipeline.

    Args:
        claim: The health claim to fact-check.
        chunking_strategy: One of 'fixed', 'semantic', 'section_aware', 'recursive'.
        retrieval_method: One of 'naive', 'hybrid', 'hybrid_reranked'.
        agent_architecture: One of 'single_pass', 'strands_multi', 'langgraph_multi', 'strands_rerouting'.
        model: Model identifier string.

    Returns:
        Dict matching FactCheckResult schema with experiment config in metadata.
    """
    start_time = time.time()

    # Dispatch based on architecture
    if agent_architecture == "single_pass":
        raw = _run_single_pass(claim, retrieval_method, model)
    elif agent_architecture == "strands_multi":
        raw = _run_strands_multi(claim, model)
    elif agent_architecture == "langgraph_multi":
        raw = _run_langgraph_multi(claim, model)
    elif agent_architecture == "strands_rerouting":
        raw = _run_strands_rerouting(claim, model)
    else:
        raise ValueError(f"Unknown agent architecture: {agent_architecture}")

    latency = time.time() - start_time
    estimated_tokens = len(str(raw)) // 4
    estimated_cost = estimated_tokens * 9e-6  # rough average

    result = FactCheckResult(
        claim=claim,
        verdict=raw.get("verdict", "INSUFFICIENT_EVIDENCE"),
        explanation=raw.get("explanation", ""),
        evidence=raw.get("evidence", []),
        metadata={
            "latency_seconds": round(latency, 2),
            "total_tokens": estimated_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "pipeline": f"{chunking_strategy}_{retrieval_method}_{agent_architecture}_{model}",
            "retrieval_method": retrieval_method,
            "agent_type": agent_architecture,
        },
    )

    output = result.model_dump()
    output["experiment_config"] = {
        "chunking_strategy": chunking_strategy,
        "retrieval_method": retrieval_method,
        "agent_architecture": agent_architecture,
        "model": model,
    }
    return output


def _run_single_pass(claim: str, retrieval_method: str, model: str) -> dict:
    """Single-pass: retrieve evidence + one LLM call for verdict."""
    # For now, delegate to the existing P1 pipeline for the baseline config
    if retrieval_method == "naive":
        from src.pipelines.p1_naive_single.pipeline import run as run_p1
        result = run_p1(claim)
        return {
            "verdict": result["verdict"],
            "explanation": result["explanation"],
            "evidence": result["evidence"],
        }

    # Hybrid / reranked retrieval with single-pass — not yet implemented
    raise NotImplementedError(
        f"Single-pass with retrieval_method='{retrieval_method}' not yet implemented.\n"
        "RAG pair (Members 2 & 3) will wire up hybrid retrieval + reranking "
        "into the single-pass flow."
    )


def _run_strands_multi(claim: str, model: str) -> dict:
    """Strands 4-agent sequential pipeline."""
    from src.agents.strands.orchestrator import run_pipeline
    raw = run_pipeline(claim)
    return raw["verdict"]


def _run_langgraph_multi(claim: str, model: str) -> dict:
    """LangGraph graph-based multi-agent pipeline."""
    raise NotImplementedError(
        "LangGraph multi-agent pipeline not yet implemented — Agent pair (Members 4 & 5).\n"
        "Wire up src/agents/langgraph/graph.py with the same 4-agent flow."
    )


def _run_strands_rerouting(claim: str, model: str) -> dict:
    """Strands multi-agent with rerouting (adaptive loop)."""
    raise NotImplementedError(
        "Rerouting agent architecture not yet implemented — Agent pair (Members 4 & 5).\n"
        "Evidence Reviewer loops back to Retrieval Agent if coverage is insufficient."
    )
