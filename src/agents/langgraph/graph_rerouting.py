"""LangGraph orchestration with rerouting — adaptive evidence loop.

After the Evidence Reviewer flags gaps, the graph can loop back to
the Retrieval Agent with refined queries instead of proceeding.
"""

import json

from langgraph.graph import StateGraph, END

from src.agents.langgraph.state import PipelineState
from src.agents.langgraph.nodes import (
    parse_claim_node,
    retrieve_evidence_node,
    review_evidence_node,
    generate_verdict_node,
    _call,
    _parse_json,
)

MAX_RETRIEVAL_LOOPS = 3


def _should_reroute(state: PipelineState) -> str:
    """Decide whether to reroute to retrieval or proceed to verdict."""
    review = state.get("review", {})
    loop_count = state.get("_loop_count", 0)

    strength = review.get("evidence_strength", "UNKNOWN").upper()

    # Proceed to verdict if evidence is strong enough or max loops reached
    if strength in ("STRONG", "MODERATE") or loop_count >= MAX_RETRIEVAL_LOOPS:
        return "generate_verdict"

    # Reroute: refine queries and retrieve again
    return "refine_queries"


def refine_queries_node(state: PipelineState) -> dict:
    """Use the reviewer's feedback to generate better search queries."""
    review = state.get("review", {})
    sub_claims = state.get("sub_claims", [])
    flags = review.get("flags", [])

    # Ask LLM to refine queries based on gaps
    gap_info = json.dumps(flags, indent=2) if flags else "No specific flags"
    sc_info = json.dumps(sub_claims, indent=2)

    response = _call(
        f"The evidence reviewer found these issues:\n{gap_info}\n\n"
        f"Current sub-claims and queries:\n{sc_info}\n\n"
        "Generate improved search queries for sub-claims that had weak evidence. "
        "Return the same JSON format with updated queries.\n"
        "Respond ONLY with valid JSON: [{\"sub_claim\": \"...\", \"query\": \"...\"}]",
        "You are a medical search specialist. Refine search queries to find better evidence.",
        state,
        max_tokens=1024,
    )

    refined = _parse_json(response["content"], fallback=sub_claims)
    if isinstance(refined, dict):
        refined = refined.get("sub_claims", sub_claims)

    loop_count = state.get("_loop_count", 0) + 1

    return {"sub_claims": refined, "_loop_count": loop_count}


def build_rerouting_graph():
    """Build and compile the fact-checking graph with rerouting."""
    graph = StateGraph(PipelineState)

    graph.add_node("parse_claim", parse_claim_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("review_evidence", review_evidence_node)
    graph.add_node("refine_queries", refine_queries_node)
    graph.add_node("generate_verdict", generate_verdict_node)

    graph.set_entry_point("parse_claim")
    graph.add_edge("parse_claim", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "review_evidence")

    # Conditional: reroute or proceed
    graph.add_conditional_edges(
        "review_evidence",
        _should_reroute,
        {
            "refine_queries": "refine_queries",
            "generate_verdict": "generate_verdict",
        },
    )
    graph.add_edge("refine_queries", "retrieve_evidence")
    graph.add_edge("generate_verdict", END)

    return graph.compile()


def run_pipeline_rerouting(
    claim: str,
    model: str | None = None,
    provider: str = "anthropic",
) -> dict:
    """Run the LangGraph pipeline with adaptive rerouting.

    Args:
        claim: The health claim to fact-check.
        model: LLM model ID.
        provider: LLM provider.

    Returns:
        Dict with pipeline results + rerouting metadata.
    """
    graph = build_rerouting_graph()
    initial_state: PipelineState = {
        "claim": claim,
        "model": model,
        "provider": provider,
    }
    result = graph.invoke(initial_state)
    return result
