"""LangGraph orchestration — builds and compiles the 4-node agent graph.

Same 4-agent flow as Strands (Claim Parser → Retrieval → Review → Verdict)
but orchestrated via LangGraph's StateGraph with explicit state passing.
"""

from langgraph.graph import StateGraph, END

from src.agents.langgraph.state import PipelineState
from src.agents.langgraph.nodes import (
    parse_claim_node,
    retrieve_evidence_node,
    review_evidence_node,
    generate_verdict_node,
)


def build_graph():
    """Build and compile the 4-node fact-checking graph."""
    graph = StateGraph(PipelineState)

    graph.add_node("parse_claim", parse_claim_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("review_evidence", review_evidence_node)
    graph.add_node("generate_verdict", generate_verdict_node)

    graph.set_entry_point("parse_claim")
    graph.add_edge("parse_claim", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "review_evidence")
    graph.add_edge("review_evidence", "generate_verdict")
    graph.add_edge("generate_verdict", END)

    return graph.compile()


def run_pipeline(
    claim: str,
    model: str | None = None,
    provider: str = "anthropic",
) -> dict:
    """Run the full LangGraph multi-agent pipeline.

    Args:
        claim: The health claim to fact-check.
        model: LLM model ID (e.g., 'claude-sonnet-4-20250514', 'gpt-4o-mini').
        provider: LLM provider ('anthropic', 'openai', 'ollama').

    Returns:
        Dict with 'sub_claims', 'evidence', 'review', 'verdict' keys.
    """
    graph = build_graph()
    initial_state: PipelineState = {
        "claim": claim,
        "model": model,
        "provider": provider,
    }
    result = graph.invoke(initial_state)
    return result
