"""LangGraph graph — assembles the 4-node pipeline and compiles it.

Usage:
    from src.agents.langgraph.graph import build_graph

    graph = build_graph()
    result = graph.invoke({"claim": "Vaccines cause autism"})
    print(result["verdict"])

The graph mirrors the Strands sequential pipeline exactly:
    parse_claim → retrieve_evidence → review_evidence → generate_verdict

Passing a model:
    result = graph.invoke({"claim": "...", "_model": "gpt-4o-mini"})

The special "_model" key in state is read by each node via _resolve_model().
If omitted, nodes default to claude-sonnet-4-20250514.
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
    """Build and compile the 4-node LangGraph fact-checking pipeline.

    Returns:
        A compiled LangGraph runnable (call .invoke() or .stream() on it).
    """
    graph = StateGraph(PipelineState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("parse_claim", parse_claim_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("review_evidence", review_evidence_node)
    graph.add_node("generate_verdict", generate_verdict_node)

    # ── Wire edges (sequential) ───────────────────────────────────────────────
    graph.set_entry_point("parse_claim")
    graph.add_edge("parse_claim", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "review_evidence")
    graph.add_edge("review_evidence", "generate_verdict")
    graph.add_edge("generate_verdict", END)

    return graph.compile()
