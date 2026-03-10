"""LangGraph Agent: Retrieval Agent node.

Implementation lives in nodes.py (retrieve_evidence_node).
This file re-exports for discoverability.
"""

from src.agents.langgraph.nodes import retrieve_evidence_node

__all__ = ["retrieve_evidence_node"]
