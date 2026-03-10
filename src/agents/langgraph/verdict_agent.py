"""LangGraph Agent: Verdict Agent node.

Implementation lives in nodes.py (generate_verdict_node).
This file re-exports for discoverability.
"""

from src.agents.langgraph.nodes import generate_verdict_node

__all__ = ["generate_verdict_node"]
