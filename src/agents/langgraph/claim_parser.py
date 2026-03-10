"""LangGraph Agent: Claim Parser node.

Implementation lives in nodes.py (parse_claim_node).
This file re-exports for discoverability.
"""

from src.agents.langgraph.nodes import parse_claim_node

__all__ = ["parse_claim_node"]
