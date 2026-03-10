"""Shared state definition for the LangGraph pipeline."""

from typing import TypedDict


class PipelineState(TypedDict, total=False):
    """State passed through the LangGraph pipeline nodes."""
    claim: str
    sub_claims: list[dict]       # [{sub_claim, query}, ...]
    evidence: list[dict]         # [{sub_claim, evidence: [{source, passage, ...}]}, ...]
    evidence_json: str           # Serialised evidence for review/verdict prompts
    review: dict                 # {summary, flags, evidence_strength, key_findings, recommendation}
    review_json: str             # Serialised review for verdict prompt
    verdict: dict                # {verdict, explanation, evidence}
    model: str                   # LLM model ID
    provider: str                # LLM provider (anthropic, openai, ollama)
    _loop_count: int             # Rerouting loop counter
