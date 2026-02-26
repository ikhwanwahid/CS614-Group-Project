"""Shared output schema for all pipelines."""

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    source: str
    passage: str
    relevance_score: float


class PipelineMetadata(BaseModel):
    latency_seconds: float
    total_tokens: int
    estimated_cost_usd: float
    pipeline: str  # "P1" through "P6"
    retrieval_method: str | None = None
    agent_type: str | None = None


class FactCheckResult(BaseModel):
    claim: str
    verdict: str  # SUPPORTED | UNSUPPORTED | OVERSTATED | INSUFFICIENT_EVIDENCE
    explanation: str
    evidence: list[EvidenceItem]
    metadata: PipelineMetadata


VALID_VERDICTS = {"SUPPORTED", "UNSUPPORTED", "OVERSTATED", "INSUFFICIENT_EVIDENCE"}
