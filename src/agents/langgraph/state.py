"""LangGraph pipeline state — shared TypedDict passed between all nodes."""

from typing import TypedDict


class PipelineState(TypedDict, total=False):
    """Shared state dict passed through every node in the LangGraph pipeline.

    Fields are populated progressively as each node runs:
      - claim:       set before graph.invoke()
      - sub_claims:  set by parse_claim_node
      - evidence:    set by retrieve_evidence_node
      - review:      set by review_evidence_node
      - verdict:     set by generate_verdict_node
    """

    claim: str
    sub_claims: list[dict]   # [{"sub_claim": str, "query": str}, ...]
    evidence: list[dict]     # [{"sub_claim": str, "evidence": [...]}, ...]
    review: dict             # ReviewedEvidence.model_dump()
    verdict: dict            # VerdictOutput.model_dump()
