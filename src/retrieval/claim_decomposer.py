"""Claim decomposition into verifiable sub-claims.

Note: For agent-based architectures, claim decomposition is handled by the
Claim Parser agent (src/agents/strands/claim_parser.py). This module provides
a standalone version for single-pass pipelines that need sub-claim decomposition
without the full agent framework.
"""


def decompose_claim(claim: str, model: str | None = None) -> list[dict]:
    """Decompose a health claim into verifiable sub-claims with search queries.

    Args:
        claim: Raw health claim text.
        model: LLM model to use for decomposition.

    Returns:
        List of dicts with 'sub_claim' and 'query' keys.
    """
    raise NotImplementedError(
        "Standalone claim decomposer not yet implemented — Agent pair (Members 4 & 5).\n"
        "For agent-based pipelines, use src/agents/strands/claim_parser.py instead."
    )
