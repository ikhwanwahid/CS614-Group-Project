"""Claim decomposition into verifiable sub-claims.

Standalone version for single-pass pipelines that need sub-claim decomposition
without the full Strands agent framework.
"""

import json
import re

from src.shared.llm import call_llm

DECOMPOSER_SYSTEM = """You are a medical claim decomposition specialist. Break down the health claim
into 2-4 specific, verifiable sub-claims. For each sub-claim, generate a targeted search query.

Respond ONLY with valid JSON:
[
    {"sub_claim": "specific verifiable statement", "query": "PubMed search query"}
]"""


def decompose_claim(
    claim: str,
    model: str | None = None,
    provider: str = "anthropic",
) -> list[dict]:
    """Decompose a health claim into verifiable sub-claims with search queries.

    Args:
        claim: Raw health claim text.
        model: LLM model to use for decomposition.
        provider: LLM provider.

    Returns:
        List of dicts with 'sub_claim' and 'query' keys.
    """
    response = call_llm(
        prompt=f"Decompose this health claim into verifiable sub-claims:\n\n{claim}",
        system=DECOMPOSER_SYSTEM,
        model=model,
        provider=provider,
        max_tokens=1024,
    )

    content = response["content"]

    # Parse JSON response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: return the original claim as a single sub-claim
    return [{"sub_claim": claim, "query": claim}]
