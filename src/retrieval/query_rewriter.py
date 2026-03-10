"""LLM-based query rewriting and expansion."""

import json

from src.shared.llm import call_llm

REWRITER_SYSTEM = (
    "You are a medical librarian. Rewrite the given health claim into an optimised "
    "PubMed search query. Add medical synonyms and MeSH terms where appropriate. "
    "Return ONLY the rewritten query string, nothing else."
)


def rewrite_query(claim: str, model: str | None = None, provider: str = "anthropic") -> str:
    """Rewrite a health claim into an optimised search query.

    Expands with medical synonyms (e.g., 'cholecalciferol' for 'Vitamin D',
    'SARS-CoV-2' for 'COVID') and reformulates for retrieval.

    Args:
        claim: Raw health claim text.
        model: LLM model to use for rewriting.
        provider: LLM provider.

    Returns:
        Rewritten search query string.
    """
    response = call_llm(
        prompt=f"Rewrite as a PubMed search query:\n\n{claim}",
        system=REWRITER_SYSTEM,
        model=model,
        provider=provider,
        max_tokens=200,
    )
    return response["content"].strip().strip('"').strip("'")
