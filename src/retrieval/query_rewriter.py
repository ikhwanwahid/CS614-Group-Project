"""LLM-based query rewriting and expansion."""

from src.shared.llm import call_llm


def rewrite_query(claim: str, model: str | None = None) -> str:
    """Rewrite a health claim into an optimised search query.

    Expands with medical synonyms (e.g., 'cholecalciferol' for 'Vitamin D',
    'SARS-CoV-2' for 'COVID') and reformulates for retrieval.

    Args:
        claim: Raw health claim text.
        model: LLM model to use for rewriting.

    Returns:
        Rewritten search query string.
    """
    prompt = (
        "Rewrite this health claim as a PubMed search query. "
        "Add medical synonyms and MeSH terms. Return ONLY the query, nothing else.\n\n"
        f"Claim: {claim}"
    )
    response = call_llm(
        prompt,
        system="You are a medical librarian specialising in PubMed literature search.",
        model=model,
        max_tokens=200,
    )
    return response["content"].strip()
