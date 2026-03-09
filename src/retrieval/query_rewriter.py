"""LLM-based query rewriting and expansion."""


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
    raise NotImplementedError(
        "Query rewriter not yet implemented — RAG pair (Members 2 & 3).\n"
        "Approach: single LLM call to expand claim with medical synonyms "
        "and reformulate as a search query."
    )
