"""Evidence hierarchy ranking.

Ranks retrieved evidence by study design quality:
systematic reviews > meta-analyses > RCTs > observational > case reports.
"""


def rank_by_evidence_hierarchy(passages: list[dict]) -> list[dict]:
    """Re-rank passages by evidence hierarchy (study design quality).

    Args:
        passages: List of passage dicts with 'text' and 'metadata'.

    Returns:
        Passages sorted by evidence hierarchy tier, then by relevance score.
    """
    raise NotImplementedError(
        "Evidence hierarchy ranker not yet implemented — RAG pair (Members 2 & 3).\n"
        "Approach: classify each passage's study type from text/metadata "
        "(systematic review, RCT, observational, etc.), assign tier scores, "
        "combine with relevance score for final ranking."
    )
