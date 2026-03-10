"""Evidence hierarchy ranking.

Ranks retrieved evidence by study design quality:
systematic reviews > meta-analyses > RCTs > observational > case reports.
"""

import re

# Evidence hierarchy tiers (higher = stronger evidence)
HIERARCHY = {
    "systematic_review": 6,
    "meta-analysis": 5,
    "rct": 4,
    "cohort": 3,
    "case_control": 2,
    "cross_sectional": 2,
    "case_report": 1,
    "expert_opinion": 1,
    "unknown": 0,
}


def _detect_study_type(text: str) -> str:
    """Detect study type from passage text."""
    lower = text.lower()
    if "systematic review" in lower:
        return "systematic_review"
    if "meta-analysis" in lower or "meta analysis" in lower:
        return "meta-analysis"
    if "randomized" in lower or "randomised" in lower or "rct" in lower:
        return "rct"
    if "cohort" in lower:
        return "cohort"
    if "case-control" in lower or "case control" in lower:
        return "case_control"
    if "cross-sectional" in lower:
        return "cross_sectional"
    if "case report" in lower:
        return "case_report"
    return "unknown"


def rank_by_evidence_hierarchy(passages: list[dict]) -> list[dict]:
    """Re-rank passages by evidence hierarchy (study design quality).

    Args:
        passages: List of passage dicts with 'text' and optionally 'score' or 'metadata'.

    Returns:
        Passages sorted by evidence hierarchy tier, then by existing relevance score.
    """
    for p in passages:
        # Check metadata first, then detect from text
        study_type = (p.get("metadata") or {}).get("study_type", "")
        if not study_type or study_type == "unknown":
            study_type = _detect_study_type(p.get("text", ""))
        p["study_type"] = study_type
        p["hierarchy_tier"] = HIERARCHY.get(study_type, 0)

    # Sort by hierarchy tier (desc), then by existing score (desc)
    return sorted(
        passages,
        key=lambda p: (p["hierarchy_tier"], p.get("score", p.get("rerank_score", 0.0))),
        reverse=True,
    )
