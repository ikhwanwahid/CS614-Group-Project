"""Confidence gate — scores local ChromaDB evidence to decide whether to short-circuit."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.shared.vector_store import get_chroma_client, get_or_create_collection, search

# ---------------------------------------------------------------------------
# Tuneable thresholds
# ---------------------------------------------------------------------------
DISTANCE_RELEVANT = 0.45       # ChromaDB L2 distance below which a hit is "relevant"
TOP_K = 5                      # hits per sub-claim query
TOP_N_AVG = 3                  # how many top hits to average for quality
GATE_SCORE_THRESHOLD = 0.70    # overall score must be >= this
GATE_COVERAGE_THRESHOLD = 0.75 # fraction of sub-claims with adequate evidence


@dataclass
class SubClaimScore:
    """Score for a single sub-claim's local evidence."""
    sub_claim: str
    query: str
    relevant_hits: int          # how many hits have distance < DISTANCE_RELEVANT
    avg_distance_top_n: float   # average distance of the top-N hits
    quality: float              # normalised quality score 0-1


@dataclass
class ConfidenceAssessment:
    """Result of the confidence gate evaluation."""
    score: float                              # overall gate score 0-1
    is_high_confidence: bool                  # True → short-circuit
    coverage_ratio: float                     # fraction of sub-claims with evidence
    sub_claim_scores: list[SubClaimScore] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_subclaim(hits: list[dict]) -> tuple[int, float, float]:
    """Compute quality metrics for one sub-claim's ChromaDB hits.

    Returns (relevant_hit_count, avg_distance_top_n, quality_0_to_1).
    """
    if not hits:
        return 0, 1.0, 0.0

    relevant = sum(1 for h in hits if h["distance"] < DISTANCE_RELEVANT)
    top_n = hits[:TOP_N_AVG]
    avg_dist = sum(h["distance"] for h in top_n) / len(top_n)
    # Convert distance to a 0-1 quality signal (lower distance = higher quality).
    # Clamp so distances >= 1.0 map to 0 quality.
    quality = max(0.0, 1.0 - avg_dist)
    return relevant, round(avg_dist, 4), round(quality, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_local_confidence(
    sub_claims: list[dict],
    *,
    distance_threshold: float = DISTANCE_RELEVANT,
    gate_score: float = GATE_SCORE_THRESHOLD,
    gate_coverage: float = GATE_COVERAGE_THRESHOLD,
    top_k: int = TOP_K,
) -> tuple[ConfidenceAssessment, dict[str, list[dict]]]:
    """Search ChromaDB for every sub-claim and score local evidence quality.

    Args:
        sub_claims: List of dicts with ``sub_claim`` and ``query`` keys
            (same format as produced by the claim parser).
        distance_threshold: Max L2 distance to count a hit as relevant.
        gate_score: Minimum overall score to trigger the short-circuit.
        gate_coverage: Minimum fraction of sub-claims with >= 1 relevant hit.
        top_k: Number of hits to retrieve per query.

    Returns:
        A tuple of (ConfidenceAssessment, local_hits_by_subclaim).
        ``local_hits_by_subclaim`` maps each sub-claim text to its raw hits
        so the caller can reuse them without re-querying.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    sc_scores: list[SubClaimScore] = []
    local_hits: dict[str, list[dict]] = {}
    covered = 0

    for sc in sub_claims:
        query = sc["query"]
        sub_claim_text = sc["sub_claim"]
        hits = search(collection, query, top_k=top_k)
        local_hits[sub_claim_text] = hits

        relevant_count, avg_dist, quality = _score_subclaim(hits)
        if relevant_count > 0:
            covered += 1

        sc_scores.append(SubClaimScore(
            sub_claim=sub_claim_text,
            query=query,
            relevant_hits=relevant_count,
            avg_distance_top_n=avg_dist,
            quality=quality,
        ))

    n = len(sub_claims) if sub_claims else 1
    coverage_ratio = covered / n
    avg_quality = sum(s.quality for s in sc_scores) / n if sc_scores else 0.0

    # Overall score: 50 % average quality + 50 % coverage ratio
    overall = 0.5 * avg_quality + 0.5 * coverage_ratio
    is_high = overall >= gate_score and coverage_ratio >= gate_coverage

    reason_parts: list[str] = []
    reason_parts.append(f"score={overall:.2f} (threshold={gate_score})")
    reason_parts.append(f"coverage={coverage_ratio:.0%} (threshold={gate_coverage:.0%})")
    if is_high:
        reason_parts.append("→ HIGH confidence, short-circuit")
    else:
        reason_parts.append("→ LOW confidence, full pipeline")

    assessment = ConfidenceAssessment(
        score=round(overall, 4),
        is_high_confidence=is_high,
        coverage_ratio=round(coverage_ratio, 4),
        sub_claim_scores=sc_scores,
        reason="; ".join(reason_parts),
    )
    return assessment, local_hits
