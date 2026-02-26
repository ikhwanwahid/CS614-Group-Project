"""Automated grounding rate — % of factual statements traceable to retrieved evidence."""

import json

from src.shared.llm import call_llm

GROUNDING_SYSTEM_PROMPT = """You are an expert at evaluating whether factual claims are grounded in evidence.

You will be given:
1. An explanation from a health claim fact-checker
2. The evidence passages that were retrieved

Your task:
1. Extract every distinct factual statement from the explanation (not opinions or hedging language)
2. For each statement, determine if it is GROUNDED (traceable to a specific evidence passage) or UNGROUNDED (not supported by any evidence passage)

Respond ONLY with valid JSON:
{
    "statements": [
        {
            "statement": "the factual statement",
            "grounded": true or false,
            "evidence_source": "PMID or source if grounded, null if not"
        }
    ],
    "total_statements": int,
    "grounded_count": int,
    "grounding_rate": float (0.0 to 1.0)
}"""


def compute_grounding_rate(explanation: str, evidence: list[dict]) -> dict:
    """Compute the grounding rate for a single explanation.

    Returns dict with individual statement assessments and overall rate.
    """
    evidence_text = "\n\n".join(
        f"[{e.get('source', 'Unknown')}]: {e.get('passage', '')}"
        for e in evidence
    )

    prompt = (
        f"Explanation to evaluate:\n{explanation}\n\n"
        f"Evidence passages:\n{evidence_text}"
    )

    response = call_llm(prompt, system=GROUNDING_SYSTEM_PROMPT, max_tokens=2048)

    try:
        result = json.loads(response["content"])
    except json.JSONDecodeError:
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response["content"], re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {
                "statements": [],
                "total_statements": 0,
                "grounded_count": 0,
                "grounding_rate": 0.0,
            }

    result["_tokens"] = response["input_tokens"] + response["output_tokens"]
    return result


def score_pipeline_grounding(results: list[dict], pipeline_key: str) -> list[dict]:
    """Compute grounding rate for all results for a given pipeline.

    Args:
        results: List of comparison results from compare.py.
        pipeline_key: 'p1' or 'p6'.

    Returns:
        List of grounding dicts, one per claim.
    """
    scored = []
    for r in results:
        p = r[pipeline_key]
        grounding = compute_grounding_rate(
            explanation=p["explanation"],
            evidence=p["evidence"],
        )
        scored.append({
            "claim": r["claim"],
            "pipeline": pipeline_key.upper(),
            "grounding_rate": grounding.get("grounding_rate", 0.0),
            "total_statements": grounding.get("total_statements", 0),
            "grounded_count": grounding.get("grounded_count", 0),
            "details": grounding.get("statements", []),
        })
    return scored


def summarize_grounding(scored: list[dict]) -> dict:
    """Compute average grounding rate across all claims."""
    rates = [s["grounding_rate"] for s in scored]
    total_statements = sum(s["total_statements"] for s in scored)
    total_grounded = sum(s["grounded_count"] for s in scored)
    return {
        "avg_grounding_rate": round(sum(rates) / len(rates), 3) if rates else 0.0,
        "overall_grounding_rate": round(total_grounded / total_statements, 3) if total_statements else 0.0,
        "total_statements": total_statements,
        "total_grounded": total_grounded,
    }
