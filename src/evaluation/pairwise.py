"""Pairwise comparison of explanations from two pipeline configurations."""

import json
import re

from src.shared.llm import call_llm

PAIRWISE_SYSTEM_PROMPT = """You are an expert evaluator comparing two health claim fact-checking explanations.

You will see Explanation A and Explanation B for the same claim. They are anonymised — you do not know which pipeline produced them.

Compare them on these dimensions:
1. FAITHFULNESS: Which better grounds its claims in the provided evidence?
2. SPECIFICITY: Which cites more specific studies, statistics, or details?
3. COMPLETENESS: Which covers more relevant aspects of the claim?
4. NUANCE: Which better acknowledges limitations and complexity?

For each dimension, choose: "A", "B", or "tie".
Then choose an overall winner: "A", "B", or "tie".

Respond ONLY with valid JSON:
{
    "faithfulness": {"winner": "A" or "B" or "tie", "reason": "brief explanation"},
    "specificity": {"winner": "A" or "B" or "tie", "reason": "brief explanation"},
    "completeness": {"winner": "A" or "B" or "tie", "reason": "brief explanation"},
    "nuance": {"winner": "A" or "B" or "tie", "reason": "brief explanation"},
    "overall": {"winner": "A" or "B" or "tie", "reason": "brief explanation"}
}"""


def _parse_json(content: str) -> dict:
    """Parse JSON from LLM response, with markdown fallback."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {"overall": {"winner": "tie", "reason": "Parse error"}}


def compare_explanations(
    claim: str,
    explanation_a: str,
    explanation_b: str,
    evidence_a: list[dict],
    evidence_b: list[dict],
    model: str | None = None,
) -> dict:
    """Use an LLM judge to compare two explanations head-to-head.

    Presents both explanations (anonymised as A and B) and asks the judge
    which is better across faithfulness, specificity, completeness, and nuance.

    Args:
        claim: The health claim being evaluated.
        explanation_a: Explanation from configuration A.
        explanation_b: Explanation from configuration B.
        evidence_a: Evidence passages from configuration A.
        evidence_b: Evidence passages from configuration B.
        model: LLM model for the judge.

    Returns:
        Dict with 'winner' ('A', 'B', or 'tie'), 'reasoning', and per-dimension winners.
    """
    evidence_a_text = "\n".join(
        f"- [{e.get('source', '?')}]: {e.get('passage', '')[:200]}"
        for e in evidence_a
    )
    evidence_b_text = "\n".join(
        f"- [{e.get('source', '?')}]: {e.get('passage', '')[:200]}"
        for e in evidence_b
    )

    prompt = (
        f"Claim: {claim}\n\n"
        f"--- Explanation A ---\n{explanation_a}\n\n"
        f"Evidence A cited:\n{evidence_a_text}\n\n"
        f"--- Explanation B ---\n{explanation_b}\n\n"
        f"Evidence B cited:\n{evidence_b_text}"
    )

    response = call_llm(prompt, system=PAIRWISE_SYSTEM_PROMPT, model=model, max_tokens=1024)
    result = _parse_json(response["content"])
    result["_tokens"] = response["input_tokens"] + response["output_tokens"]
    return result


def compute_win_rates(comparisons: list[dict]) -> dict:
    """Aggregate pairwise comparisons into win rates.

    Args:
        comparisons: List of comparison result dicts from compare_explanations().

    Returns:
        Dict with 'a_wins', 'b_wins', 'ties', 'a_win_rate', 'b_win_rate'.
    """
    total = len(comparisons)
    if total == 0:
        return {"a_wins": 0, "b_wins": 0, "ties": 0, "a_win_rate": 0, "b_win_rate": 0, "total": 0}

    a_wins = sum(1 for c in comparisons if c.get("overall", {}).get("winner") == "A")
    b_wins = sum(1 for c in comparisons if c.get("overall", {}).get("winner") == "B")
    ties = total - a_wins - b_wins

    result = {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "total": total,
        "a_win_rate": round(a_wins / total, 4),
        "b_win_rate": round(b_wins / total, 4),
    }

    # Per-dimension breakdown
    for dim in ["faithfulness", "specificity", "completeness", "nuance"]:
        dim_a = sum(1 for c in comparisons if c.get(dim, {}).get("winner") == "A")
        dim_b = sum(1 for c in comparisons if c.get(dim, {}).get("winner") == "B")
        result[f"{dim}_a_wins"] = dim_a
        result[f"{dim}_b_wins"] = dim_b

    return result
