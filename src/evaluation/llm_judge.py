"""LLM-as-judge rubric scoring for explanation quality."""

import json

from src.shared.llm import call_llm

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of health claim fact-checking explanations.

You will be given:
- A health claim
- The pipeline's verdict
- The pipeline's explanation
- The evidence passages the pipeline cited

Score the explanation on these 4 dimensions, each from 1 (lowest) to 5 (highest):

1. FAITHFULNESS: Are all factual claims in the explanation grounded in the retrieved evidence?
   1 = Makes claims not supported by any evidence
   3 = Most claims grounded, but some unsupported assertions
   5 = Every factual claim is traceable to the evidence

2. SPECIFICITY: Does the explanation cite specific studies, statistics, sample sizes, or dates?
   1 = Vague, generic statements with no citations
   3 = Some specific references but mostly general
   5 = Cites specific studies (by PMID), sample sizes, effect sizes, confidence intervals

3. COMPLETENESS: Does the explanation address all relevant dimensions of the claim?
   1 = Misses key aspects of what the claim asserts
   3 = Covers the main point but misses important nuances
   5 = Addresses all relevant dimensions including scope, mechanism, and strength of effect

4. NUANCE: Does the explanation acknowledge limitations, caveats, and evidence quality?
   1 = Binary verdict with no caveats
   3 = Some acknowledgment of complexity
   5 = Acknowledges limitations, population specificity, evidence hierarchy, and areas of uncertainty

Respond ONLY with valid JSON:
{
    "faithfulness": {"score": 1-5, "rationale": "brief explanation"},
    "specificity": {"score": 1-5, "rationale": "brief explanation"},
    "completeness": {"score": 1-5, "rationale": "brief explanation"},
    "nuance": {"score": 1-5, "rationale": "brief explanation"}
}"""


def score_explanation(claim: str, verdict: str, explanation: str, evidence: list[dict]) -> dict:
    """Score a single explanation using LLM-as-judge.

    Returns dict with scores and rationales for each dimension.
    """
    evidence_text = "\n".join(
        f"- [{e.get('source', 'Unknown')}]: {e.get('passage', '')[:300]}"
        for e in evidence
    )

    prompt = (
        f"Health Claim: {claim}\n\n"
        f"Verdict: {verdict}\n\n"
        f"Explanation:\n{explanation}\n\n"
        f"Evidence cited:\n{evidence_text}"
    )

    response = call_llm(prompt, system=JUDGE_SYSTEM_PROMPT, max_tokens=1024)

    try:
        scores = json.loads(response["content"])
    except json.JSONDecodeError:
        # Try extracting from markdown block
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response["content"], re.DOTALL)
        if match:
            scores = json.loads(match.group(1))
        else:
            scores = {
                "faithfulness": {"score": 0, "rationale": "Parse error"},
                "specificity": {"score": 0, "rationale": "Parse error"},
                "completeness": {"score": 0, "rationale": "Parse error"},
                "nuance": {"score": 0, "rationale": "Parse error"},
            }

    scores["_tokens"] = response["input_tokens"] + response["output_tokens"]
    return scores


def score_pipeline_results(results: list[dict], pipeline_key: str) -> list[dict]:
    """Score all results for a given pipeline (e.g. 'p1' or 'p6').

    Args:
        results: List of comparison results from compare.py.
        pipeline_key: 'p1' or 'p6'.

    Returns:
        List of score dicts, one per claim.
    """
    scored = []
    for r in results:
        p = r[pipeline_key]
        scores = score_explanation(
            claim=r["claim"],
            verdict=p["verdict"],
            explanation=p["explanation"],
            evidence=p["evidence"],
        )
        scored.append({
            "claim": r["claim"],
            "pipeline": pipeline_key.upper(),
            "verdict": p["verdict"],
            "scores": scores,
        })
    return scored


def summarize_scores(scored: list[dict]) -> dict:
    """Compute average scores across all claims for a pipeline."""
    dimensions = ["faithfulness", "specificity", "completeness", "nuance"]
    totals = {d: 0.0 for d in dimensions}
    count = len(scored)

    for s in scored:
        for d in dimensions:
            totals[d] += s["scores"][d]["score"]

    averages = {d: round(totals[d] / count, 2) for d in dimensions}
    averages["overall"] = round(sum(averages.values()) / len(dimensions), 2)
    return averages
