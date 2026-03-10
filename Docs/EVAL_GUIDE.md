# Evaluation Guide — Member 6 (Output & Evaluation)

You own the **evaluation framework** — everything that measures how well the pipelines perform. Your work is used to compare all 12 experiments and determine which configurations actually make a meaningful difference.

---

## Your Deliverables

| # | Task | Files | Priority | Blocks |
|---|------|-------|----------|--------|
| 1 | Implement verdict accuracy metrics (Macro-F1) | `src/evaluation/metrics.py` | **Critical** | All experiment analysis |
| 2 | Implement McNemar's test | `src/evaluation/metrics.py` | **Critical** | Statistical significance |
| 3 | Implement bootstrap confidence intervals | `src/evaluation/metrics.py` | **Critical** | Error bars on all metrics |
| 4 | Implement pairwise explanation comparison | `src/evaluation/pairwise.py` | **High** | Head-to-head analysis |
| 5 | Update evaluation harness for configurable pipeline | `src/evaluation/run_eval.py` | **High** | Batch evaluation |
| 6 | Build results analysis notebook / report tables | `notebooks/` | **High** | Final report |

---

## What Already Exists (Working)

These are implemented and tested in the POC:

- **`src/evaluation/llm_judge.py`** — LLM-as-Judge scoring (4 dimensions: faithfulness, specificity, completeness, nuance, each 1-5). Uses Claude with a detailed rubric prompt.
- **`src/evaluation/grounding_rate.py`** — Automated grounding rate computation (% of factual statements traceable to evidence). Also uses Claude.
- **`src/evaluation/run_eval.py`** — Evaluation harness that runs both LLM judge and grounding on P1 vs P6 results.
- **`src/experiment_runner.py`** — Batch execution with resumption support, saves results per experiment.

## What Needs Implementation (Stubs)

- **`src/evaluation/metrics.py`** — `compute_verdict_accuracy()`, `mcnemar_test()`, `bootstrap_ci()`
- **`src/evaluation/pairwise.py`** — `compare_explanations()`, `compute_win_rates()`

---

## Task 1: Verdict Accuracy Metrics

**File:** `src/evaluation/metrics.py` — replace `compute_verdict_accuracy()` stub

### What it computes

Given pipeline verdicts and ground-truth labels, compute:
- Overall accuracy
- Per-class precision, recall, F1
- Macro-F1 (average F1 across all 4 verdict classes)
- Confusion matrix

### Implementation

```python
"""Quantitative evaluation metrics."""

from collections import Counter


VERDICT_CLASSES = ["SUPPORTED", "UNSUPPORTED", "OVERSTATED", "INSUFFICIENT_EVIDENCE"]


def compute_verdict_accuracy(results: list[dict], expected: list[dict]) -> dict:
    """Compute verdict accuracy, per-class metrics, and macro-F1.

    Args:
        results: List of pipeline output dicts with 'verdict' key.
        expected: List of dicts with 'expected_verdict' key.

    Returns:
        Dict with 'accuracy', 'macro_f1', 'per_class', 'confusion_matrix'.
    """
    predicted = [r["verdict"] for r in results]
    actual = [e["expected_verdict"] for e in expected]

    # Overall accuracy
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    accuracy = correct / len(actual) if actual else 0.0

    # Per-class precision/recall/F1
    per_class = {}
    f1_scores = []
    for cls in VERDICT_CLASSES:
        tp = sum(1 for p, a in zip(predicted, actual) if p == cls and a == cls)
        fp = sum(1 for p, a in zip(predicted, actual) if p == cls and a != cls)
        fn = sum(1 for p, a in zip(predicted, actual) if p != cls and a == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for a in actual if a == cls),
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Confusion matrix
    confusion = {a: {p: 0 for p in VERDICT_CLASSES} for a in VERDICT_CLASSES}
    for p, a in zip(predicted, actual):
        if a in confusion and p in confusion[a]:
            confusion[a][p] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "correct": correct,
        "total": len(actual),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
```

### Alternative: use sklearn

If you prefer, `sklearn.metrics` has all of this:

```python
from sklearn.metrics import classification_report, confusion_matrix, f1_score

report = classification_report(actual, predicted, output_dict=True)
macro_f1 = f1_score(actual, predicted, average="macro")
cm = confusion_matrix(actual, predicted, labels=VERDICT_CLASSES)
```

Either approach is fine. Manual computation avoids the sklearn dependency but sklearn is cleaner.

---

## Task 2: McNemar's Test

**File:** `src/evaluation/metrics.py` — replace `mcnemar_test()` stub

### What it does

McNemar's test checks whether two systems disagree in a statistically significant way. It looks at the 2x2 table of:
- Both correct
- A correct, B wrong
- A wrong, B correct
- Both wrong

If the off-diagonal cells (A right/B wrong vs A wrong/B right) are significantly different, the systems perform differently.

### Implementation

```python
def mcnemar_test(results_a: list[str], results_b: list[str], expected: list[str]) -> dict:
    """McNemar's test for paired comparison of two systems.

    Args:
        results_a: Verdicts from system A.
        results_b: Verdicts from system B.
        expected: Ground-truth verdicts.

    Returns:
        Dict with 'statistic', 'p_value', 'significant', 'contingency_table'.
    """
    # Build contingency table
    # b = A correct, B wrong
    # c = A wrong, B correct
    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct

    for a_pred, b_pred, true in zip(results_a, results_b, expected):
        a_correct = (a_pred == true)
        b_correct = (b_pred == true)

        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    # McNemar's statistic (with continuity correction)
    if b + c == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "b_count": b,
            "c_count": c,
            "note": "No discordant pairs",
        }

    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared distribution with 1 df
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "b_count": b,  # A correct, B wrong
        "c_count": c,  # A wrong, B correct
    }
```

### Dependencies

```bash
uv add scipy
```

### Interpretation

- `p_value < 0.05` → statistically significant difference between systems
- `b_count > c_count` → System A is better
- `b_count < c_count` → System B is better
- `b_count ≈ c_count` → No significant difference

---

## Task 3: Bootstrap Confidence Intervals

**File:** `src/evaluation/metrics.py` — replace `bootstrap_ci()` stub

### What it does

Compute confidence intervals for any metric (accuracy, F1, explanation quality scores) by resampling.

### Implementation

```python
import numpy as np


def bootstrap_ci(scores: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Compute bootstrap confidence interval.

    Args:
        scores: List of per-claim scores (e.g., [1, 0, 1, 1, 0] for accuracy
                or [3.5, 4.0, 2.5] for judge scores).
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (default 0.95 = 95%).

    Returns:
        Dict with 'mean', 'lower', 'upper', 'ci_level', 'std'.
    """
    scores = np.array(scores)
    n = len(scores)

    # Generate bootstrap samples
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)

    # Compute percentile CI
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)

    return {
        "mean": round(float(np.mean(scores)), 4),
        "lower": round(float(lower), 4),
        "upper": round(float(upper), 4),
        "ci_level": ci,
        "std": round(float(np.std(boot_means)), 4),
    }
```

### Usage examples

```python
# Accuracy CI: pass 1 for correct, 0 for wrong
accuracy_scores = [1 if r["verdict"] == r["expected_verdict"] else 0 for r in results]
ci = bootstrap_ci(accuracy_scores)
print(f"Accuracy: {ci['mean']:.1%} ({ci['lower']:.1%} – {ci['upper']:.1%})")

# Explanation quality CI
quality_scores = [s["scores"]["faithfulness"]["score"] for s in scored]
ci = bootstrap_ci(quality_scores)
print(f"Faithfulness: {ci['mean']:.2f} ({ci['lower']:.2f} – {ci['upper']:.2f})")
```

---

## Task 4: Pairwise Explanation Comparison

**File:** `src/evaluation/pairwise.py` — replace both stubs

### What it does

Present two anonymised explanations (A and B) to an LLM judge and ask which is better. This is more robust than absolute scoring because it eliminates scale bias.

### Implementation

```python
"""Pairwise comparison of explanations."""

import json
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
    "faithfulness": {"winner": "A"|"B"|"tie", "reason": "brief"},
    "specificity": {"winner": "A"|"B"|"tie", "reason": "brief"},
    "completeness": {"winner": "A"|"B"|"tie", "reason": "brief"},
    "nuance": {"winner": "A"|"B"|"tie", "reason": "brief"},
    "overall": {"winner": "A"|"B"|"tie", "reason": "brief"}
}"""


def compare_explanations(
    claim: str,
    explanation_a: str,
    explanation_b: str,
    evidence_a: list[dict],
    evidence_b: list[dict],
    model: str | None = None,
) -> dict:
    """Compare two explanations head-to-head using an LLM judge.

    Returns dict with per-dimension winners and overall winner.
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

    try:
        result = json.loads(response["content"])
    except json.JSONDecodeError:
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response["content"], re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {
                "overall": {"winner": "tie", "reason": "Parse error"},
            }

    result["_tokens"] = response["input_tokens"] + response["output_tokens"]
    return result


def compute_win_rates(comparisons: list[dict]) -> dict:
    """Aggregate pairwise comparisons into win rates.

    Args:
        comparisons: List of comparison dicts from compare_explanations().

    Returns:
        Dict with 'a_wins', 'b_wins', 'ties', 'a_win_rate', 'b_win_rate',
        and per-dimension breakdowns.
    """
    total = len(comparisons)
    if total == 0:
        return {"a_wins": 0, "b_wins": 0, "ties": 0, "a_win_rate": 0, "b_win_rate": 0}

    # Overall
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

    # Per-dimension
    for dim in ["faithfulness", "specificity", "completeness", "nuance"]:
        dim_a = sum(1 for c in comparisons if c.get(dim, {}).get("winner") == "A")
        dim_b = sum(1 for c in comparisons if c.get(dim, {}).get("winner") == "B")
        result[f"{dim}_a_wins"] = dim_a
        result[f"{dim}_b_wins"] = dim_b

    return result
```

### Position bias mitigation

LLM judges tend to prefer whichever explanation comes first. To mitigate this, run each comparison twice with A/B swapped and only count it as a win if both orderings agree:

```python
def compare_with_swap(claim, expl_a, expl_b, evidence_a, evidence_b):
    """Compare twice with A/B swapped to mitigate position bias."""
    result_ab = compare_explanations(claim, expl_a, expl_b, evidence_a, evidence_b)
    result_ba = compare_explanations(claim, expl_b, expl_a, evidence_b, evidence_a)

    # Flip result_ba winners
    for key in ["overall", "faithfulness", "specificity", "completeness", "nuance"]:
        if key in result_ba:
            w = result_ba[key].get("winner", "tie")
            result_ba[key]["winner"] = "B" if w == "A" else ("A" if w == "B" else "tie")

    # Only count as a win if both orderings agree
    final = {}
    for key in ["overall", "faithfulness", "specificity", "completeness", "nuance"]:
        w_ab = result_ab.get(key, {}).get("winner", "tie")
        w_ba = result_ba.get(key, {}).get("winner", "tie")
        final[key] = {"winner": w_ab if w_ab == w_ba else "tie"}

    return final
```

This doubles the LLM cost but is important for credibility.

---

## Task 5: Update Evaluation Harness

**File:** `src/evaluation/run_eval.py`

The current harness compares P1 vs P6 only. Update it to work with the experiment runner's output format.

### New flow

```python
"""Run full evaluation across experiment results."""

import json
from pathlib import Path
from src.evaluation.metrics import compute_verdict_accuracy, mcnemar_test, bootstrap_ci
from src.evaluation.llm_judge import score_explanation, summarize_scores
from src.evaluation.grounding_rate import compute_grounding_rate
from src.evaluation.pairwise import compare_explanations, compute_win_rates


def evaluate_experiment(experiment_file: str) -> dict:
    """Run all metrics on a single experiment's results."""
    with open(experiment_file) as f:
        data = json.load(f)

    results = data["results"]
    experiment_id = data["experiment_id"]

    # 1. Verdict accuracy
    expected = [{"expected_verdict": r["expected_verdict"]} for r in results]
    accuracy = compute_verdict_accuracy(results, expected)

    # 2. Bootstrap CI on accuracy
    accuracy_scores = [1 if r["verdict"] == r["expected_verdict"] else 0 for r in results]
    accuracy_ci = bootstrap_ci(accuracy_scores)

    # 3. LLM judge scores
    judge_scores = []
    for r in results:
        scores = score_explanation(r["claim"], r["verdict"], r["explanation"], r["evidence"])
        judge_scores.append(scores)

    # 4. Grounding rate
    grounding = []
    for r in results:
        gr = compute_grounding_rate(r["explanation"], r["evidence"])
        grounding.append(gr)

    return {
        "experiment_id": experiment_id,
        "verdict_accuracy": accuracy,
        "accuracy_ci": accuracy_ci,
        "judge_scores": judge_scores,
        "grounding": grounding,
    }


def compare_experiments(exp_a_file: str, exp_b_file: str) -> dict:
    """Compare two experiments with McNemar's test and pairwise comparison."""
    with open(exp_a_file) as f:
        data_a = json.load(f)
    with open(exp_b_file) as f:
        data_b = json.load(f)

    results_a = data_a["results"]
    results_b = data_b["results"]
    expected = [r["expected_verdict"] for r in results_a]

    # McNemar's test
    verdicts_a = [r["verdict"] for r in results_a]
    verdicts_b = [r["verdict"] for r in results_b]
    mcnemar = mcnemar_test(verdicts_a, verdicts_b, expected)

    # Pairwise explanation comparison
    comparisons = []
    for ra, rb in zip(results_a, results_b):
        comp = compare_explanations(
            ra["claim"], ra["explanation"], rb["explanation"],
            ra["evidence"], rb["evidence"],
        )
        comparisons.append(comp)
    win_rates = compute_win_rates(comparisons)

    return {
        "experiment_a": data_a["experiment_id"],
        "experiment_b": data_b["experiment_id"],
        "mcnemar": mcnemar,
        "win_rates": win_rates,
    }
```

---

## Task 6: Results Analysis

Build tables and charts for the final report. Key comparisons:

### Tables to generate

1. **Main results table** — All 12 experiments with accuracy, Macro-F1, explanation quality, grounding rate, latency, cost
2. **Chunking ablation** — E1 vs E2 vs E3 vs E4 (same retrieval/agent/model, different chunking)
3. **Agent architecture comparison** — E2 vs E5 vs E6 vs E7 (same RAG, different agents)
4. **Model comparison** — E2 vs E8 vs E9 vs E10 (same RAG/agent, different models)
5. **Statistical significance matrix** — McNemar p-values for all key pairs
6. **Pairwise win rates** — Head-to-head explanation quality

### Charts to generate

- Bar chart: Macro-F1 by experiment
- Bar chart: Explanation quality (4 dimensions) by experiment
- Scatter: Accuracy vs latency (cost-quality frontier)
- Heatmap: Confusion matrix per experiment
- Error bars: Bootstrap CIs on accuracy

---

## How Evaluation Gets Called

The experiment runner saves results to `results/experiments/E1.json`, `E2.json`, etc. Each file has this structure:

```json
{
  "experiment_id": "E1",
  "config": {
    "name": "Baseline (P1 equivalent)",
    "chunking_strategy": "fixed",
    "retrieval_method": "naive",
    "agent_architecture": "single_pass",
    "model": "claude-sonnet-4"
  },
  "total_claims": 120,
  "correct": 78,
  "accuracy": 0.65,
  "results": [
    {
      "claim": "Vaccines cause autism",
      "verdict": "UNSUPPORTED",
      "explanation": "Multiple studies...",
      "evidence": [...],
      "metadata": {...},
      "expected_verdict": "UNSUPPORTED",
      "correct": true
    }
  ]
}
```

Your evaluation code reads these files and produces the analysis.

---

## Understanding the Evaluation Metrics

### Verdict Accuracy (Macro-F1)

Why Macro-F1 and not just accuracy?

- Accuracy can be misleading with imbalanced classes (if 60% of claims are UNSUPPORTED, always guessing UNSUPPORTED gives 60% accuracy)
- Macro-F1 averages F1 across all 4 verdict classes equally, so performance on rare classes (INSUFFICIENT_EVIDENCE) matters as much as common ones

### LLM-as-Judge

The judge scores 4 dimensions of the pipeline's own output:

- **Faithfulness:** Are claims in the explanation grounded in the retrieved evidence? (Not: are they factually correct in general)
- **Specificity:** Does it cite specific studies, numbers, dates?
- **Completeness:** Does it address all aspects of the claim?
- **Nuance:** Does it acknowledge limitations and complexity?

Important: The judge evaluates the explanation against its own evidence, NOT against a reference explanation. No gold-standard explanations are needed.

### Grounding Rate

What % of factual statements in the explanation can be traced to a specific retrieved passage. A pipeline with 95% grounding is less likely to hallucinate than one with 60%.

### McNemar's Test

Tells you if the difference between two systems is statistically significant or just noise. With 120+ claims, even a 5% accuracy difference might be significant.

### Pairwise Comparison

More robust than absolute scoring for explanation quality. "Is A better than B?" is easier for a judge to answer than "Rate A on a 1-5 scale."

---

## Dependencies to Add

```bash
uv add scipy numpy
# sklearn is optional but helpful
uv add scikit-learn
```

---

## Files Reference

| File | Status | Owner |
|------|--------|-------|
| `src/evaluation/llm_judge.py` | Done — working | — |
| `src/evaluation/grounding_rate.py` | Done — working | — |
| `src/evaluation/metrics.py` | **Stub → implement** | You |
| `src/evaluation/pairwise.py` | **Stub → implement** | You |
| `src/evaluation/run_eval.py` | Needs update for new format | You |
| `src/experiment_runner.py` | Done — saves results | — |
| `results/experiments/` | Will contain E1.json, E2.json, ... | — |
