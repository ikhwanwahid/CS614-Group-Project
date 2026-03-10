"""Run full evaluation on experiment results.

Supports both the legacy P1-vs-P6 comparison format and the new
per-experiment format from experiment_runner.py.
"""

import json
import sys
from pathlib import Path

from src.evaluation.llm_judge import score_explanation, summarize_scores
from src.evaluation.grounding_rate import compute_grounding_rate, summarize_grounding
from src.evaluation.metrics import compute_verdict_accuracy, mcnemar_test, bootstrap_ci


def evaluate_experiment(experiment_file: str) -> dict:
    """Run all metrics on a single experiment's results.

    Args:
        experiment_file: Path to experiment JSON (from experiment_runner).

    Returns:
        Dict with verdict_accuracy, accuracy_ci, judge_scores, grounding.
    """
    with open(experiment_file) as f:
        data = json.load(f)

    results = data["results"]
    experiment_id = data["experiment_id"]

    print(f"\n{'=' * 70}")
    print(f"EVALUATING {experiment_id}: {data['config']['name']}")
    print(f"{'=' * 70}")

    # 1. Verdict accuracy
    expected = [{"expected_verdict": r["expected_verdict"]} for r in results]
    accuracy = compute_verdict_accuracy(results, expected)
    print(f"\nVerdict Accuracy: {accuracy['correct']}/{accuracy['total']} = {accuracy['accuracy']:.1%}")
    print(f"Macro-F1: {accuracy['macro_f1']:.4f}")

    # 2. Bootstrap CI on accuracy
    accuracy_scores = [1.0 if r["verdict"] == r["expected_verdict"] else 0.0 for r in results]
    accuracy_ci = bootstrap_ci(accuracy_scores)
    print(f"95% CI: [{accuracy_ci['lower']:.4f}, {accuracy_ci['upper']:.4f}]")

    # 3. LLM judge scores
    print("\nScoring explanations with LLM judge...")
    judge_scores = []
    for r in results:
        if r.get("verdict") == "ERROR":
            continue
        scores = score_explanation(r["claim"], r["verdict"], r["explanation"], r["evidence"])
        judge_scores.append({"claim": r["claim"], "scores": scores})
        dims = {d: scores[d]["score"] for d in ["faithfulness", "specificity", "completeness", "nuance"]}
        print(f"  {r['claim'][:50]:<52} F={dims['faithfulness']} S={dims['specificity']} C={dims['completeness']} N={dims['nuance']}")

    # Summarise judge scores
    if judge_scores:
        avg_scores = {}
        for dim in ["faithfulness", "specificity", "completeness", "nuance"]:
            dim_scores = [s["scores"][dim]["score"] for s in judge_scores]
            avg_scores[dim] = round(sum(dim_scores) / len(dim_scores), 2)
        avg_scores["overall"] = round(sum(avg_scores.values()) / 4, 2)
        print(f"\nAvg scores: {avg_scores}")

    # 4. Grounding rate
    print("\nComputing grounding rates...")
    grounding = []
    for r in results:
        if r.get("verdict") == "ERROR":
            continue
        gr = compute_grounding_rate(r["explanation"], r["evidence"])
        grounding.append({
            "claim": r["claim"],
            "grounding_rate": gr.get("grounding_rate", 0.0),
            "total_statements": gr.get("total_statements", 0),
            "grounded_count": gr.get("grounded_count", 0),
        })
        print(f"  {r['claim'][:50]:<52} {gr.get('grounded_count', 0)}/{gr.get('total_statements', 0)} = {gr.get('grounding_rate', 0):.0%}")

    gr_summary = summarize_grounding(grounding) if grounding else {}
    print(f"\nAvg grounding rate: {gr_summary.get('avg_grounding_rate', 0):.1%}")

    return {
        "experiment_id": experiment_id,
        "config": data["config"],
        "verdict_accuracy": accuracy,
        "accuracy_ci": accuracy_ci,
        "judge_scores": judge_scores,
        "grounding": grounding,
    }


def compare_experiments(exp_a_file: str, exp_b_file: str) -> dict:
    """Compare two experiments with McNemar's test.

    Args:
        exp_a_file: Path to experiment A JSON.
        exp_b_file: Path to experiment B JSON.

    Returns:
        Dict with McNemar's test results.
    """
    with open(exp_a_file) as f:
        data_a = json.load(f)
    with open(exp_b_file) as f:
        data_b = json.load(f)

    results_a = data_a["results"]
    results_b = data_b["results"]
    expected = [r["expected_verdict"] for r in results_a]

    verdicts_a = [r["verdict"] for r in results_a]
    verdicts_b = [r["verdict"] for r in results_b]

    mcnemar = mcnemar_test(verdicts_a, verdicts_b, expected)

    print(f"\nMcNemar's test: {data_a['experiment_id']} vs {data_b['experiment_id']}")
    print(f"  Statistic: {mcnemar['statistic']}")
    print(f"  p-value: {mcnemar['p_value']}")
    print(f"  Significant: {mcnemar['significant']}")

    return {
        "experiment_a": data_a["experiment_id"],
        "experiment_b": data_b["experiment_id"],
        "mcnemar": mcnemar,
    }


def main():
    """Run evaluation — supports both old and new formats."""
    if len(sys.argv) > 1:
        # New format: evaluate specific experiments
        for path in sys.argv[1:]:
            result = evaluate_experiment(path)
            output_path = Path(path).with_suffix(".eval.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved evaluation to {output_path}")
        return

    # Legacy format: P1 vs P6 comparison
    comparison_path = Path("results/comparison.json")
    if not comparison_path.exists():
        print("No experiment files specified and results/comparison.json not found.")
        print("Usage: python -m src.evaluation.run_eval [experiment_file ...]")
        sys.exit(1)

    from src.evaluation.llm_judge import score_pipeline_results, summarize_scores
    from src.evaluation.grounding_rate import score_pipeline_grounding

    with open(comparison_path) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} comparison results\n")

    # LLM-as-Judge
    print("=" * 70)
    print("LLM-AS-JUDGE: Explanation Quality Scoring")
    print("=" * 70)

    print("\nScoring P1 explanations...")
    p1_scores = score_pipeline_results(results, "p1")
    print("\nScoring P6 explanations...")
    p6_scores = score_pipeline_results(results, "p6")

    p1_avg = summarize_scores(p1_scores)
    p6_avg = summarize_scores(p6_scores)

    print(f"\n{'Dimension':<20} {'P1':>6} {'P6':>6} {'Delta':>8}")
    print("-" * 42)
    for d in ["faithfulness", "specificity", "completeness", "nuance", "overall"]:
        delta = p6_avg[d] - p1_avg[d]
        sign = "+" if delta > 0 else ""
        print(f"{d.capitalize():<20} {p1_avg[d]:>6.2f} {p6_avg[d]:>6.2f} {sign}{delta:>7.2f}")

    # Grounding Rate
    print(f"\n{'=' * 70}")
    print("GROUNDING RATE")
    print("=" * 70)

    print("\nComputing P1 grounding rates...")
    p1_grounding = score_pipeline_grounding(results, "p1")
    print("\nComputing P6 grounding rates...")
    p6_grounding = score_pipeline_grounding(results, "p6")

    p1_gr = summarize_grounding(p1_grounding)
    p6_gr = summarize_grounding(p6_grounding)

    print(f"\n{'Metric':<30} {'P1':>10} {'P6':>10}")
    print("-" * 52)
    print(f"{'Avg grounding rate':<30} {p1_gr['avg_grounding_rate']:>9.1%} {p6_gr['avg_grounding_rate']:>9.1%}")

    # Save
    eval_results = {
        "explanation_quality": {"p1_averages": p1_avg, "p6_averages": p6_avg},
        "grounding_rate": {"p1_summary": p1_gr, "p6_summary": p6_gr},
    }
    output_path = Path("results/evaluation.json")
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
