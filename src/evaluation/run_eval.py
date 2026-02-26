"""Run full evaluation on saved comparison results."""

import json
from pathlib import Path

from src.evaluation.llm_judge import score_pipeline_results, summarize_scores
from src.evaluation.grounding_rate import score_pipeline_grounding, summarize_grounding


def main():
    # Load comparison results
    with open("results/comparison.json") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} comparison results\n")

    # === LLM-as-Judge: Explanation Quality ===
    print("=" * 70)
    print("LLM-AS-JUDGE: Explanation Quality Scoring")
    print("=" * 70)

    print("\nScoring P1 explanations...")
    p1_scores = score_pipeline_results(results, "p1")
    for s in p1_scores:
        dims = {d: s["scores"][d]["score"] for d in ["faithfulness", "specificity", "completeness", "nuance"]}
        print(f"  {s['claim'][:50]:<52} F={dims['faithfulness']} S={dims['specificity']} C={dims['completeness']} N={dims['nuance']}")

    print("\nScoring P6 explanations...")
    p6_scores = score_pipeline_results(results, "p6")
    for s in p6_scores:
        dims = {d: s["scores"][d]["score"] for d in ["faithfulness", "specificity", "completeness", "nuance"]}
        print(f"  {s['claim'][:50]:<52} F={dims['faithfulness']} S={dims['specificity']} C={dims['completeness']} N={dims['nuance']}")

    p1_avg = summarize_scores(p1_scores)
    p6_avg = summarize_scores(p6_scores)

    print(f"\n{'Dimension':<20} {'P1':>6} {'P6':>6} {'Delta':>8}")
    print("-" * 42)
    for d in ["faithfulness", "specificity", "completeness", "nuance", "overall"]:
        delta = p6_avg[d] - p1_avg[d]
        sign = "+" if delta > 0 else ""
        print(f"{d.capitalize():<20} {p1_avg[d]:>6.2f} {p6_avg[d]:>6.2f} {sign}{delta:>7.2f}")

    # === Grounding Rate ===
    print(f"\n{'=' * 70}")
    print("GROUNDING RATE: Factual Statement Traceability")
    print("=" * 70)

    print("\nComputing P1 grounding rates...")
    p1_grounding = score_pipeline_grounding(results, "p1")
    for g in p1_grounding:
        print(f"  {g['claim'][:50]:<52} {g['grounded_count']}/{g['total_statements']} = {g['grounding_rate']:.0%}")

    print("\nComputing P6 grounding rates...")
    p6_grounding = score_pipeline_grounding(results, "p6")
    for g in p6_grounding:
        print(f"  {g['claim'][:50]:<52} {g['grounded_count']}/{g['total_statements']} = {g['grounding_rate']:.0%}")

    p1_gr_summary = summarize_grounding(p1_grounding)
    p6_gr_summary = summarize_grounding(p6_grounding)

    print(f"\n{'Metric':<30} {'P1':>10} {'P6':>10}")
    print("-" * 52)
    print(f"{'Avg grounding rate':<30} {p1_gr_summary['avg_grounding_rate']:>9.1%} {p6_gr_summary['avg_grounding_rate']:>9.1%}")
    print(f"{'Overall grounding rate':<30} {p1_gr_summary['overall_grounding_rate']:>9.1%} {p6_gr_summary['overall_grounding_rate']:>9.1%}")
    print(f"{'Total statements':<30} {p1_gr_summary['total_statements']:>10} {p6_gr_summary['total_statements']:>10}")
    print(f"{'Grounded statements':<30} {p1_gr_summary['total_grounded']:>10} {p6_gr_summary['total_grounded']:>10}")

    # === Save full evaluation results ===
    eval_results = {
        "explanation_quality": {
            "p1_scores": p1_scores,
            "p6_scores": p6_scores,
            "p1_averages": p1_avg,
            "p6_averages": p6_avg,
        },
        "grounding_rate": {
            "p1_details": p1_grounding,
            "p6_details": p6_grounding,
            "p1_summary": p1_gr_summary,
            "p6_summary": p6_gr_summary,
        },
    }

    output_path = Path("results/evaluation.json")
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nFull evaluation saved to {output_path}")


if __name__ == "__main__":
    main()
