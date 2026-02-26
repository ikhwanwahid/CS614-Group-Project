"""Compare P1 and P6 pipeline outputs side-by-side."""

import json
import time
from pathlib import Path

from src.pipelines.p1_naive_single.pipeline import run as run_p1
from src.pipelines.p6_adv_multi.pipeline import run as run_p6


def load_test_claims(path: str = "data/test_claims.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compare(claims: list[dict]) -> list[dict]:
    """Run each claim through P1 and P6, return comparison results."""
    results = []
    total = len(claims)

    for idx, item in enumerate(claims, 1):
        claim = item["claim"]
        expected = item.get("expected_verdict", "N/A")

        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] Claim: {claim}")
        print(f"       Expected: {expected}")
        print(f"{'='*70}")

        # Run P1
        print("  Running P1 (Naive RAG + Single-Pass)...")
        t0 = time.time()
        p1_result = run_p1(claim)
        p1_time = time.time() - t0
        p1_match = "✓" if p1_result["verdict"] == expected else "✗"
        print(f"  P1 done in {p1_time:.1f}s — verdict: {p1_result['verdict']} {p1_match}")

        # Run P6
        print("  Running P6 (Advanced RAG + Multi-Agent)...")
        t0 = time.time()
        p6_result = run_p6(claim)
        p6_time = time.time() - t0
        p6_match = "✓" if p6_result["verdict"] == expected else "✗"
        print(f"  P6 done in {p6_time:.1f}s — verdict: {p6_result['verdict']} {p6_match}")

        verdict_agree = p1_result["verdict"] == p6_result["verdict"]
        print(f"  P1 vs P6 agree: {verdict_agree}")

        results.append({
            "claim": claim,
            "expected_verdict": expected,
            "difficulty": item.get("difficulty", ""),
            "p1": p1_result,
            "p6": p6_result,
            "verdict_match": verdict_agree,
            "p1_correct": p1_result["verdict"] == expected,
            "p6_correct": p6_result["verdict"] == expected,
        })

    return results


def print_summary(results: list[dict]):
    """Print a summary table of results."""
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    # Verdict table
    print(f"{'Claim':<50} {'Expected':<15} {'P1':<15} {'P6':<15}")
    print("-" * 95)
    for r in results:
        claim_short = r["claim"][:48]
        p1_mark = "✓" if r["p1_correct"] else "✗"
        p6_mark = "✓" if r["p6_correct"] else "✗"
        print(f"{claim_short:<50} {r['expected_verdict']:<15} {r['p1']['verdict']:<13}{p1_mark} {r['p6']['verdict']:<13}{p6_mark}")

    # Accuracy
    p1_correct = sum(1 for r in results if r["p1_correct"])
    p6_correct = sum(1 for r in results if r["p6_correct"])
    total = len(results)
    print(f"\nVerdict accuracy:  P1: {p1_correct}/{total}  |  P6: {p6_correct}/{total}")

    # Agreement
    agree = sum(1 for r in results if r["verdict_match"])
    print(f"P1-P6 agreement:  {agree}/{total}")

    # Latency & cost
    p1_latency = sum(r["p1"]["metadata"]["latency_seconds"] for r in results)
    p6_latency = sum(r["p6"]["metadata"]["latency_seconds"] for r in results)
    p1_cost = sum(r["p1"]["metadata"]["estimated_cost_usd"] for r in results)
    p6_cost = sum(r["p6"]["metadata"]["estimated_cost_usd"] for r in results)
    p1_tokens = sum(r["p1"]["metadata"]["total_tokens"] for r in results)
    p6_tokens = sum(r["p6"]["metadata"]["total_tokens"] for r in results)

    print(f"\nTotal latency:    P1: {p1_latency:.1f}s  |  P6: {p6_latency:.1f}s  ({p6_latency/p1_latency:.1f}x)")
    print(f"Total tokens:     P1: {p1_tokens}  |  P6: {p6_tokens}  ({p6_tokens/p1_tokens:.1f}x)")
    print(f"Total cost:       P1: ${p1_cost:.4f}  |  P6: ${p6_cost:.4f}  ({p6_cost/p1_cost:.1f}x)")


def main():
    claims = load_test_claims()
    results = compare(claims)

    print_summary(results)

    # Save full results
    output_path = Path("results/comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
