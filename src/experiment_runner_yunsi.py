"""Experiment runner — batch execution of pipeline configurations over claim sets.

Runs a specified experiment configuration across all claims, saves results
with full provenance, and supports resumption of interrupted runs.
"""

import json
import time
from pathlib import Path

from src.pipelines.configurable import run_experiment


# Default experiment configurations (from proposal v5, Section 3.5)
EXPERIMENT_CONFIGS = {
    "E1": {
        "name": "Baseline (P1 equivalent)",
        "chunking_strategy": "fixed",
        "retrieval_method": "hybrid_reranked",
        "agent_architecture": "single_pass",
        "model": "llama3.1:8b",
    },
    "E2": {
        "name": "Best RAG + single-pass",
        "chunking_strategy": "semantic",
        "retrieval_method": "hybrid_reranked",
        "agent_architecture": "single_pass",
        "model": "llama3.1:8b",
    },
    "E3": {
        "name": "Section-aware chunking",
        "chunking_strategy": "section_aware",
        "retrieval_method": "hybrid_reranked",
        "agent_architecture": "single_pass",
        "model": "llama3.1:8b",
    },
    "E4": {
        "name": "Recursive chunking + metadata",
        "chunking_strategy": "recursive",
        "retrieval_method": "hybrid_reranked",
        "agent_architecture": "single_pass",
        "model": "llama3.1:8b",
    }
}


def run_batch(
    experiment_id: str,
    claims: list[dict],
    output_dir: str = "results/experiments",
    resume: bool = True,
) -> list[dict]:
    """Run an experiment configuration across all claims.

    Args:
        experiment_id: Key from EXPERIMENT_CONFIGS (e.g., 'E1').
        claims: List of claim dicts with 'claim' and 'expected_verdict'.
        output_dir: Directory to save results.
        resume: If True, skip claims already processed in a previous run.

    Returns:
        List of result dicts (one per claim).
    """
    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_id}. Choose from: {list(EXPERIMENT_CONFIGS.keys())}")

    config = EXPERIMENT_CONFIGS[experiment_id]
    pipeline_config = {k: v for k, v in config.items() if k != "name"}

    output_path = Path(output_dir) / f"{experiment_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resumption
    existing_results = []
    processed_claims = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
            existing_results = data.get("results", [])
            processed_claims = {r["claim"] for r in existing_results}
        print(f"Resuming {experiment_id}: {len(processed_claims)} claims already processed")

    results = list(existing_results)

    for i, item in enumerate(claims):
        claim = item["claim"]
        if claim in processed_claims:
            continue

        print(f"[{experiment_id}] {i+1}/{len(claims)}: {claim[:60]}...", end=" ", flush=True)

        try:
            result = run_experiment(claim, **pipeline_config)
            result["expected_verdict"] = item.get("expected_verdict", "")
            result["correct"] = result["verdict"] == item.get("expected_verdict", "")
            results.append(result)
            print(f"→ {result['verdict']} ({result['metadata']['latency_seconds']}s)")
        except Exception as e:
            print(f"→ ERROR: {e}")
            results.append({
                "claim": claim,
                "verdict": "ERROR",
                "explanation": str(e),
                "evidence": [],
                "metadata": {"pipeline": experiment_id},
                "expected_verdict": item.get("expected_verdict", ""),
                "correct": False,
                "error": str(e),
            })

        # Save after each claim for resumption safety
        _save_results(output_path, experiment_id, config, results)

    return results


def _save_results(path: Path, experiment_id: str, config: dict, results: list[dict]):
    """Save results with experiment metadata."""
    output = {
        "experiment_id": experiment_id,
        "config": config,
        "total_claims": len(results),
        "correct": sum(1 for r in results if r.get("correct", False)),
        "accuracy": sum(1 for r in results if r.get("correct", False)) / max(len(results), 1),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def list_experiments() -> None:
    """Print all available experiment configurations."""
    for eid, config in EXPERIMENT_CONFIGS.items():
        print(f"  {eid}: {config['name']}")
        print(f"      chunking={config['chunking_strategy']}  retrieval={config['retrieval_method']}  "
              f"agent={config['agent_architecture']}  model={config['model']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.experiment_runner <experiment_id>")
        print("\nAvailable experiments:")
        list_experiments()
        sys.exit(1)

    experiment_id = sys.argv[1].upper()

    with open("data/test_claims.json") as f:
        claims = json.load(f)

    print(f"Running {experiment_id} on {len(claims)} claims...")
    results = run_batch(experiment_id, claims)
    correct = sum(1 for r in results if r.get("correct", False))
    print(f"\nDone. Accuracy: {correct}/{len(results)}")
