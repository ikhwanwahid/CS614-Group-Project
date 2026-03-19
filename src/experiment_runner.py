"""Experiment runner — batch execution of pipeline configurations over claim sets.

Runs a specified experiment configuration across all claims, saves results
with full provenance, and supports resumption of interrupted runs.
"""

import json
import time
import traceback
from pathlib import Path

from src.pipelines.configurable import run_experiment


BEST_CHUNKING_STRATEGY = "recursive"
BEST_RETRIEVAL_METHOD = "naive"


# Default experiment configurations for chunking-first evaluation.
EXPERIMENT_CONFIGS = {
    "E1": {
        "name": "Fixed chunking + naive RAG",
        "chunking_strategy": "fixed",
        "retrieval_method": "naive",
        "agent_architecture": "single_pass",
        "model": "claude-sonnet-4-20250514",
    },
    "E2": {
        "name": "Section-aware chunking + naive RAG",
        "chunking_strategy": "section_aware",
        "retrieval_method": "naive",
        "agent_architecture": "single_pass",
        "model": "claude-sonnet-4-20250514",
    },
    "E3": {
        "name": "Semantic chunking + naive RAG",
        "chunking_strategy": "semantic",
        "retrieval_method": "naive",
        "agent_architecture": "single_pass",
        "model": "claude-sonnet-4-20250514",
    },
    "E4": {
        "name": "Recursive chunking + naive RAG",
        "chunking_strategy": "recursive",
        "retrieval_method": "naive",
        "agent_architecture": "single_pass",
        "model": "claude-sonnet-4-20250514",
    },
    "E5": {
        "name": "Best chunking + hybrid RAG",
        "chunking_strategy": BEST_CHUNKING_STRATEGY,
        "retrieval_method": "hybrid",
        "agent_architecture": "single_pass",
        "model": "claude-sonnet-4-20250514",
    },
    "E6": {
        "name": "Best chunking + hybrid reranked RAG",
        "chunking_strategy": BEST_CHUNKING_STRATEGY,
        "retrieval_method": "hybrid_reranked",
        "agent_architecture": "single_pass",
        "model": "claude-sonnet-4-20250514",
    },
    "E7": {
        "name": "Best RAG + Strands agents",
        "chunking_strategy": BEST_CHUNKING_STRATEGY,
        "retrieval_method": BEST_RETRIEVAL_METHOD,
        "agent_architecture": "strands_multi",
        "model": "claude-sonnet-4-20250514",
    },
    "E8": {
        "name": "Best RAG + LangGraph agents",
        "chunking_strategy": BEST_CHUNKING_STRATEGY,
        "retrieval_method": BEST_RETRIEVAL_METHOD,
        "agent_architecture": "langgraph_multi",
        "model": "claude-sonnet-4-20250514",
    },
    "E9": {
        "name": "Best RAG + rerouting",
        "chunking_strategy": BEST_CHUNKING_STRATEGY,
        "retrieval_method": BEST_RETRIEVAL_METHOD,
        "agent_architecture": "strands_rerouting",
        "model": "claude-sonnet-4-20250514",
    },
    "E10": {
        "name": "GPT-4o-mini + single-pass",
        "chunking_strategy": BEST_CHUNKING_STRATEGY,
        "retrieval_method": BEST_RETRIEVAL_METHOD,
        "agent_architecture": "single_pass",
        "model": "gpt-4o-mini",
    },
    "E11": {
        "name": "Llama 3.1 8B baseline",
        "chunking_strategy": BEST_CHUNKING_STRATEGY,
        "retrieval_method": BEST_RETRIEVAL_METHOD,
        "agent_architecture": "single_pass",
        "model": "llama3.1:8b",
    },
    "E12": {
        "name": "GPT-4o-mini + agents",
        "chunking_strategy": BEST_CHUNKING_STRATEGY,
        "retrieval_method": BEST_RETRIEVAL_METHOD,
        "agent_architecture": "strands_multi",
        "model": "gpt-4o-mini",
    },
    "E13": {
        "name": "Budget baseline",
        "chunking_strategy": "fixed",
        "retrieval_method": "naive",
        "agent_architecture": "single_pass",
        "model": "gpt-4o-mini",
    },
}


def run_batch(
    experiment_id: str,
    claims: list[dict],
    output_dir: str = "results/experiments",
    resume: bool = True,
    force_rebuild_chunks: bool = False,
) -> list[dict]:
    """Run an experiment configuration across all claims.

    Args:
        experiment_id: Key from EXPERIMENT_CONFIGS (e.g., 'E1').
        claims: List of claim dicts with 'claim' and either native SciFact
            'evidence_label' or the older 'expected_verdict'.
        output_dir: Directory to save results.
        resume: If True, skip claims already processed in a previous run.

    Returns:
        List of result dicts (one per claim).
    """
    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_id}. Choose from: {list(EXPERIMENT_CONFIGS.keys())}")

    config = EXPERIMENT_CONFIGS[experiment_id]
    pipeline_config = {k: v for k, v in config.items() if k != "name"}
    pipeline_config["force_rebuild_chunks"] = force_rebuild_chunks

    print(
        f"Starting {experiment_id}: {config['name']} "
        f"(chunking={config['chunking_strategy']}, retrieval={config['retrieval_method']}, claims={len(claims)})"
    )

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

    results = list(existing_results)

    for i, item in enumerate(claims):
        claim = item["claim"]
        expected_verdict = item.get("expected_verdict", "INSUFFICIENT_EVIDENCE")
        if claim in processed_claims:
            continue

        t0 = time.perf_counter()
        try:
            result = run_experiment(claim, **pipeline_config)
            result["expected_verdict"] = expected_verdict
            result["correct"] = result["verdict"] == expected_verdict
            results.append(result)
            elapsed = time.perf_counter() - t0
            print(f"[{experiment_id}] {i + 1}/{len(claims)}: {claim[:55]}... → {result['verdict']} ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"[{experiment_id}] {i + 1}/{len(claims)}: {claim[:55]}... → ERROR ({elapsed:.2f}s): {type(e).__name__}: {e}")
            results.append({
                "claim": claim,
                "verdict": "ERROR",
                "explanation": f"{type(e).__name__}: {e}",
                "evidence": [],
                "metadata": {
                    "pipeline": experiment_id,
                    "retrieval_method": config["retrieval_method"],
                    "agent_type": config["agent_architecture"],
                },
                "expected_verdict": expected_verdict,
                "correct": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(limit=8),
                },
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

    # Optional local debug hook; keep disabled so normal runs stay quiet.
    # print("Starting experiment runner")

    if len(sys.argv) < 2:
        print("Usage: python -m src.experiment_runner <experiment_id> [--force-rebuild-chunks] [--no-resume]")
        print("\nAvailable experiments:")
        list_experiments()
        sys.exit(1)

    experiment_id = sys.argv[1].upper()
    force_rebuild_chunks = "--force-rebuild-chunks" in sys.argv[2:]
    resume = "--no-resume" not in sys.argv[2:]

    with open("data/test_claims.json") as f:
        claims = json.load(f)

    results = run_batch(
        experiment_id,
        claims,
        resume=resume,
        force_rebuild_chunks=force_rebuild_chunks,
    )
    correct = sum(1 for r in results if r.get("correct", False))
    print(f"\nDone. Accuracy: {correct}/{len(results)}")
