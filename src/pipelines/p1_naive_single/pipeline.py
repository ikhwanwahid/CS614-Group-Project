"""P1: Naive RAG + single-pass pipeline (thin wrapper over configurable pipeline)."""

from src.pipelines.configurable import run_experiment


def run(claim: str) -> dict:
    """Run P1 pipeline on a claim. Returns output matching shared schema."""
    return run_experiment(
        claim,
        chunking_strategy="fixed",
        retrieval_method="naive",
        agent_architecture="single_pass",
        model="claude-sonnet-4-20250514",
    )
