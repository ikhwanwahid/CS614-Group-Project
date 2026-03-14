"""Configurable pipeline — single entry point for all experiment configurations.

Dispatches to the appropriate chunking strategy, retrieval method, agent
architecture, and model based on the experiment config.
"""

import json
import re
import time

from src.shared.schema import FactCheckResult

# Supported values for each axis
CHUNKING_STRATEGIES = ("fixed", "semantic", "section_aware", "recursive")
RETRIEVAL_METHODS = ("naive", "hybrid", "hybrid_reranked")
AGENT_ARCHITECTURES = ("single_pass", "strands_multi", "langgraph_multi", "strands_rerouting")
MODELS = ("claude-sonnet-4", "gpt-4o-mini", "claude-haiku", "llama-3.1-8b", "llama-3.1-8b-ft", "llama-3.1-70b")

SYSTEM_PROMPT = """You are a health claim fact-checker. Given the following evidence passages and a health claim, provide:
1. A verdict: SUPPORTED, UNSUPPORTED, OVERSTATED, or INSUFFICIENT_EVIDENCE
2. An explanation justifying your verdict (2-3 sentences)
3. Which evidence passages you relied on

Respond ONLY with valid JSON matching this schema:
{
    "verdict": "SUPPORTED | UNSUPPORTED | OVERSTATED | INSUFFICIENT_EVIDENCE",
    "explanation": "Your explanation here",
    "evidence": [
        {"source": "PMID or author reference", "passage": "key passage text", "relevance_score": 0.0-1.0}
    ]
}"""


def get_collection(chunking_strategy: str):
    """Get or create and populate a ChromaDB collection for the given chunking strategy."""
    from src.shared.vector_store import get_chroma_client, get_or_create_collection, add_documents
    from src.shared.corpus_loader import load_corpus
    from src.chunking import chunk_corpus

    collection_name = f"health_corpus_{chunking_strategy}"
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name=collection_name)

    if collection.count() > 0:
        return collection

    corpus = load_corpus()
    chunks = chunk_corpus(corpus, strategy=chunking_strategy)
    add_documents(collection, chunks)
    return collection


def _sanitize_json(text: str) -> str:
    """Fix common invalid JSON escapes produced by LLMs (e.g. \\%)."""
    return re.sub(r'\\(?=[^"\\bfnrtu/])', r'\\\\', text)


def parse_json_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    for text in [content, _sanitize_json(content)]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        raw = match.group(1)
        for text in [raw, _sanitize_json(raw)]:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"(Failed to parse JSON response) {content}")


def run_experiment(
    claim: str,
    chunking_strategy: str = "fixed",
    retrieval_method: str = "naive",
    agent_architecture: str = "single_pass",
    model: str = "claude-sonnet-4",
) -> dict:
    """Run a single claim through a configured pipeline.

    Args:
        claim: The health claim to fact-check.
        chunking_strategy: One of 'fixed', 'semantic', 'section_aware', 'recursive'.
        retrieval_method: One of 'naive', 'hybrid', 'hybrid_reranked'.
        agent_architecture: One of 'single_pass', 'strands_multi', 'langgraph_multi', 'strands_rerouting'.
        model: Model identifier string.

    Returns:
        Dict matching FactCheckResult schema with experiment config in metadata.
    """
    start_time = time.time()

    # Dispatch based on architecture
    if agent_architecture == "single_pass":
        raw = _run_single_pass(claim, chunking_strategy, retrieval_method, model)
    elif agent_architecture == "strands_multi":
        raw = _run_strands_multi(claim, model)
    elif agent_architecture == "langgraph_multi":
        raw = _run_langgraph_multi(claim, model)
    elif agent_architecture == "strands_rerouting":
        raw = _run_strands_rerouting(claim, model)
    else:
        raise ValueError(f"Unknown agent architecture: {agent_architecture}")

    latency = time.time() - start_time
    estimated_tokens = len(str(raw)) // 4
    estimated_cost = estimated_tokens * 9e-6  # rough average

    if "verdict" not in raw:
        raise KeyError(f"LLM response missing 'verdict' key: {raw}")

    result = FactCheckResult(
        claim=claim,
        verdict=raw["verdict"],
        explanation=raw.get("explanation", ""),
        evidence=raw.get("evidence", []),
        metadata={
            "latency_seconds": round(latency, 2),
            "total_tokens": estimated_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "pipeline": f"{chunking_strategy}_{retrieval_method}_{agent_architecture}_{model}",
            "retrieval_method": retrieval_method,
            "agent_type": agent_architecture,
        },
    )

    output = result.model_dump()
    output["experiment_config"] = {
        "chunking_strategy": chunking_strategy,
        "retrieval_method": retrieval_method,
        "agent_architecture": agent_architecture,
        "model": model,
    }
    return output


def _run_single_pass(claim: str, chunking_strategy: str, retrieval_method: str, model: str) -> dict:
    """Single-pass: retrieve evidence + one LLM call for verdict."""
    if retrieval_method == "naive":
        from src.pipelines.p1_naive_single.pipeline import run as run_p1
        result = run_p1(claim, model=model)
        return {
            "verdict": result["verdict"],
            "explanation": result["explanation"],
            "evidence": result["evidence"],
        }

    # Hybrid / reranked retrieval with single-pass
    collection = get_collection(chunking_strategy)

    from src.retrieval.hybrid import retrieve_hybrid
    hits = retrieve_hybrid(claim, collection, top_k=10)

    if retrieval_method == "hybrid_reranked":
        from src.retrieval.reranker import rerank
        hits = rerank(claim, hits, top_k=5)
    else:
        hits = hits[:5]

    # Format evidence passages for the LLM
    passages = "\n\n".join(
        f"[{i+1}] (PMID: {h['metadata'].get('pmid', 'N/A')}) {h['text']}"
        for i, h in enumerate(hits)
    )

    from src.shared.llm import call_llm
    prompt = f"Claim: {claim}\n\nEvidence:\n{passages}"
    response = call_llm(prompt, system=SYSTEM_PROMPT, model=model)

    result_data = parse_json_response(response["content"])
    if "verdict" not in result_data:
        raise KeyError(f"LLM response missing 'verdict' key: {result_data}")
    return {
        "verdict": result_data["verdict"],
        "explanation": result_data.get("explanation", ""),
        "evidence": result_data.get("evidence", []),
    }


def _run_strands_multi(claim: str, model: str) -> dict:
    """Strands 4-agent sequential pipeline."""
    from src.agents.strands.orchestrator import run_pipeline
    raw = run_pipeline(claim)
    return raw["verdict"]


def _run_langgraph_multi(claim: str, model: str) -> dict:
    """LangGraph graph-based multi-agent pipeline."""
    raise NotImplementedError(
        "LangGraph multi-agent pipeline not yet implemented — Agent pair (Members 4 & 5).\n"
        "Wire up src/agents/langgraph/graph.py with the same 4-agent flow."
    )


def _run_strands_rerouting(claim: str, model: str) -> dict:
    """Strands multi-agent with gated rerouting (adaptive loop)."""
    from src.agents.strands.orchestrator_gated import run_pipeline_with_gating
    raw = run_pipeline_with_gating(claim)
    return raw["verdict"]
