"""Configurable pipeline — single entry point for all experiment configurations.

Dispatches to the appropriate chunking strategy, retrieval method, agent
architecture, and model based on the experiment config.
"""

import json
import re
import time

from src.shared.schema import FactCheckResult
from src.shared.llm import call_llm
from src.shared.vector_store import get_chroma_client, get_or_create_collection, search

# Supported values for each axis
CHUNKING_STRATEGIES = ("fixed", "semantic", "section_aware", "recursive")
RETRIEVAL_METHODS = ("naive", "hybrid", "hybrid_reranked")
AGENT_ARCHITECTURES = ("single_pass", "strands_multi", "langgraph_multi", "strands_rerouting")
MODELS = ("claude-sonnet-4", "gpt-4o-mini", "claude-haiku", "llama-3.1-8b", "llama-3.1-8b-ft", "llama-3.1-70b")

# Model name → (actual model ID, provider)
MODEL_REGISTRY = {
    "claude-sonnet-4": ("claude-sonnet-4-20250514", "anthropic"),
    "claude-haiku": ("claude-haiku-4-5-20251001", "anthropic"),
    "gpt-4o-mini": ("gpt-4o-mini", "openai"),
    "llama-3.1-8b": ("llama3.1:8b", "ollama"),
    "llama-3.1-8b-ft": ("llama3.1:8b", "ollama"),  # same base, different adapter
    "llama-3.1-70b": ("llama3.1:70b", "ollama"),
}

# Verdict prompt (shared across single-pass configurations)
VERDICT_SYSTEM = """You are a health claim fact-checker. Given the following evidence passages and a health claim, provide:
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


def _resolve_model(model: str) -> tuple[str, str]:
    """Resolve experiment model name to (model_id, provider)."""
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    # Assume direct model ID with anthropic as default
    return model, "anthropic"


def _parse_json_response(content: str) -> dict:
    """Extract JSON from LLM response."""
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
    return {"verdict": "INSUFFICIENT_EVIDENCE", "explanation": content, "evidence": []}


def _get_collection(chunking_strategy: str = "fixed"):
    """Get or create a ChromaDB collection for the given chunking strategy.

    Each strategy gets its own collection so different chunk sizes don't conflict.
    """
    client = get_chroma_client()
    collection_name = f"health_corpus_{chunking_strategy}" if chunking_strategy != "fixed" else "health_corpus"
    collection = get_or_create_collection(client, collection_name=collection_name)

    # Check if collection has documents; if empty, index the corpus
    if collection.count() == 0:
        _index_corpus(collection, chunking_strategy)

    return collection


def _index_corpus(collection, chunking_strategy: str):
    """Index the corpus into a ChromaDB collection using the given chunking strategy."""
    from src.shared.corpus_loader import load_corpus
    from src.chunking import chunk_corpus
    from src.shared.vector_store import add_documents

    corpus = load_corpus("data/corpus.json")
    chunks = chunk_corpus(corpus, strategy=chunking_strategy)
    add_documents(collection, chunks)
    print(f"Indexed {len(chunks)} chunks with strategy '{chunking_strategy}'")


def _retrieve(query: str, collection, retrieval_method: str, top_k: int = 5) -> list[dict]:
    """Retrieve evidence using the specified method."""
    if retrieval_method == "naive":
        return search(collection, query, top_k=top_k)

    elif retrieval_method == "hybrid":
        from src.retrieval.hybrid import retrieve_hybrid
        hits = retrieve_hybrid(query, collection, top_k=top_k)
        return hits

    elif retrieval_method == "hybrid_reranked":
        from src.retrieval.hybrid import retrieve_hybrid
        from src.retrieval.reranker import rerank
        hits = retrieve_hybrid(query, collection, top_k=top_k * 2)
        return rerank(query, hits, top_k=top_k)

    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}")


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
        raw = _run_langgraph_multi(claim, chunking_strategy, retrieval_method, model)
    elif agent_architecture == "strands_rerouting":
        raw = _run_strands_rerouting(claim, chunking_strategy, retrieval_method, model)
    else:
        raise ValueError(f"Unknown agent architecture: {agent_architecture}")

    latency = time.time() - start_time
    estimated_tokens = len(str(raw)) // 4
    estimated_cost = estimated_tokens * 9e-6  # rough average

    result = FactCheckResult(
        claim=claim,
        verdict=raw.get("verdict", "INSUFFICIENT_EVIDENCE"),
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
    model_id, provider = _resolve_model(model)
    collection = _get_collection(chunking_strategy)

    # Retrieve evidence
    hits = _retrieve(claim, collection, retrieval_method, top_k=5)

    # Format evidence passages
    passages = "\n\n".join(
        f"[{i+1}] (PMID: {h.get('metadata', {}).get('pmid', 'N/A')}) {h.get('text', '')}"
        for i, h in enumerate(hits)
    )

    # Single LLM call
    prompt = f"Claim: {claim}\n\nEvidence:\n{passages}"
    response = call_llm(prompt, system=VERDICT_SYSTEM, model=model_id, provider=provider)
    result = _parse_json_response(response["content"])

    return {
        "verdict": result.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "explanation": result.get("explanation", ""),
        "evidence": result.get("evidence", []),
    }


def _run_strands_multi(claim: str, model: str) -> dict:
    """Strands 4-agent sequential pipeline (uses Bedrock only)."""
    from src.agents.strands.orchestrator import run_pipeline
    raw = run_pipeline(claim)
    return raw["verdict"]


def _run_langgraph_multi(
    claim: str, chunking_strategy: str, retrieval_method: str, model: str,
) -> dict:
    """LangGraph graph-based multi-agent pipeline."""
    from src.agents.langgraph.graph import run_pipeline

    model_id, provider = _resolve_model(model)
    result = run_pipeline(claim, model=model_id, provider=provider)

    verdict = result.get("verdict", {})
    return {
        "verdict": verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "explanation": verdict.get("explanation", ""),
        "evidence": verdict.get("evidence", []),
    }


def _run_strands_rerouting(
    claim: str, chunking_strategy: str, retrieval_method: str, model: str,
) -> dict:
    """LangGraph multi-agent with rerouting (adaptive loop)."""
    from src.agents.langgraph.graph_rerouting import run_pipeline_rerouting

    model_id, provider = _resolve_model(model)
    result = run_pipeline_rerouting(claim, model=model_id, provider=provider)

    verdict = result.get("verdict", {})
    return {
        "verdict": verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "explanation": verdict.get("explanation", ""),
        "evidence": verdict.get("evidence", []),
    }
