"""Configurable pipeline — single entry point for all experiment configurations.

Dispatches to the appropriate chunking strategy, retrieval method, agent
architecture, and model based on the experiment config.
"""

import json
import re
import time

from src.shared.chunking_utils import clear_chunk_artifacts, export_chunk_artifacts
from src.shared.schema import VALID_VERDICTS

# Supported values for each axis
CHUNKING_STRATEGIES = ("fixed", "semantic", "section_aware", "recursive")
RETRIEVAL_METHODS = ("naive", "hybrid", "hybrid_reranked")
AGENT_ARCHITECTURES = ("single_pass", "strands_multi", "langgraph_multi", "strands_rerouting", "strands_rerouting_ext", "strands_rerouting_ext_v2")
MODELS = (
    "claude-sonnet-4",
    "gpt-4o-mini",
    "claude-haiku",
    "llama-3.1-8b",
    "llama-3.1-8b-ft",
    "llama-3.1-70b",
)

DEFAULT_CHUNKING_STRATEGY = "recursive"
DEFAULT_RETRIEVAL_METHOD = "naive"
DEFAULT_AGENT_COLLECTION = "health_corpus_recursive"

# ── Model resolution ──────────────────────────────────────────────────────────

_MODEL_ID_MAP = {
    "claude-sonnet-4":  "claude-sonnet-4-20250514",
    "claude-haiku":     "claude-haiku-4-5-20251001",
    "llama-3.1-8b":    "llama3.1:8b",
    "llama-3.1-8b-ft": "llama3.1:8b",  # fine-tuned adapter loaded separately
    "llama-3.1-70b":   "llama3.1:70b",
}


def _resolve_model_id(model: str) -> str:
    """Map experiment-level model aliases to actual provider model IDs."""
    return _MODEL_ID_MAP.get(model, model)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scientific claim fact-checker. Given evidence passages from scientific abstracts, determine whether each claim is supported, contradicted, or lacks sufficient evidence.

Verification Logic:
1. Read the claim carefully. Identify the core scientific assertion.
2. Examine each evidence passage. Does it directly address the claim's topic?
3. If evidence directly supports the claim's assertion, verdict is SUPPORTED.
4. If evidence directly contradicts the claim's assertion, verdict is UNSUPPORTED.
5. If the evidence does not address the claim, or is only tangentially related, verdict is INSUFFICIENT_EVIDENCE.

Important: Base your verdict ONLY on the provided evidence. If the evidence partially relates to the topic but does not confirm or deny the specific claim, use INSUFFICIENT_EVIDENCE.

Respond ONLY with valid JSON:
{
    "verdict": "SUPPORTED | UNSUPPORTED | INSUFFICIENT_EVIDENCE",
    "explanation": "2-3 sentences justifying your verdict based on the evidence.",
    "evidence": [
        {"source": "Doc ID", "passage": "key passage text", "relevance_score": 0.0-1.0}
    ]
}"""


# ── ChromaDB collection helpers ───────────────────────────────────────────────

def get_collection(
    chunking_strategy: str,
    force_rebuild: bool = False,
    collection_name: str | None = None,
):
    """Get or create and populate a ChromaDB collection for the given chunking strategy."""
    from src.shared.vector_store import add_documents, get_chroma_client, get_or_create_collection, reset_collection
    from src.shared.corpus_loader import load_corpus
    from src.chunking import chunk_corpus

    collection_name = collection_name or f"health_corpus_{chunking_strategy}"
    client = get_chroma_client()

    if force_rebuild:
        reset_collection(client, collection_name)
        clear_chunk_artifacts(chunking_strategy)

    collection = get_or_create_collection(client, collection_name=collection_name)

    # Skip chunking if ChromaDB already has data for this strategy
    if not force_rebuild and collection.count() > 0:
        return collection

    corpus = load_corpus()
    chunks = chunk_corpus(corpus, strategy=chunking_strategy)
    export_chunk_artifacts(
        strategy=chunking_strategy,
        chunks=chunks,
        corpus_size=len(corpus),
        parameters={"chunking_strategy": chunking_strategy},
    )
    add_documents(collection, chunks)
    return collection


def get_agent_collection(force_rebuild: bool = False):
    """Return the default indexed local corpus used by the multi-agent flows."""
    return get_collection(
        DEFAULT_CHUNKING_STRATEGY,
        force_rebuild=force_rebuild,
        collection_name=DEFAULT_AGENT_COLLECTION,
    )


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _sanitize_json(text: str) -> str:
    """Fix common invalid JSON escapes produced by LLMs (e.g. \\%)."""
    return re.sub(r'\\(?=[^"\\bfnrtu/])', r'\\\\', text)


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced top-level JSON object substring, if present."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _repair_json_like(text: str) -> str:
    """Repair common model output issues that violate strict JSON syntax."""
    # Remove parenthetical commentary after primitive values, e.g. false (note...)
    text = re.sub(
        r'(:\s*(?:true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))'  r'\s*\([^\n\r{}\[\]]*\)',
        r"\1",
        text,
    )
    # Quote bareword values that appear before a comma or closing brace.
    return re.sub(
        r'(:\s*)([A-Za-z][A-Za-z0-9_\- ]+?)(\s*(?:,|\}))',
        lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}',
        text,
    )


def _fallback_from_verdict(content: str) -> dict | None:
    """Fallback parser: recover verdict/explanation from non-JSON output."""
    verdict_match = re.search(
        r'"?verdict"?\s*[:=]\s*"?(SUPPORTED|UNSUPPORTED|INSUFFICIENT_EVIDENCE)"?',
        content,
        re.IGNORECASE,
    )
    if not verdict_match:
        return None

    verdict = verdict_match.group(1).upper()
    explanation = ""
    explanation_match = re.search(r'"?explanation"?\s*[:=]\s*"(.*?)"', content, re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    return {"verdict": verdict, "explanation": explanation, "evidence": []}


def _normalize_parsed_response(result: dict) -> dict:
    """Normalize LLM JSON into the project schema."""
    verdict = str(result.get("verdict", "")).strip().upper()
    verdict_aliases = {
        "REFUTED": "UNSUPPORTED",
        "CONTRADICTED": "UNSUPPORTED",
        "NOT_ENOUGH_INFO": "INSUFFICIENT_EVIDENCE",
        "NEI": "INSUFFICIENT_EVIDENCE",
    }
    # Keep the shared label space unchanged so historical experiment outputs remain comparable.
    verdict = verdict_aliases.get(verdict, verdict)
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Unsupported verdict {verdict!r}. Expected one of {sorted(VALID_VERDICTS)}.")

    explanation = str(result.get("explanation", "") or "").strip()
    evidence_items = []

    for item in result.get("evidence", []) or []:
        if not isinstance(item, dict):
            continue
        evidence_items.append(
            {
                "source": str(item.get("source", "N/A") or "N/A"),
                "passage": str(item.get("passage", item.get("text", "")) or ""),
                "relevance_score": float(item.get("relevance_score", item.get("score", 0.0)) or 0.0),
            }
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "evidence": evidence_items,
    }


def parse_json_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    candidates = [content, _sanitize_json(content)]

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        raw = match.group(1)
        candidates.extend([raw, _sanitize_json(raw)])

    extracted = _extract_first_json_object(content)
    if extracted:
        candidates.extend([extracted, _sanitize_json(extracted), _repair_json_like(extracted), _sanitize_json(_repair_json_like(extracted))])

    seen = set()
    deduped = []
    for text in candidates:
        if text and text not in seen:
            deduped.append(text)
            seen.add(text)

    for text in deduped:
        try:
            return _normalize_parsed_response(json.loads(text))
        except json.JSONDecodeError:
            pass

    fallback = _fallback_from_verdict(content)
    if fallback:
        return _normalize_parsed_response(fallback)

    raise ValueError(f"(Failed to parse JSON response) {content}")


# ── Main experiment entry point ───────────────────────────────────────────────

def run_experiment(
    claim: str,
    chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY,
    retrieval_method: str = DEFAULT_RETRIEVAL_METHOD,
    agent_architecture: str = "single_pass",
    model: str = "claude-sonnet-4",
    force_rebuild_chunks: bool = False,
) -> dict:
    """Run a single claim through a configured pipeline.

    Args:
        claim: The health claim to fact-check.
        chunking_strategy: One of 'fixed', 'semantic', 'section_aware', 'recursive'.
        retrieval_method: One of 'naive', 'hybrid', 'hybrid_reranked'.
        agent_architecture: One of 'single_pass', 'strands_multi', 'langgraph_multi', 'strands_rerouting'.
        model: Model identifier string (see MODELS tuple for valid values).

    Returns:
        Dict matching FactCheckResult schema with experiment config in metadata.
    """
    from src.shared.schema import FactCheckResult

    start_time = time.time()

    # Dispatch based on architecture
    if agent_architecture == "single_pass":
        raw = _run_single_pass(claim, chunking_strategy, retrieval_method, model, force_rebuild_chunks=force_rebuild_chunks)
    elif agent_architecture == "strands_multi":
        raw = _run_strands_multi(claim, model)
    elif agent_architecture == "langgraph_multi":
        raw = _run_langgraph_multi(claim, model)
    elif agent_architecture == "strands_rerouting":
        raw = _run_strands_rerouting(claim, model)
    elif agent_architecture == "strands_rerouting_ext":
        raw = _run_strands_rerouting_ext(claim, model)
    elif agent_architecture == "strands_rerouting_ext_v2":
        raw = _run_strands_rerouting_ext_v2(claim, model)
    else:
        raise ValueError(f"Unknown agent architecture: {agent_architecture}")

    latency = time.time() - start_time
    usage = raw.pop("_usage", {})
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    estimated_tokens = input_tokens + output_tokens or len(str(raw)) // 4
    estimated_cost = usage.get("estimated_cost_usd", estimated_tokens * 9e-6)

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
            "estimated_cost_usd": round(float(estimated_cost), 6),
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
        "force_rebuild_chunks": force_rebuild_chunks,
    }
    return output


# ── Architecture dispatchers ──────────────────────────────────────────────────

def _run_single_pass(
    claim: str,
    chunking_strategy: str,
    retrieval_method: str,
    model: str,
    force_rebuild_chunks: bool = False,
) -> dict:
    """Single-pass: retrieve evidence locally, then make one verdict LLM call."""
    model_id = _resolve_model_id(model)
    collection = get_collection(chunking_strategy, force_rebuild=force_rebuild_chunks)

    if retrieval_method == "naive":
        from src.retrieval.naive import retrieve

        hits = retrieve(claim, collection=collection, top_k=5)
    else:
        from src.retrieval.hybrid import retrieve_hybrid

        hits = retrieve_hybrid(claim, collection, top_k=10)

        if retrieval_method == "hybrid_reranked":
            from src.retrieval.reranker import rerank

            hits = rerank(claim, hits, top_k=5)
        else:
            hits = hits[:5]

    passages = "\n\n".join(
        f"[{i+1}] (Doc ID: {h['metadata'].get('doc_id', 'N/A')}) {h['text']}"
        for i, h in enumerate(hits)
    )

    from src.shared.llm import call_llm
    prompt = f"Claim: {claim}\n\nEvidence:\n{passages}"
    response = call_llm(prompt, system=SYSTEM_PROMPT, model=model_id)

    result_data = parse_json_response(response["content"])
    return {
        "verdict": result_data["verdict"],
        "explanation": result_data.get("explanation", ""),
        "evidence": result_data.get("evidence", []),
        "_usage": {
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
        },
    }


def _run_strands_multi(claim: str, model: str) -> dict:
    """Run the current Strands multi-agent pipeline against the local corpus."""
    from src.agents.strands.orchestrator import run_pipeline

    # The current Strands agent stack resolves its own Bedrock model internally.
    get_agent_collection()
    raw = run_pipeline(claim)
    return raw["verdict"]


def _run_langgraph_multi(claim: str, model: str) -> dict:
    """LangGraph graph-based multi-agent pipeline."""
    from src.agents.langgraph.graph import build_graph

    model_id = _resolve_model_id(model)
    graph = build_graph()

    # '_model' is a private state key read by nodes via _resolve_model()
    result = graph.invoke({"claim": claim, "_model": model_id})

    verdict_dict = result.get("verdict", {})
    return {
        "verdict": verdict_dict.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "explanation": verdict_dict.get("explanation", ""),
        "evidence": verdict_dict.get("evidence", []),
    }


def _run_strands_rerouting(claim: str, model: str) -> dict:
    """Strands multi-agent with adaptive rerouting (evidence gap loop)."""
    from src.agents.strands.orchestrator_rerouting import run_pipeline_rerouting

    get_agent_collection()
    raw = run_pipeline_rerouting(claim)
    verdict = raw["verdict"]

    return {
        "verdict": verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "explanation": verdict.get("explanation", ""),
        "evidence": verdict.get("evidence", []),
        "rerouting_loops": raw.get("rerouting_loops", 1),
    }


def _run_strands_rerouting_ext(claim: str, model: str) -> dict:
    """Strands rerouting with Semantic Scholar external search fallback."""
    from src.agents.strands.orchestrator_rerouting_ext import run_pipeline_rerouting_ext

    get_agent_collection()
    raw = run_pipeline_rerouting_ext(claim)
    verdict = raw["verdict"]

    return {
        "verdict": verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "explanation": verdict.get("explanation", ""),
        "evidence": verdict.get("evidence", []),
        "rerouting_loops": raw.get("rerouting_loops", 1),
        "external_search_used": raw.get("external_search_used", False),
        "external_papers_added": raw.get("external_papers_added", 0),
    }


def _run_strands_rerouting_ext_v2(claim: str, model: str) -> dict:
    """Revamped Strands rerouting: original-claim-first + smart gating + S2 fallback."""
    from src.agents.strands.orchestrator_rerouting_ext_v2 import run_pipeline_rerouting_ext_v2

    get_agent_collection()
    raw = run_pipeline_rerouting_ext_v2(claim)
    verdict = raw["verdict"]

    return {
        "verdict": verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "explanation": verdict.get("explanation", ""),
        "evidence": verdict.get("evidence", []),
        "rerouting_loops": raw.get("rerouting_loops", 1),
        "external_search_used": raw.get("external_search_used", False),
        "external_papers_added": raw.get("external_papers_added", 0),
        "path": raw.get("path", "unknown"),
    }
