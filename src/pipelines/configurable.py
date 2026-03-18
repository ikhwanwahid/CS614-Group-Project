"""Configurable pipeline — single entry point for all experiment configurations.

Dispatches to the appropriate chunking strategy, retrieval method, agent
architecture, and model based on the experiment config.
"""

import json
import re
import time

from src.shared.chunking_utils import chunk_artifacts_exist, clear_chunk_artifacts, export_chunk_artifacts

# Supported values for each axis
CHUNKING_STRATEGIES = ("fixed", "semantic", "section_aware", "recursive")
RETRIEVAL_METHODS = ("naive", "hybrid", "hybrid_reranked")
AGENT_ARCHITECTURES = ("single_pass", "strands_multi", "langgraph_multi", "strands_rerouting")
MODELS = (
    "claude-sonnet-4",
    "claude-sonnet-4-20250514",
    "gpt-4o-mini",
    "claude-haiku",
    "llama-3.1-8b",
    "llama-3.1-8b-ft",
    "llama-3.1-70b",
)

# SYSTEM_PROMPT = """You are a health claim fact-checker. Given the following evidence passages and a health claim, provide:
# 1. A verdict: SUPPORTED, UNSUPPORTED, OVERSTATED, or INSUFFICIENT_EVIDENCE
# 2. An explanation justifying your verdict (2-3 sentences)
# 3. Which evidence passages you relied on

# Respond ONLY with valid JSON matching this schema:
# {
#     "verdict": "SUPPORTED | UNSUPPORTED | OVERSTATED | INSUFFICIENT_EVIDENCE",
#     "explanation": "Your explanation here",
#     "evidence": [
#         {"source": "PMID or author reference", "passage": "key passage text", "relevance_score": 0.0-1.0}
#     ]
# }"""
# SYSTEM_PROMPT = """You are a rigorous health claim fact-checker. 
# Focus strictly on the 'Disease' and the 'Population' mentioned.

# For each claim, follow this verification logic:
# 1. Extract Claim Entities: [Disease/Condition] and [Population/Group].
# 2. Scan Evidence: Does the evidence explicitly name the SAME Disease and SAME Population?
# 3. Identity Gap: If the evidence uses broader terms (e.g., 'vaccination' vs 'flu vaccine' or 'adults' vs 'elderly'), you must flag this as a 'MISMATCH' or 'TOO GENERAL'.

# Respond ONLY with valid JSON:
# {
#     "entity_verification": {
#         "claim": {"disease": "...", "population": "..."},
#         "evidence_match": {
#             "disease_matched": true/false,
#             "population_matched": true/false,
#             "notes": "Explain if the evidence is talking about a different or more general group/disease."
#         }
#     },
#     "verdict": "SUPPORTED | UNSUPPORTED | OVERSTATED | INSUFFICIENT_EVIDENCE",
#     "explanation": "If entities do not match exactly, explain that the evidence is not specific enough to support the claim.",
#     "evidence": [
#         {"source": "PMID/Author", "passage": "text", "relevance_score": 0.0-1.0}
#     ]
# }"""


# SYSTEM_PROMPT = """You are a rigorous health claim fact-checker. 
# Focus strictly on the 'Disease' and the 'Population' mentioned.

# Verification & Weighting Logic:
# 1. Extract Claim Entities: [Disease] and [Population].
# 2. Prioritize Specificity: Evidence matching BOTH entities explicitly (e.g., 'Flu vaccine' AND 'elderly') is HIGH-PRIORITY.
# 3. Penalty for Ambiguity: If evidence uses general terms (e.g., 'vaccination' instead of 'flu vaccine'), you MUST downgrade its importance. It cannot be used as the sole basis for SUPPORTED or UNSUPPORTED.
# 4. Final Verdict: If the high-priority evidence is missing, the verdict should likely be INSUFFICIENT_EVIDENCE.

# Respond ONLY with valid JSON:
# {
#     "entity_verification": {
#          "claim": {"disease": "...", "population": "..."},
#          "evidence_match": {
#              "disease_matched": true/false,
#              "population_matched": true/false,
#              "notes": "Explain if the evidence is talking about a different or more general group/disease."
#          }
#      },
#      "verdict": "SUPPORTED | UNSUPPORTED | OVERSTATED | INSUFFICIENT_EVIDENCE",
#      "explanation": "If entities do not match exactly, explain that the evidence is not specific enough to support the claim.",
#      "evidence": [
#          {"source": "PMID/Author", "passage": "text", "relevance_score": 0.0-1.0}
#      ]
# }"""
SYSTEM_PROMPT = """You are a scientific claim fact-checker. Given evidence passages from research abstracts, determine whether each claim is supported, contradicted, or lacks sufficient evidence.

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
        {"source": "PMID/Author", "passage": "key passage text", "relevance_score": 0.0-1.0}
    ]
}"""

def get_collection(chunking_strategy: str, force_rebuild: bool = False):
    """Get or create and populate a ChromaDB collection for the given chunking strategy."""
    from src.shared.vector_store import add_documents, get_chroma_client, get_or_create_collection, reset_collection
    from src.shared.corpus_loader import load_corpus
    from src.chunking import chunk_corpus

    collection_name = f"health_corpus_{chunking_strategy}"
    client = get_chroma_client()
    artifacts_ready = chunk_artifacts_exist(chunking_strategy)

    print(f"[chunking] Preparing collection '{collection_name}' with strategy='{chunking_strategy}'")

    if force_rebuild:
        print(f"[chunking] Force rebuild requested for strategy='{chunking_strategy}'")
        reset_collection(client, collection_name)
        clear_chunk_artifacts(chunking_strategy)

    collection = get_or_create_collection(client, collection_name=collection_name)
    if not force_rebuild and collection.count() > 0 and artifacts_ready:
        print(
            f"[chunking] Reusing cached collection '{collection_name}' "
            f"({collection.count()} chunks already indexed)"
        )
        return collection

    if collection.count() > 0:
        print(f"[chunking] Resetting existing collection '{collection_name}' before rebuild")
        reset_collection(client, collection_name)
        collection = get_or_create_collection(client, collection_name=collection_name)

    corpus = load_corpus()
    print(f"[chunking] Running strategy='{chunking_strategy}' on {len(corpus)} corpus articles")
    chunks = chunk_corpus(corpus, strategy=chunking_strategy)
    print(f"[chunking] Generated {len(chunks)} chunks for strategy='{chunking_strategy}'")
    export_chunk_artifacts(
        strategy=chunking_strategy,
        chunks=chunks,
        corpus_size=len(corpus),
        parameters={"chunking_strategy": chunking_strategy},
    )
    add_documents(collection, chunks)
    print(f"[chunking] Indexed {len(chunks)} chunks into collection '{collection_name}'")
    return collection


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
    fixed = text
    # Remove parenthetical commentary after primitive values, e.g. false (note...)
    fixed = re.sub(
        r'(:\s*(?:true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))\s*\([^\n\r{}\[\]]*\)',
        r"\1",
        fixed,
    )
    # Quote bareword values that appear before a comma or closing brace.
    fixed = re.sub(
        r'(:\s*)([A-Za-z][A-Za-z0-9_\- ]+?)(\s*(?:,|\}))',
        lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}',
        fixed,
    )
    return fixed


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


def run_experiment(
    claim: str,
    chunking_strategy: str = "fixed",
    retrieval_method: str = "naive",
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
        model: Model identifier string.

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


def _run_single_pass(
    claim: str,
    chunking_strategy: str,
    retrieval_method: str,
    model: str,
    force_rebuild_chunks: bool = False,
) -> dict:
    """Single-pass: retrieve evidence + one LLM call for verdict."""
    if retrieval_method == "naive":
        from src.shared.vector_store import search
        print(f"[retrieval] Running naive retrieval with chunking='{chunking_strategy}'")
        collection = get_collection(chunking_strategy, force_rebuild=force_rebuild_chunks)
        hits = search(collection, claim, top_k=5)
        print(f"[retrieval] Naive retrieval returned {len(hits)} hits")
    else:
        print(f"[retrieval] Running {retrieval_method} retrieval with chunking='{chunking_strategy}'")
        collection = get_collection(chunking_strategy, force_rebuild=force_rebuild_chunks)

        from src.retrieval.hybrid import retrieve_hybrid
        hits = retrieve_hybrid(claim, collection, top_k=10)
        print(f"[retrieval] Hybrid retrieval returned {len(hits)} candidate hits")

        if retrieval_method == "hybrid_reranked":
            from src.retrieval.reranker import rerank
            hits = rerank(claim, hits, top_k=5)
            print(f"[retrieval] Reranker kept {len(hits)} hits")
        else:
            hits = hits[:5]
            print(f"[retrieval] Using top {len(hits)} hybrid hits without reranking")

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
        "_usage": {
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
        },
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
