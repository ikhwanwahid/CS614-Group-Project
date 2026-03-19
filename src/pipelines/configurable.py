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

<<<<<<< HEAD
# ── Model resolution ──────────────────────────────────────────────────────────

_MODEL_ID_MAP = {
    "claude-sonnet-4":    "claude-sonnet-4-20250514",
    "claude-haiku":       "claude-haiku-4-5-20251001",
    "gpt-4o-mini":        "gpt-4o-mini",
    "llama-3.1-8b":       "llama3.1:8b",
    "llama-3.1-8b-ft":    "llama3.1:8b",   # fine-tuned adapter loaded separately
    "llama-3.1-70b":      "llama3.1:70b",
}


def _resolve_model_id(model: str) -> str:
    """Map experiment-level model aliases to actual provider model IDs."""
    return _MODEL_ID_MAP.get(model, model)


# ── System prompt ─────────────────────────────────────────────────────────────

=======
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
>>>>>>> d8747033885e48da397e67fd4560abac72d8bc80
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

<<<<<<< HEAD

# ── ChromaDB collection helper ────────────────────────────────────────────────

=======
>>>>>>> d8747033885e48da397e67fd4560abac72d8bc80
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


# ── JSON parsing helpers (unchanged) ─────────────────────────────────────────

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
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    fallback = _fallback_from_verdict(content)
    if fallback:
        return fallback

    raise ValueError(f"(Failed to parse JSON response) {content}")


# ── Main experiment entry point ───────────────────────────────────────────────

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
        model: Model identifier string (see MODELS tuple for valid values).

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


# ── Architecture dispatchers ──────────────────────────────────────────────────

def _run_single_pass(claim: str, chunking_strategy: str, retrieval_method: str, model: str) -> dict:
    """Single-pass: retrieve evidence + one LLM call for verdict.

    For naive retrieval, delegates to p1_naive_single which already handles
    model routing via call_llm.  For hybrid/reranked, builds the evidence
    block here and calls call_llm directly with the resolved model ID.
    """
    model_id = _resolve_model_id(model)

    if retrieval_method == "naive":
        from src.pipelines.p1_naive_single.pipeline import run as run_p1
        result = run_p1(claim, model=model_id)
        return {
            "verdict": result["verdict"],
            "explanation": result["explanation"],
            "evidence": result["evidence"],
        }

    # ── Hybrid / reranked retrieval ───────────────────────────────────────────
    collection = get_collection(chunking_strategy)

    from src.retrieval.hybrid import retrieve_hybrid
    hits = retrieve_hybrid(claim, collection, top_k=10)

    if retrieval_method == "hybrid_reranked":
        from src.retrieval.reranker import rerank
        hits = rerank(claim, hits, top_k=5)
    else:
        hits = hits[:5]

    passages = "\n\n".join(
        f"[{i+1}] (PMID: {h['metadata'].get('pmid', 'N/A')}) {h['text']}"
        for i, h in enumerate(hits)
    )

    from src.shared.llm import call_llm
    prompt = f"Claim: {claim}\n\nEvidence:\n{passages}"
    response = call_llm(prompt, system=SYSTEM_PROMPT, model=model_id)

    result_data = parse_json_response(response["content"])
    if "verdict" not in result_data:
        raise KeyError(f"LLM response missing 'verdict' key: {result_data}")
    return {
        "verdict": result_data["verdict"],
        "explanation": result_data.get("explanation", ""),
        "evidence": result_data.get("evidence", []),
    }


def _run_strands_multi(claim: str, model: str) -> dict:
    """Strands 4-agent sequential pipeline (uses Bedrock internally)."""
    from src.agents.strands.orchestrator import run_pipeline
    raw = run_pipeline(claim)
    return raw["verdict"]


def _run_langgraph_multi(claim: str, model: str) -> dict:
    """LangGraph graph-based multi-agent pipeline.

    Accepts any model supported by call_llm (Anthropic, OpenAI, Ollama).
    The model alias is resolved to a provider model ID and injected into
    graph state via the special '_model' key, which each node reads.
    """
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

    raw = run_pipeline_rerouting(claim)
    verdict = raw["verdict"]

    return {
        "verdict": verdict.get("verdict", "INSUFFICIENT_EVIDENCE"),
        "explanation": verdict.get("explanation", ""),
        "evidence": verdict.get("evidence", []),
        # Pass rerouting metadata up so it lands in the experiment output
        "rerouting_loops": raw.get("rerouting_loops", 1),
    }
