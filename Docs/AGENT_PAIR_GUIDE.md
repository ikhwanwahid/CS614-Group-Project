# Agent Pair Guide — Members 4 & 5 (Agent Architectures + Models)

You own the **agent architectures** and **model integration** — everything about how the LLM reasons over retrieved evidence. Your work covers experiments E5-E7 (agent architectures), E8-E12 (model comparisons), and the rerouting/adaptive logic.

---

## Your Deliverables

| # | Task | Files | Priority | Blocks |
|---|------|-------|----------|--------|
| 1 | Implement LangGraph multi-agent pipeline | `src/agents/langgraph/` | **Critical** | E6 |
| 2 | Implement rerouting/adaptive architecture | `src/agents/strands/orchestrator_rerouting.py` | **Critical** | E7 |
| 3 | Wire GPT-4o-mini into configurable pipeline | `src/pipelines/configurable.py` | **Critical** | E8, E11, E12 |
| 4 | Wire Llama 3.1 (Ollama) into configurable pipeline | `src/pipelines/configurable.py` | **Critical** | E9 |
| 5 | Fine-tune Llama 3.1 8B on health claims (stretch) | `src/finetuning/` | Stretch | E10 |
| 6 | Wire all architectures into configurable pipeline | `src/pipelines/configurable.py` | **Critical** | All agent experiments |

---

## Architecture Overview

```
evidence → [Agent Architecture] → verdict + explanation
              ↑
         uses [Model] for LLM calls
```

You control the right side of the pipeline. The RAG pair controls what evidence reaches you.

### What already exists

**Working:**
- `src/agents/strands/orchestrator.py` — 4-agent sequential pipeline (Claim Parser → Retrieval → Reviewer → Verdict)
- `src/agents/strands/claim_parser.py` — Agent 1
- `src/agents/strands/retrieval_agent.py` — Agent 2
- `src/agents/strands/evidence_reviewer.py` — Agent 3
- `src/agents/strands/verdict_agent.py` — Agent 4
- `src/shared/llm.py` — Multi-provider LLM client (Anthropic, OpenAI, Ollama already wired)

**Stubs (you implement):**
- `src/agents/langgraph/` — empty directory
- `_run_langgraph_multi()` in `configurable.py` — raises `NotImplementedError`
- `_run_strands_rerouting()` in `configurable.py` — raises `NotImplementedError`

---

## Task 1: LangGraph Multi-Agent Pipeline

**Directory:** `src/agents/langgraph/`

**What it does:** Same 4-agent decomposition as Strands (Claim Parser → Retrieval → Reviewer → Verdict), but implemented using LangGraph's graph-based orchestration with nodes, edges, and shared state.

### Why LangGraph?

The research question is: **does the orchestration framework matter?** Strands uses event-driven sequential execution; LangGraph uses a directed graph with explicit state passing. Same agents, different wiring — any quality difference tells us about the framework, not the agents.

### Approach

```python
# src/agents/langgraph/graph.py

from langgraph.graph import StateGraph, END
from typing import TypedDict


class PipelineState(TypedDict):
    claim: str
    sub_claims: list[dict]
    evidence: list[dict]
    review: dict
    verdict: dict


def build_graph():
    """Build the 4-node LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("parse_claim", parse_claim_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("review_evidence", review_evidence_node)
    graph.add_node("generate_verdict", generate_verdict_node)

    graph.set_entry_point("parse_claim")
    graph.add_edge("parse_claim", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "review_evidence")
    graph.add_edge("review_evidence", "generate_verdict")
    graph.add_edge("generate_verdict", END)

    return graph.compile()


def parse_claim_node(state: PipelineState) -> dict:
    """Node 1: Decompose claim into sub-claims."""
    # Use the same LLM call as Strands, but adapted for LangGraph state
    from src.shared.llm import call_llm
    # ... parse claim, return {"sub_claims": [...]}


def retrieve_evidence_node(state: PipelineState) -> dict:
    """Node 2: Retrieve evidence for each sub-claim."""
    # Use the shared vector store + retrieval method
    # ... return {"evidence": [...]}


def review_evidence_node(state: PipelineState) -> dict:
    """Node 3: Review evidence quality."""
    # ... return {"review": {...}}


def generate_verdict_node(state: PipelineState) -> dict:
    """Node 4: Generate final verdict."""
    # ... return {"verdict": {...}}
```

### Key design decisions

1. **Reuse LLM prompts from Strands agents** — the prompts in `claim_parser.py`, `evidence_reviewer.py`, etc. should be the same. Only the orchestration differs.
2. **State passing:** LangGraph passes a shared state dict between nodes. Each node reads what it needs and adds its output.
3. **Use `call_llm()` from `src/shared/llm.py`** — this lets you swap models easily via the `model` and `provider` parameters.

### Dependencies

```bash
uv add langgraph langchain-core
```

### Files to create

```
src/agents/langgraph/
├── __init__.py
├── graph.py           # Graph definition + compile
├── nodes.py           # Node functions (parse, retrieve, review, verdict)
└── state.py           # PipelineState TypedDict
```

---

## Task 2: Rerouting / Adaptive Architecture

**File:** `src/agents/strands/orchestrator_rerouting.py`

**What it does:** After the Evidence Reviewer flags gaps, loop back to the Retrieval Agent to fill them — instead of proceeding with incomplete evidence.

### Flow

```
claim → [Claim Parser] → [Retrieval Agent] → [Evidence Reviewer]
                              ↑                     |
                              |                     v
                              +--- INSUFFICIENT ----+
                              |
                              v (SUFFICIENT)
                        [Verdict Agent] → output
```

### Approach

```python
"""Strands orchestrator with rerouting — adaptive evidence loop."""

from src.agents.strands.claim_parser import parse_claim
from src.agents.strands.retrieval_agent import retrieve_evidence
from src.agents.strands.evidence_reviewer import review_evidence
from src.agents.strands.verdict_agent import generate_verdict

MAX_RETRIEVAL_LOOPS = 3  # Prevent infinite loops


def run_pipeline_rerouting(claim: str) -> dict:
    """Run the multi-agent pipeline with adaptive rerouting."""

    # Step 1: Parse claim
    parsed = parse_claim(claim)
    sub_claims = [{"sub_claim": sc.sub_claim, "query": sc.query} for sc in parsed.sub_claims]

    for loop in range(MAX_RETRIEVAL_LOOPS):
        # Step 2: Retrieve evidence
        retrieval_output = retrieve_evidence(sub_claims)

        # Step 3: Review evidence
        review = review_evidence(claim, format_evidence(retrieval_output))

        # Step 4: Check if evidence is sufficient
        if review.evidence_strength in ("strong", "moderate") or loop == MAX_RETRIEVAL_LOOPS - 1:
            break

        # Reroute: modify queries based on reviewer feedback
        sub_claims = _refine_queries(sub_claims, review)

    # Step 5: Generate verdict
    verdict = generate_verdict(claim, format_review(review, retrieval_output))

    return {
        "parsed_claims": parsed.model_dump(),
        "retrieval": retrieval_output.model_dump(),
        "review": review.model_dump(),
        "verdict": verdict.model_dump(),
        "rerouting_loops": loop + 1,
    }


def _refine_queries(sub_claims, review):
    """Refine search queries based on reviewer feedback."""
    # Use the reviewer's flags to identify which sub-claims need better evidence
    # Modify queries to be more specific or try different search terms
    ...
```

### Key decisions
- **When to reroute:** When `evidence_strength` is "weak" or "insufficient"
- **How to refine queries:** Use the reviewer's flags to identify gaps, then ask the LLM to generate better search queries
- **Max loops:** 3 is reasonable. More than that and you're spending too much time on retrieval

---

## Task 3: Wire GPT-4o-mini

**File:** `src/pipelines/configurable.py` (modify) + `src/shared/llm.py` (already done)

The LLM client (`src/shared/llm.py`) already supports OpenAI. You need to make the pipeline dispatch to the right provider based on the model parameter.

### What to change

In `configurable.py`, the `_run_single_pass()` function needs to pass the model through to `call_llm()`:

```python
def _run_single_pass(claim, retrieval_method, model):
    # Determine provider from model name
    provider = _get_provider(model)

    if retrieval_method == "naive":
        # Instead of calling run_p1() which hardcodes Anthropic,
        # call the retrieval + LLM directly with the specified model
        evidence = _retrieve_naive(claim)
        response = call_llm(
            prompt=_build_verdict_prompt(claim, evidence),
            model=_resolve_model_id(model),
            provider=provider,
        )
        return _parse_verdict_response(response["content"], evidence)


def _get_provider(model: str) -> str:
    """Map model name to provider."""
    if model.startswith("claude") or model.startswith("anthropic"):
        return "anthropic"
    elif model.startswith("gpt"):
        return "openai"
    elif model.startswith("llama") or model.startswith("mistral"):
        return "ollama"
    return "anthropic"


def _resolve_model_id(model: str) -> str:
    """Map experiment model names to actual model IDs."""
    mapping = {
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "gpt-4o-mini": "gpt-4o-mini",
        "llama-3.1-8b": "llama3.1:8b",
        "llama-3.1-8b-ft": "llama3.1:8b",  # Will use fine-tuned adapter
    }
    return mapping.get(model, model)
```

### For multi-agent + GPT-4o-mini (E11)

The Strands agents currently use AWS Bedrock (Claude). To run them with GPT-4o-mini, you'll need to either:

**Option A:** Modify the Strands agent prompts to use `call_llm()` with OpenAI provider instead of Bedrock
**Option B:** Create parallel LangGraph implementations that use `call_llm()` (more flexible)

Option B is cleaner — the LangGraph pipeline (Task 1) should accept a `model` parameter from the start.

---

## Task 4: Wire Llama 3.1 (Ollama)

### Setup Ollama locally

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model (~4.7GB)
ollama pull llama3.1:8b

# Verify it's running
curl http://localhost:11434/api/tags
```

The `src/shared/llm.py` already treats Ollama as an OpenAI-compatible endpoint at `localhost:11434/v1`. So once Ollama is running, the code should work with `provider="ollama"`.

### Test it

```python
from src.shared.llm import call_llm

result = call_llm(
    "Is vitamin D effective against COVID?",
    provider="ollama",
    model="llama3.1:8b",
)
print(result["content"])
```

### Expected quality

Llama 3.1 8B will produce lower-quality explanations than Claude Sonnet. That's expected — the experiment measures *how much* lower. Don't tune prompts to compensate; use the same prompts as Claude to make the comparison fair.

---

## Task 5: Fine-Tune Llama 3.1 8B (Stretch Goal)

**Directory:** `src/finetuning/`

If time permits, fine-tune Llama 3.1 8B on health claim verification to see if domain-specific training closes the gap with Claude.

### Approach

1. **Training data:** Use PUBHEALTH claims + verdicts as (input, output) pairs
2. **Method:** LoRA (Low-Rank Adaptation) — fine-tunes only a small number of parameters
3. **Tools:** `unsloth` or `peft` + `transformers`
4. **Format:** Instruction-tuning format:
   ```
   [INST] Fact-check this health claim: "Vaccines cause autism"
   Evidence: <retrieved passages>
   Respond with verdict and explanation. [/INST]
   VERDICT: UNSUPPORTED
   EXPLANATION: Multiple large-scale studies...
   ```

### Files to create

```
src/finetuning/
├── prepare_data.py    # Format PUBHEALTH into training pairs
├── train.py           # LoRA fine-tuning script
└── eval_ft.py         # Evaluate fine-tuned model
```

This is a **stretch goal**. Focus on Tasks 1-4 first.

---

## Task 6: Wire Into Configurable Pipeline

All architectures need to be callable from `src/pipelines/configurable.py`:

```python
def _run_langgraph_multi(claim, model):
    from src.agents.langgraph.graph import build_graph
    graph = build_graph()
    result = graph.invoke({"claim": claim})
    return {
        "verdict": result["verdict"]["verdict"],
        "explanation": result["verdict"]["explanation"],
        "evidence": result["evidence"],
    }


def _run_strands_rerouting(claim, model):
    from src.agents.strands.orchestrator_rerouting import run_pipeline_rerouting
    raw = run_pipeline_rerouting(claim)
    return raw["verdict"]
```

---

## Suggested Division of Work (Members 4 & 5)

| Member | Tasks | Why |
|--------|-------|-----|
| **Member 4** | LangGraph pipeline (Task 1) + Pipeline wiring (Task 6) | LangGraph is a self-contained framework to learn |
| **Member 5** | Rerouting (Task 2) + Model integration (Tasks 3-4) + Fine-tuning (Task 5) | Builds on existing Strands code |

---

## Testing Your Code

### Test LangGraph pipeline

```python
from src.agents.langgraph.graph import build_graph

graph = build_graph()
result = graph.invoke({"claim": "Vaccines cause autism"})
print(result["verdict"])
```

### Test rerouting

```python
from src.agents.strands.orchestrator_rerouting import run_pipeline_rerouting

result = run_pipeline_rerouting("Vitamin D supplements prevent COVID infection")
print(f"Loops: {result['rerouting_loops']}")
print(f"Verdict: {result['verdict']}")
```

### Test model switching

```python
from src.pipelines.configurable import run_experiment

# Claude (should work already)
r1 = run_experiment("Vaccines cause autism", model="claude-sonnet-4")

# GPT-4o-mini
r2 = run_experiment("Vaccines cause autism", model="gpt-4o-mini")

# Llama (needs Ollama running)
r3 = run_experiment("Vaccines cause autism", model="llama-3.1-8b")

# Compare
for r in [r1, r2, r3]:
    print(f"{r['experiment_config']['model']}: {r['verdict']}")
```

### Run via experiment runner

```bash
uv run python -m src.experiment_runner E6   # LangGraph
uv run python -m src.experiment_runner E7   # Rerouting
uv run python -m src.experiment_runner E8   # GPT-4o-mini
uv run python -m src.experiment_runner E9   # Llama
```

---

## Dependencies to Add

```bash
uv add langgraph langchain-core

# For fine-tuning (stretch)
uv add peft transformers datasets accelerate bitsandbytes
```

---

## Files Reference

| File | Status | Owner |
|------|--------|-------|
| `src/agents/strands/orchestrator.py` | Done — sequential pipeline | — |
| `src/agents/strands/claim_parser.py` | Done | — |
| `src/agents/strands/retrieval_agent.py` | Done | — |
| `src/agents/strands/evidence_reviewer.py` | Done | — |
| `src/agents/strands/verdict_agent.py` | Done | — |
| `src/agents/langgraph/` | **Empty → implement** | You |
| `src/agents/strands/orchestrator_rerouting.py` | **New → implement** | You |
| `src/pipelines/configurable.py` | Needs LangGraph + rerouting + model dispatch | You |
| `src/shared/llm.py` | Done — multi-provider | — |
