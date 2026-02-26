# Health Claims Fact-Checker POC — Build Guide

## Objective

Build P1 (Naive RAG + Single-Pass) and P6 (Advanced RAG + Multi-Agent) to validate whether the 3×2 comparison study is viable. Run both pipelines on the same claims and compare outputs.

Refer to `Health_Claims_FactChecker_Proposal_v3.md` for full project context.

---

## Shared Components

### Output Schema

All pipelines must return this structure:

```python
{
    "claim": str,
    "verdict": str,  # SUPPORTED | UNSUPPORTED | OVERSTATED | INSUFFICIENT_EVIDENCE
    "explanation": str,
    "evidence": [
        {
            "source": str,
            "passage": str,
            "relevance_score": float
        }
    ],
    "metadata": {
        "latency_seconds": float,
        "total_tokens": int,
        "estimated_cost_usd": float,
        "pipeline": str,  # "P1" or "P6"
    }
}
```

### Corpus

~20-30 vaccine-related PubMed Open Access abstracts. Download manually or via PubMed E-utilities API.

Search queries to seed the corpus:
- "COVID-19 vaccine efficacy"
- "MMR vaccine autism"
- "Vitamin D COVID prevention"
- "mRNA vaccine DNA"
- "HPV vaccine safety"

Store as JSON: `data/corpus.json` with fields `pmid`, `title`, `abstract`, `authors`, `year`.

### Test Claims

```python
TEST_CLAIMS = [
    # Easy — clear verdict
    "Vaccines cause autism",
    "The MMR vaccine is linked to autism in children",
    
    # Nuanced — overstated
    "Vitamin D supplements prevent COVID infection",
    "Intermittent fasting reverses Type 2 diabetes",
    
    # Mechanistic — needs explanation quality
    "mRNA vaccines alter your DNA",
    
    # Mixed evidence
    "COVID vaccines are effective against all variants",
    
    # Simple supported
    "Flu vaccines reduce hospitalisation in elderly patients",
]
```

### LLM

Use the same model for both pipelines. Options:
- Claude Sonnet via AWS Bedrock (preferred — aligns with Strands)
- GPT-4o-mini via OpenAI API (fallback)

---

## P1: Naive RAG + Single-Pass

### Flow

```
claim → embed → vector search (top-5) → single LLM call → output
```

### Implementation

1. **Chunk corpus**: Split each abstract into ~200 token chunks with 50 token overlap
2. **Embed**: Use `text-embedding-3-small` (OpenAI) or Titan Embeddings (Bedrock)
3. **Vector store**: ChromaDB (local, no server needed)
4. **Retrieve**: Cosine similarity, return top-5 chunks
5. **Generate**: Single LLM call with prompt:

```
You are a health claim fact-checker. Given the following evidence passages and a health claim, provide:
1. A verdict: SUPPORTED, UNSUPPORTED, OVERSTATED, or INSUFFICIENT_EVIDENCE
2. An explanation justifying your verdict (2-3 sentences)
3. Which evidence passages you relied on

Claim: {claim}

Evidence:
{passages}

Respond in JSON format matching this schema:
{output_schema}
```

### Dependencies

- `chromadb`
- `openai` or `boto3` (for embeddings)
- `anthropic` or `boto3` (for LLM)

---

## P6: Advanced RAG + Multi-Agent

### Flow

```
claim → [Claim Parser Agent]
      → [Retrieval Agent] (per sub-claim: PubMed API + local corpus, re-ranking)
      → [Evidence Reviewer Agent]
      → [Verdict Agent]
      → output
```

### Implementation with Strands Agent SDK

#### Agent 1: Claim Parser

**Input:** Raw claim string
**Output:** List of sub-claims with retrieval queries

```python
# Tool: decompose_claim
# The agent decomposes the claim into verifiable sub-claims
# and generates a targeted search query for each.
#
# Example:
# Input: "Vitamin D supplements prevent COVID infection"
# Output: [
#     {"sub_claim": "Vitamin D affects COVID outcomes", "query": "vitamin D COVID-19 outcomes RCT"},
#     {"sub_claim": "The effect is prevention not just severity reduction", "query": "vitamin D COVID prevention vs severity"},
#     {"sub_claim": "Effect applies to general population", "query": "vitamin D COVID deficiency vs general population"}
# ]
```

#### Agent 2: Retrieval Agent

**Input:** Sub-claims with queries from Agent 1
**Output:** Retrieved evidence per sub-claim, ranked

For each sub-claim:
1. Search local ChromaDB (same as P1 corpus)
2. Search PubMed API (`esearch` + `efetch` via E-utilities) — top 5 results per query
3. Combine results
4. Re-rank using a simple cross-encoder or LLM-based relevance scoring
5. Return top-3 passages per sub-claim

Tools to implement:
- `search_local_corpus(query: str) -> list[dict]`
- `search_pubmed(query: str) -> list[dict]`
- `rerank_passages(query: str, passages: list[dict]) -> list[dict]`

#### Agent 3: Evidence Reviewer

**Input:** All retrieved evidence across sub-claims
**Output:** Reviewed evidence with flags

The agent reviews evidence and:
- Flags contradictions between sub-claim evidence
- Identifies gaps (sub-claims with weak evidence)
- Notes evidence quality (study type, sample size if available)

#### Agent 4: Verdict Agent

**Input:** Reviewed evidence from Agent 3
**Output:** Final output matching shared schema

Generates:
- Overall verdict
- Explanation that addresses each sub-claim
- Cited evidence

### Strands Orchestration

```python
from strands import Agent
from strands.models.bedrock import BedrockModel

model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1"
)

# Define each agent with its tools and system prompt
claim_parser = Agent(
    model=model,
    system_prompt="You are a claim decomposition specialist...",
    tools=[decompose_claim]
)

retrieval_agent = Agent(
    model=model,
    system_prompt="You are a medical evidence retrieval specialist...",
    tools=[search_local_corpus, search_pubmed, rerank_passages]
)

evidence_reviewer = Agent(
    model=model,
    system_prompt="You are a medical evidence reviewer...",
    tools=[review_evidence]
)

verdict_agent = Agent(
    model=model,
    system_prompt="You are a health claim verdict generator...",
    tools=[generate_verdict]
)

# Orchestrate: run sequentially, pass output as input to next
def run_p6(claim: str) -> dict:
    sub_claims = claim_parser(f"Decompose this claim: {claim}")
    evidence = retrieval_agent(f"Retrieve evidence for: {sub_claims}")
    reviewed = evidence_reviewer(f"Review this evidence: {evidence}")
    result = verdict_agent(f"Generate verdict: {reviewed}")
    return result
```

Adjust orchestration pattern based on Strands SDK conventions — the above is illustrative.

---

## Project Structure

This structure serves as both the POC and the full project skeleton. Members slot into their respective folders.

```
health-claims-factchecker/
├── data/
│   ├── corpus/
│   │   ├── raw/                     # Raw PubMed abstracts, WHO docs, CDC docs
│   │   ├── processed/               # Chunked, cleaned
│   │   └── embeddings/              # Pre-computed embeddings
│   ├── eval/
│   │   ├── antivax.json             # ANTi-Vax benchmark
│   │   ├── pubhealth.json           # PUBHEALTH benchmark
│   │   └── custom_claims.json       # Team-curated claims with ground truth
│   └── test_claims.json             # POC test claims with expected verdicts
│
├── src/
│   ├── shared/                      # Used by ALL pipelines
│   │   ├── schema.py                # Output schema definition
│   │   ├── corpus_loader.py         # Load and chunk corpus
│   │   ├── embeddings.py            # Embedding utilities
│   │   ├── vector_store.py          # ChromaDB setup and search
│   │   └── llm.py                   # LLM client wrapper (single model config)
│   │
│   ├── pipelines/
│   │   ├── p1_naive_single/         # Member 2
│   │   │   └── pipeline.py
│   │   ├── p2_naive_multi/          # Member 2
│   │   │   └── pipeline.py
│   │   ├── p3_inter_single/         # Member 3
│   │   │   └── pipeline.py
│   │   ├── p4_inter_multi/          # Member 3
│   │   │   └── pipeline.py
│   │   ├── p5_adv_single/           # Member 4
│   │   │   └── pipeline.py
│   │   └── p6_adv_multi/            # Member 4
│   │       └── pipeline.py
│   │
│   ├── retrieval/                   # Shared retrieval components
│   │   ├── naive.py                 # Cosine similarity search
│   │   ├── hybrid.py                # BM25 + dense + RRF
│   │   ├── reranker.py              # Cross-encoder re-ranking
│   │   ├── pubmed_search.py         # PubMed E-utilities API
│   │   ├── query_rewriter.py        # LLM query expansion
│   │   ├── claim_decomposer.py      # Claim → sub-queries
│   │   └── evidence_ranker.py       # Evidence hierarchy ranking
│   │
│   ├── agents/                      # Member 5 — shared agent templates
│   │   ├── langgraph/
│   │   │   ├── claim_parser.py
│   │   │   ├── retrieval_agent.py
│   │   │   ├── evidence_reviewer.py
│   │   │   ├── verdict_agent.py
│   │   │   └── graph.py             # LangGraph orchestration
│   │   └── strands/
│   │       ├── claim_parser.py
│   │       ├── retrieval_agent.py
│   │       ├── evidence_reviewer.py
│   │       ├── verdict_agent.py
│   │       └── orchestrator.py      # Strands orchestration
│   │
│   └── evaluation/                  # Member 6
│       ├── llm_judge.py             # LLM-as-judge rubric scoring
│       ├── pairwise.py              # Pairwise comparison
│       ├── grounding_rate.py        # Automated grounding rate
│       ├── metrics.py               # Accuracy, F1 computation
│       └── run_eval.py              # Run full evaluation across all pipelines
│
├── app/
│   └── streamlit_app.py             # Member 6 — demo UI
│
├── results/
│   ├── comparison.json              # Pipeline comparison outputs
│   └── figures/                     # Charts, tables for report
│
├── notebooks/
│   └── poc_comparison.ipynb         # POC side-by-side analysis
│
├── requirements.txt
└── README.md
```

### How Members Slot In

| Member | Works in | Depends on |
|--------|----------|-----------|
| Member 1 (Data) | `data/`, `src/shared/` | — |
| Member 2 (Naive RAG) | `src/pipelines/p1_*`, `src/pipelines/p2_*`, `src/retrieval/naive.py` | `src/shared/`, `src/agents/` |
| Member 3 (Intermediate RAG) | `src/pipelines/p3_*`, `src/pipelines/p4_*`, `src/retrieval/hybrid.py`, `src/retrieval/reranker.py` | `src/shared/`, `src/agents/` |
| Member 4 (Advanced RAG) | `src/pipelines/p5_*`, `src/pipelines/p6_*`, `src/retrieval/claim_decomposer.py`, `src/retrieval/pubmed_search.py`, `src/retrieval/evidence_ranker.py` | `src/shared/`, `src/agents/` |
| Member 5 (Agent Frameworks) | `src/agents/` | `src/shared/` |
| Member 6 (Eval & Demo) | `src/evaluation/`, `app/` | All pipelines via shared schema |

Each pipeline's `pipeline.py` must implement:

```python
def run(claim: str) -> dict:
    """Run pipeline on a claim. Returns output matching shared schema."""
    ...
```

This contract lets `src/evaluation/run_eval.py` and `app/streamlit_app.py` call any pipeline interchangeably.

---

## Comparison Script

`src/compare.py` should:

1. Load test claims from `data/test_claims.json`
2. Run each claim through P1 and P6 (import from `src/pipelines/p1_naive_single/pipeline.py` and `src/pipelines/p6_adv_multi/pipeline.py`)
3. Print side-by-side output for each claim:
   - Verdict (match/mismatch)
   - Explanation (qualitative comparison)
   - Evidence sources used
   - Latency and token count
4. Save results to `results/comparison.json`

---

## Success Criteria

The POC validates the study if:

1. **P6 produces visibly better explanations than P1** on at least 3-4 of the nuanced claims
2. **P1 and P6 agree on easy claims** (both get "Vaccines cause autism" → UNSUPPORTED)
3. **P6 shows measurably higher latency/cost** — confirming there is a real tradeoff to study
4. **The output schema works** for both pipelines without modification

If P1 and P6 produce near-identical outputs across all claims, the study premise is weak and needs rethinking.

---

## Notes

- Don't optimise prompts heavily — the POC is about validating the structure, not maximising performance
- Keep the corpus small — 20-30 abstracts is enough to prove the concept
- PubMed E-utilities API is free but requires an API key for higher rate limits (register at NCBI)
- ChromaDB runs locally with no server setup needed
