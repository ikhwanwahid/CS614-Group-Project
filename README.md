# Health Claims Fact-Checker

Comparing RAG and Agent Architectures for Health Claim Verification.

A systematic study comparing **6 pipeline variants** across a 3x2 matrix of retrieval sophistication (Naive / Intermediate / Advanced RAG) and agent orchestration (Single-Pass / Multi-Agent) for automated health claim fact-checking.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Corpus Preparation](#corpus-preparation)
  - [Jupyter Notebook Kernel](#jupyter-notebook-kernel)
- [Pipeline Architectures](#pipeline-architectures)
- [What the POC Covers](#what-the-poc-covers)
  - [P1: Naive RAG + Single-Pass](#p1-naive-rag--single-pass)
  - [P6: Advanced RAG + Multi-Agent](#p6-advanced-rag--multi-agent)
  - [P6-Gated: Confidence Gating](#p6-gated-confidence-gating)
  - [Evaluation Framework](#evaluation-framework)
- [Usage](#usage)
- [Test Claims](#test-claims)
- [Output Schema](#output-schema)
- [Tech Stack](#tech-stack)

---

## Overview

Given a health claim like _"Vaccines cause autism"_, each pipeline:

1. **Retrieves** relevant evidence from a local PubMed-sourced corpus (ChromaDB) and optionally live PubMed API
2. **Reasons** over the evidence using one or more LLM agents
3. **Returns** a structured verdict (`SUPPORTED`, `UNSUPPORTED`, `OVERSTATED`, or `INSUFFICIENT_EVIDENCE`) with cited evidence and an explanation

The research question: _Where do gains in accuracy and explanation quality justify the added complexity and cost of better retrieval and agentic task decomposition?_

---

## Project Structure

```
health-claims-factchecker/
├── app/
│   └── streamlit_app.py              # Interactive demo UI
├── data/
│   ├── corpus.json                    # 36 PubMed abstracts (vaccine & health topics)
│   ├── test_claims.json               # 7 test claims with expected verdicts
│   └── corpus/
│       ├── processed/chunks.json      # Chunked corpus (84 chunks)
│       └── embeddings/chroma_db/      # ChromaDB persistent vector store
├── Docs/
│   ├── Health_Claims_FactChecker_Proposal_v3.md
│   └── POC_BUILD_GUIDE.md
├── notebooks/
│   └── poc_comparison.ipynb           # Full POC walkthrough & analysis
├── results/
│   ├── comparison.json                # P1 vs P6 outputs
│   ├── evaluation.json                # LLM judge scores & grounding metrics
│   ├── p1_results.json
│   └── figures/                       # Generated charts
├── scripts/
│   └── fetch_corpus.py                # Fetch PubMed abstracts
├── src/
│   ├── agents/
│   │   ├── strands/                   # AWS Strands Agent SDK agents
│   │   │   ├── claim_parser.py        # Agent 1: Decompose claim into sub-claims
│   │   │   ├── retrieval_agent.py     # Agent 2: Search corpus + PubMed
│   │   │   ├── evidence_reviewer.py   # Agent 3: Flag contradictions & gaps
│   │   │   ├── verdict_agent.py       # Agent 4: Generate final verdict
│   │   │   ├── orchestrator.py        # Sequential 4-agent orchestrator
│   │   │   ├── confidence_gate.py     # Confidence scoring for gating
│   │   │   └── orchestrator_gated.py  # Gated orchestrator (skip agents when possible)
│   │   └── langgraph/                 # LangGraph-based agents (alternative framework)
│   ├── pipelines/
│   │   ├── p1_naive_single/           # Naive RAG + Single-Pass (implemented)
│   │   ├── p2_naive_multi/            # Naive RAG + Multi-Agent
│   │   ├── p3_inter_single/           # Intermediate RAG + Single-Pass
│   │   ├── p4_inter_multi/            # Intermediate RAG + Multi-Agent
│   │   ├── p5_adv_single/            # Advanced RAG + Single-Pass
│   │   └── p6_adv_multi/             # Advanced RAG + Multi-Agent (implemented)
│   │       ├── pipeline.py            # Standard P6 pipeline
│   │       └── pipeline_gated.py      # P6 with confidence gating
│   ├── retrieval/                     # Retrieval components
│   │   ├── naive.py                   # Cosine similarity search
│   │   ├── hybrid.py                  # BM25 + dense search
│   │   ├── reranker.py                # Cross-encoder re-ranking
│   │   ├── pubmed_search.py           # Live PubMed E-utilities API
│   │   ├── query_rewriter.py          # LLM-based query expansion
│   │   └── claim_decomposer.py        # Claim decomposition
│   ├── evaluation/
│   │   ├── llm_judge.py               # LLM-as-judge scoring (4 dimensions)
│   │   ├── grounding_rate.py          # Grounding rate computation
│   │   └── run_eval.py                # Full evaluation harness
│   ├── shared/
│   │   ├── schema.py                  # FactCheckResult unified output schema
│   │   ├── llm.py                     # LLM client wrapper (Anthropic API)
│   │   ├── vector_store.py            # ChromaDB setup & search
│   │   ├── embeddings.py              # Embedding model config
│   │   └── corpus_loader.py           # Corpus loading & chunking
│   └── compare.py                     # Run P1 vs P6 comparison
├── pyproject.toml
├── .env.example
└── uv.lock
```

---

## Setup

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or pip
- **API keys** for Anthropic (P1) and AWS Bedrock (P6) — see [Environment Variables](#environment-variables)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repo-url>
   cd CS614-Group-Project
   ```

2. **Install dependencies with uv:**

   ```bash
   uv sync
   ```

   Or with pip:

   ```bash
   pip install -e .
   ```

### Environment Variables

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Anthropic API key (for P1 LLM calls and evaluation)
ANTHROPIC_API_KEY=your-anthropic-api-key

# Model configuration
ANTHROPIC_MODEL=claude-sonnet-4-20250514
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0

# AWS credentials (for Bedrock / Strands — needed for P6)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1
```

| Variable | Required For | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | P1, Evaluation | Anthropic Claude API key |
| `ANTHROPIC_MODEL` | P1 | Model ID (default: `claude-sonnet-4-20250514`) |
| `AWS_ACCESS_KEY_ID` | P6 | AWS credentials for Bedrock |
| `AWS_SECRET_ACCESS_KEY` | P6 | AWS credentials for Bedrock |
| `AWS_DEFAULT_REGION` | P6 | AWS region (default: `us-east-1`) |
| `BEDROCK_MODEL_ID` | P6 | Bedrock model ID (default: `us.anthropic.claude-sonnet-4-20250514-v1:0`) |

### Corpus Preparation

The corpus is already included in the repository (`data/corpus.json`). To re-fetch from PubMed or index into ChromaDB:

```bash
# Re-fetch PubMed abstracts (optional — corpus.json is already provided)
uv run python scripts/fetch_corpus.py

# Indexing into ChromaDB happens automatically on first pipeline run
# or when running the notebook (Section 2)
```

### Jupyter Notebook Kernel

To run the POC notebook, you need to register the project's virtual environment as a Jupyter kernel:

1. **Install `ipykernel` into the project environment:**

   ```bash
   uv pip install ipykernel
   ```

2. **Register the kernel:**

   ```bash
   uv run python -m ipykernel install --user --name health-claims --display-name "Health Claims Fact-Checker (Python 3.11)"
   ```

3. **Launch Jupyter:**

   ```bash
   uv run jupyter notebook
   ```

4. **Open `notebooks/poc_comparison.ipynb`** and select the **"Health Claims Fact-Checker (Python 3.11)"** kernel from the kernel picker (top right or Kernel > Change Kernel).

> **Tip:** If you see `ModuleNotFoundError` when running cells, make sure you selected the correct kernel — not the system Python.

To remove the kernel later:

```bash
jupyter kernelspec uninstall health-claims
```

---

## Pipeline Architectures

The study compares 6 pipelines across a 3x2 matrix:

| | Single-Pass | Multi-Agent |
|---|---|---|
| **Naive RAG** | P1 (implemented) | P2 |
| **Intermediate RAG** | P3 | P4 |
| **Advanced RAG** | P5 | P6 (implemented) |

Every pipeline implements the same contract:

```python
def run(claim: str) -> dict:
    """Returns output matching FactCheckResult schema."""
```

This allows any pipeline to be used interchangeably with the evaluation harness and the Streamlit app.

---

## What the POC Covers

The POC validates the study premise by implementing and comparing the two extremes of the matrix: **P1** (simplest) and **P6** (most complex), plus a **P6-Gated** variant that optimises P6's latency.

### P1: Naive RAG + Single-Pass

```
claim → embed → cosine similarity search (top-5) → single LLM call → verdict
```

- **Retrieval:** Raw claim embedded with PubMedBERT, top-5 ChromaDB cosine similarity
- **Reasoning:** Single Claude Sonnet call with all evidence in one prompt
- **Latency:** ~5s per claim
- **Strengths:** Fast, cheap, simple
- **Weaknesses:** No claim decomposition, no re-ranking, single-shot reasoning

### P6: Advanced RAG + Multi-Agent

```
claim → [Claim Parser] → [Retrieval Agent] → [Evidence Reviewer] → [Verdict Agent] → verdict
```

4 specialised agents orchestrated sequentially via the AWS Strands Agent SDK:

| Agent | Role | Key Capability |
|-------|------|----------------|
| **Claim Parser** | Decomposes claim into 2-4 verifiable sub-claims | Generates targeted PubMed search queries per sub-claim |
| **Retrieval Agent** | Retrieves evidence per sub-claim | Searches both local ChromaDB corpus and live PubMed API |
| **Evidence Reviewer** | Quality control | Flags contradictions, gaps, weak evidence |
| **Verdict Agent** | Final verdict | Synthesises evidence into nuanced verdict with citations |

- **Latency:** ~90s per claim
- **Strengths:** Nuanced verdicts, multi-source retrieval, structured multi-step reasoning
- **Weaknesses:** Slower, more expensive, more failure points

### P6-Gated: Confidence Gating

An optimisation that short-circuits expensive agents when local evidence is already decisive:

```
claim → [Claim Parser] → [Local ChromaDB Search] → [Confidence Gate]
   HIGH → [Verdict Agent with local evidence]            (~50s, 2 agent calls)
   LOW  → [Full pipeline: Retrieval + Reviewer + Verdict] (~90s, 4 agent calls)
```

**Scoring logic:** For each sub-claim, search ChromaDB (top-5) and score based on:
- **Hit relevance:** How many hits have L2 distance < 0.45
- **Average distance:** Of the top-3 hits per sub-claim
- **Coverage ratio:** What percentage of sub-claims have adequate local evidence

Gate triggers at **score >= 0.7 AND coverage >= 75%**.

Expected savings: ~40-50% latency reduction on claims with strong local corpus coverage.

### Evaluation Framework

The POC includes three evaluation dimensions:

| Metric | Method | What It Measures |
|--------|--------|-----------------|
| **Verdict Accuracy** | Comparison against expected verdicts | Correctness |
| **Explanation Quality** | LLM-as-Judge (4 dimensions, 1-5 scale) | Faithfulness, Specificity, Completeness, Nuance |
| **Grounding Rate** | Automated statement-level check | % of factual statements traceable to retrieved evidence |

Plus cost/latency tracking per claim.

**POC Results (7 test claims):**

| Metric | P1 | P6 |
|--------|----|----|
| Verdict Accuracy | 4/7 | 4/7 |
| Explanation Quality (avg) | 3.68 | 4.71 |
| Grounding Rate | 81% | 79% |
| Total Latency | 36s | 888s |
| Total Cost | $0.055 | $0.279 |

Key finding: P6 produces significantly richer and more nuanced explanations (+1.03 overall quality score), especially on nuanced claims, at the cost of ~25x latency and ~5x cost.

---

## Usage

### Run a single pipeline

```python
from src.pipelines.p1_naive_single.pipeline import run as run_p1
from src.pipelines.p6_adv_multi.pipeline import run as run_p6
from src.pipelines.p6_adv_multi.pipeline_gated import run as run_p6g

result = run_p1("Vaccines cause autism")
print(result["verdict"])       # UNSUPPORTED
print(result["explanation"])

result = run_p6("Vaccines cause autism")
print(result["verdict"])       # UNSUPPORTED

result = run_p6g("Vaccines cause autism")
print(result["gating_info"]["path"])  # SHORT_CIRCUIT or FULL_PIPELINE
```

### Run the full comparison

```bash
uv run python src/compare.py
```

### Run the evaluation harness

```bash
uv run python src/evaluation/run_eval.py
```

### Run the POC notebook

```bash
uv run jupyter notebook notebooks/poc_comparison.ipynb
```

The notebook walks through every step: corpus preparation, embedding, retrieval testing, running both pipelines on all 7 claims, verdict comparison, LLM-as-Judge evaluation, grounding rate analysis, cost/latency charts, and the confidence gating experiment.

---

## Test Claims

7 claims of varying difficulty defined in `data/test_claims.json`:

| # | Claim | Expected Verdict | Difficulty |
|---|-------|-----------------|------------|
| 1 | Vaccines cause autism | UNSUPPORTED | easy |
| 2 | The MMR vaccine is linked to autism in children | UNSUPPORTED | easy |
| 3 | Vitamin D supplements prevent COVID infection | OVERSTATED | nuanced |
| 4 | Intermittent fasting reverses Type 2 diabetes | OVERSTATED | nuanced |
| 5 | mRNA vaccines alter your DNA | UNSUPPORTED | mechanistic |
| 6 | COVID vaccines are effective against all variants | OVERSTATED | mixed_evidence |
| 7 | Flu vaccines reduce hospitalisation in elderly patients | SUPPORTED | simple_supported |

---

## Output Schema

All pipelines return a `FactCheckResult` (defined in `src/shared/schema.py`):

```json
{
  "claim": "Vaccines cause autism",
  "verdict": "UNSUPPORTED",
  "explanation": "Multiple large-scale studies...",
  "evidence": [
    {
      "source": "PMID:30986133",
      "passage": "Findings suggest that vaccinations are not associated...",
      "relevance_score": 0.85
    }
  ],
  "metadata": {
    "latency_seconds": 5.42,
    "total_tokens": 1465,
    "estimated_cost_usd": 0.0091,
    "pipeline": "P1",
    "retrieval_method": "naive_rag",
    "agent_type": "single_pass"
  }
}
```

Valid verdicts: `SUPPORTED`, `UNSUPPORTED`, `OVERSTATED`, `INSUFFICIENT_EVIDENCE`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| Language | Python 3.11+ |
| Embeddings | [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) (local, no API key) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) (local, persisted) |
| LLM (P1) | Claude Sonnet 4 via Anthropic API |
| LLM (P6) | Claude Sonnet 4 via AWS Bedrock |
| Agent Framework | [Strands Agent SDK](https://github.com/strands-agents/sdk-python) |
| PubMed Access | Biopython Entrez (E-utilities API) |
| Evaluation | LLM-as-Judge + automated grounding rate |
| Notebook | Jupyter |
| Demo | Streamlit |
