# Scientific Claims Fact-Checker

Comparing RAG and Agent Architectures for Scienstific Claim Verification.

A systematic study comparing **chunking strategies**, **retrieval methods**, **agent architectures**, and **LLM models** for automated scientific claim fact-checking — evaluated across claims with statistical significance testing.

Current data/loading path is based on the local **SciFact** corpus and claim set in `data/`, with **3 verdict classes**: `SUPPORTED`, `UNSUPPORTED`, and `INSUFFICIENT_EVIDENCE`.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Corpus Preparation](#corpus-preparation)
  - [Jupyter Notebook Kernel](#jupyter-notebook-kernel)
- [Experiment Design](#experiment-design)
  - [Three Configuration Axes](#three-configuration-axes)
  - [Experiment Configurations](#experiment-configurations)
- [What the POC Covers](#what-the-poc-covers)
- [Usage](#usage)
- [Evaluation Framework](#evaluation-framework)
- [Output Schema](#output-schema)
- [Tech Stack](#tech-stack)

---

## Overview

Given a health claim like _"Vaccines cause autism"_, the configurable pipeline:

1. **Chunks** the corpus using one of 4 strategies (fixed, semantic, section-aware, recursive)
2. **Retrieves** relevant evidence via naive cosine similarity or hybrid BM25+dense search with re-ranking
3. **Reasons** over the evidence using a single-pass LLM call or multi-agent orchestration
4. **Returns** a structured verdict (`SUPPORTED`, `UNSUPPORTED`, or `INSUFFICIENT_EVIDENCE`) with cited evidence and an explanation

The research question: _How do chunking sophistication, retrieval method, agent architecture, and model choice independently contribute to fact-checking accuracy and explanation quality?_

---

## Project Structure

```
health-claims-factchecker/
├── app/
│   └── streamlit_app.py                # Interactive demo UI
├── data/
│   ├── corpus.json                     # SciFact corpus (native fields: doc_id, title, abstract, structured)
│   ├── test_claims.json                # Balanced SciFact claims (normalized with expected_verdict)
│   └── corpus/
│       ├── processed/chunks.json       # Chunked corpus
│       └── embeddings/chroma_db/       # ChromaDB persistent vector store
├── Docs/
│   ├── Health_Claims_FactChecker_Proposal_v5.md
│   └── POC_BUILD_GUIDE.md
├── notebooks/
│   └── poc_comparison.ipynb            # Full POC walkthrough & analysis
├── results/
│   ├── experiments/                    # Per-experiment batch results (E1.json, etc.)
│   ├── comparison.json                 # P1 vs P6 POC outputs
│   ├── evaluation.json                 # LLM judge scores & grounding metrics
│   └── figures/                        # Generated charts
├── scripts/
│   └── build_scifact_dataset.py        # Build SciFact corpus and balanced claim set
├── src/
│   ├── chunking/                       # Chunking strategies
│   │   ├── __init__.py                 # Dispatcher: chunk_corpus(corpus, strategy=...)
│   │   ├── fixed.py                    # Fixed-size 200-token chunks (implemented)
│   │   ├── semantic.py                 # Sentence-embedding similarity boundaries
│   │   ├── section_aware.py            # Section-header-aware splitting
│   │   └── recursive.py               # Recursive splitting with metadata
│   ├── agents/
│   │   ├── strands/                    # AWS Strands Agent SDK agents
│   │   │   ├── claim_parser.py         # Agent 1: Decompose claim into sub-claims
│   │   │   ├── retrieval_agent.py      # Agent 2: Retrieval agent
│   │   │   ├── evidence_reviewer.py    # Agent 3: Flag contradictions & gaps
│   │   │   ├── verdict_agent.py        # Agent 4: Generate final verdict
│   │   │   ├── orchestrator.py         # Sequential 4-agent orchestrator
│   │   │   ├── confidence_gate.py      # Confidence scoring for gating (P7 POC)
│   │   │   └── orchestrator_gated.py   # Gated orchestrator (P7 POC)
│   │   └── langgraph/                  # LangGraph graph-based agents (stub)
│   ├── pipelines/
│   │   ├── configurable.py             # Single entry point: run_experiment()
│   │   └── p6_adv_multi/              # Multi-agent: Strands 4-agent pipeline
│   ├── retrieval/
│   │   ├── naive.py                    # Cosine similarity search (implemented)
│   │   ├── hybrid.py                   # BM25 + dense fusion
│   │   └── reranker.py                 # Cross-encoder re-ranking
│   ├── evaluation/
│   │   ├── llm_judge.py                # LLM-as-judge (4 dimensions, 1-5 scale)
│   │   ├── grounding_rate.py           # Grounding rate computation
│   │   ├── metrics.py                  # Verdict accuracy, McNemar's test, bootstrap CIs
│   │   ├── pairwise.py                 # Pairwise explanation comparison
│   │   └── run_eval.py                 # Full evaluation harness
│   ├── shared/
│   │   ├── schema.py                   # FactCheckResult unified output schema
│   │   ├── llm.py                      # Multi-provider LLM client (Anthropic, OpenAI, Ollama)
│   │   ├── vector_store.py             # ChromaDB setup & search
│   │   ├── embeddings.py               # Embedding model config
│   │   └── corpus_loader.py            # SciFact corpus loading helpers
│   ├── experiment_runner.py            # Batch execution with resumption (E1-E12)
│   └── compare.py                      # Run P1 vs P6 comparison
├── pyproject.toml
├── .env.example
└── uv.lock
```

---

## Setup

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or pip
- **API keys** — see [Environment Variables](#environment-variables)

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
# Anthropic API key (for Claude-based experiments and evaluation)
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# OpenAI API key (for GPT-4o-mini experiments E8, E11, E12)
OPENAI_API_KEY=your-openai-api-key

# AWS credentials (for Bedrock / Strands multi-agent experiments)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0

# Ollama (for open-source model experiments E9, E10 — runs locally)
OLLAMA_MODEL=llama3.1:8b
```

| Variable | Required For | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | E1-E7, Evaluation | Anthropic Claude API key |
| `OPENAI_API_KEY` | E8, E11, E12 | OpenAI API key for GPT-4o-mini |
| `AWS_ACCESS_KEY_ID` | E5-E7 | AWS credentials for Bedrock (Strands agents) |
| `AWS_SECRET_ACCESS_KEY` | E5-E7 | AWS credentials for Bedrock |
| `OLLAMA_MODEL` | E9, E10 | Local Ollama model for Llama experiments |

### Corpus Preparation

The SciFact corpus and balanced claim set are already included in the repository. To regenerate them or rebuild ChromaDB:

```bash
# Rebuild SciFact corpus.json and test_claims.json from the source archive
uv run python scripts/build_scifact_dataset.py --claims-per-class 100

# Indexing into ChromaDB happens automatically on first pipeline run
```

`data/test_claims.json` is stored in the repo's normalized format with `expected_verdict`; `src/experiment_runner.py` also accepts native SciFact-style `evidence_label` if you pass raw claim records programmatically.

### Jupyter Notebook Kernel

To run the POC notebook, register the project's virtual environment as a Jupyter kernel:

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

4. **Open `notebooks/poc_comparison.ipynb`** and select the **"Health Claims Fact-Checker (Python 3.11)"** kernel from the kernel picker.

> **Tip:** If you see `ModuleNotFoundError` when running cells, make sure you selected the correct kernel — not the system Python.

To remove the kernel later:

```bash
jupyter kernelspec uninstall health-claims
```

---

## Experiment Design

### Three Configuration Axes

The configurable pipeline (`src/pipelines/configurable.py`) accepts 4 parameters:

| Axis | Options | Description |
|------|---------|-------------|
| **Chunking Strategy** | `fixed`, `semantic`, `section_aware`, `recursive` | How the corpus is split into retrievable units |
| **Retrieval Method** | `naive`, `hybrid`, `hybrid_reranked` | How evidence is searched and ranked |
| **Agent Architecture** | `single_pass`, `strands_multi`, `langgraph_multi`, `strands_rerouting` | How reasoning is orchestrated |
| **Model** | `claude-sonnet-4`, `gpt-4o-mini`, `llama-3.1-8b`, `llama-3.1-8b-ft` | Which LLM performs reasoning |

### Experiment Configurations

Experiments currently defined in `src/experiment_runner.py`:

| ID | Name | Chunking | Retrieval | Agent | Model |
|----|------|----------|-----------|-------|-------|
| **E1** | Fixed chunking + naive RAG | fixed | naive | single_pass | Claude Sonnet |
| **E2** | Section-aware chunking + naive RAG | section_aware | naive | single_pass | Claude Sonnet |
| **E3** | Semantic chunking + naive RAG | semantic | naive | single_pass | Claude Sonnet |
| **E4** | Recursive chunking + naive RAG | recursive | naive | single_pass | Claude Sonnet |
| **E5** | Best chunking + hybrid RAG | recursive | hybrid | single_pass | Claude Sonnet |
| **E6** | Best chunking + hybrid reranked RAG | recursive | hybrid_reranked | single_pass | Claude Sonnet |
| **E7** | Best RAG + Strands agents | recursive | hybrid_reranked | strands_multi | Claude Sonnet |
| **E8** | Best RAG + LangGraph agents | recursive | hybrid_reranked | langgraph_multi | Claude Sonnet |
| **E9** | Best RAG + rerouting | recursive | hybrid_reranked | strands_rerouting | Claude Sonnet |
| **E10** | GPT-4o-mini + single-pass | recursive | hybrid_reranked | single_pass | GPT-4o-mini |
| **E11** | Llama 3.1 8B baseline | recursive | hybrid_reranked | single_pass | Llama 3.1 8B |
| **E12** | GPT-4o-mini + agents | recursive | hybrid_reranked | strands_multi | GPT-4o-mini |
| **E13** | Budget baseline | fixed | naive | single_pass | GPT-4o-mini |

---

## What the POC Covers

The POC validates the study premise by implementing and comparing the two extremes: **E1/P1** (simplest) and **P6** (most complex), plus a **P6-Gated** variant that optimises latency via confidence gating.

**POC Results (7 test claims):**

| Metric | P1 (Baseline) | P6 (Multi-Agent) |
|--------|---------------|-------------------|
| Verdict Accuracy | 4/7 | 4/7 |
| Explanation Quality (avg) | 3.68 | 4.71 |
| Grounding Rate | 81% | 79% |
| Total Latency | 36s | 888s |
| Total Cost | $0.055 | $0.279 |

Key finding: P6 produces significantly richer explanations (+1.03 quality score), but verdict accuracy is identical — suggesting retrieval quality is the bottleneck, not reasoning depth.

---

## Usage

### Run a single experiment

```python
from src.pipelines.configurable import run_experiment

result = run_experiment(
    "Vaccines cause autism",
    chunking_strategy="fixed",
    retrieval_method="naive",
    agent_architecture="single_pass",
    model="claude-sonnet-4",
)
print(result["verdict"])  # UNSUPPORTED
```

### Run a batch experiment

```bash
# List all available experiments
uv run python -m src.experiment_runner

# Run a specific experiment on all claims
uv run python -m src.experiment_runner E1
```

### Run the POC notebook

```bash
uv run jupyter notebook notebooks/poc_comparison.ipynb
```

### Run the evaluation harness

```bash
uv run python src/evaluation/run_eval.py
```

---

## Evaluation Framework

| Metric | Method | What It Measures |
|--------|--------|-----------------|
| **Verdict Accuracy** | Accuracy + Macro-F1 against SciFact labels | Correctness across verdict classes |
| **Explanation Quality** | LLM-as-Judge (4 dimensions, 1-5 scale) | Faithfulness, Specificity, Completeness, Nuance |
| **Grounding Rate** | Automated statement-level check | % of factual statements traceable to retrieved evidence |
| **Statistical Significance** | McNemar's test + bootstrap CIs | Whether differences between experiments are significant |
| **Pairwise Comparison** | LLM judge head-to-head | Which experiment produces better explanations |

Data sources for ground truth:
- **Verdict labels**: From SciFact claim annotations, normalized to the repo's 3-label schema
- **Explanation quality**: No reference explanations needed — LLM judge evaluates properties of the pipeline's own output (faithfulness to its own evidence, specificity, etc.)

---

## Output Schema

All experiments return a `FactCheckResult` (defined in `src/shared/schema.py`):

```json
{
  "claim": "Vaccines cause autism",
  "verdict": "UNSUPPORTED",
  "explanation": "Multiple large-scale studies...",
  "evidence": [
    {
      "source": "30986133",
      "passage": "Findings suggest that vaccinations are not associated...",
      "relevance_score": 0.85
    }
  ],
  "metadata": {
    "latency_seconds": 5.42,
    "total_tokens": 1465,
    "estimated_cost_usd": 0.0091,
    "pipeline": "fixed_naive_single_pass_claude-sonnet-4",
    "retrieval_method": "naive",
    "agent_type": "single_pass"
  },
  "experiment_config": {
    "chunking_strategy": "fixed",
    "retrieval_method": "naive",
    "agent_architecture": "single_pass",
    "model": "claude-sonnet-4"
  }
}
```

Valid verdicts: `SUPPORTED`, `UNSUPPORTED`, `INSUFFICIENT_EVIDENCE`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| Language | Python 3.11+ |
| Embeddings | [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) (local) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) (local, persisted) |
| LLM (Frontier) | Claude Sonnet 4 via Anthropic API / AWS Bedrock |
| LLM (Budget) | GPT-4o-mini via OpenAI API |
| LLM (Open-Source) | Llama 3.1 8B/70B via Ollama (local) |
| Agent Frameworks | [Strands Agent SDK](https://github.com/strands-agents/sdk-python), LangGraph |
| Dataset | SciFact corpus + claims (stored locally in `data/`) |
| Evaluation | LLM-as-Judge, McNemar's test, bootstrap CIs |
| Notebook | Jupyter |
| Demo | Streamlit |
