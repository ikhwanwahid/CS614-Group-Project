# Health Claims Fact-Checker

Automated health claim verification using RAG and multi-agent architectures, evaluated on the SciFact benchmark.

A systematic ablation study comparing **chunking strategies**, **retrieval methods**, **agent architectures**, and **LLM models** across 13 experiments on 300 balanced claims.

## Key Results

| Pipeline | Accuracy | Latency | Cost/Claim |
|----------|----------|---------|------------|
| Single-pass RAG (E4) | **82.0%** | 8.4s | $0.012 |
| Strands Multi-Agent (E7) | 65.3% | 68s | ~$0.30 |
| LangGraph Multi-Agent (E8) | 66.0% | 32s | ~$0.10 |
| Llama 3.1 8B RAG (E11) | 62.7% | 98s | ~$0 |
| E9c Smart Rerouting + S2 (out-of-corpus) | 66.7% | 22s | — |

RAG wins on closed corpora (+16-17pp over agents). Agents with Semantic Scholar external search win on out-of-corpus claims (+44.5pp over RAG).

## Project Structure

```
CS614-Group-Project/
├── data/
│   ├── corpus.json                       # SciFact corpus (5,183 abstracts)
│   ├── test_claims.json                  # 300 balanced claims (100 per verdict)
│   ├── e9b_test_claims.json              # 79 targeted E9c test claims
│   ├── out_of_corpus_claims.json         # 18 out-of-corpus claims for E9d
│   └── corpus/
│       ├── processed/                    # Chunked corpus per strategy
│       └── embeddings/chroma_db/         # ChromaDB vector store (gitignored)
├── notebooks/
│   ├── experiment_analysis.ipynb         # Main analysis (E1-E11)
│   └── e9c_analysis.ipynb               # E9c/E9d analysis (13 sections)
├── results/
│   ├── experiments/                      # Per-experiment results (E1-E11, E4b, E9c, E9d)
│   ├── accuracy_comparison.png           # All experiments bar chart
│   ├── confusion_matrices.png            # E4/E7/E8 confusion matrices
│   ├── cost_latency_analysis.png         # Cost vs accuracy scatter
│   ├── easy_vs_hard_claims.png           # Accuracy by claim difficulty
│   ├── e9c_*.png                         # E9c analysis charts
│   ├── e9d_out_of_corpus_comparison.png  # E9d vs E4b
│   └── s2_impact_comparison.png          # S2 value in-corpus vs out-of-corpus
├── scripts/
│   ├── fetch_corpus.py                   # PubMed corpus fetcher
│   └── reindex_corpus.py                 # Rebuild ChromaDB indexes
├── src/
│   ├── chunking/                         # 4 strategies: fixed, semantic, section_aware, recursive
│   ├── agents/
│   │   ├── strands/                      # Strands SDK agents
│   │   │   ├── claim_parser.py           # Agent 1: Decompose claim
│   │   │   ├── retrieval_agent.py        # Agent 2: Search corpus
│   │   │   ├── evidence_reviewer.py      # Agent 3: Review evidence quality
│   │   │   ├── verdict_agent.py          # Agent 4: Final verdict
│   │   │   ├── orchestrator.py           # E7: Sequential 4-agent pipeline
│   │   │   ├── orchestrator_rerouting_ext_v2.py  # E9c: Smart rerouting + S2
│   │   │   └── ...                       # Other orchestrator variants
│   │   └── langgraph/
│   │       ├── graph.py                  # E8: LangGraph state graph pipeline
│   │       ├── nodes.py                  # Graph node implementations
│   │       └── state.py                  # Typed pipeline state
│   ├── pipelines/
│   │   └── configurable.py              # Main entry point: run_experiment()
│   ├── retrieval/
│   │   ├── naive.py                     # Dense cosine similarity search
│   │   ├── hybrid.py                    # BM25 + dense fusion
│   │   ├── reranker.py                  # Cross-encoder reranking
│   │   └── semantic_scholar.py          # Semantic Scholar API (E9c/E9d)
│   ├── evaluation/
│   │   ├── llm_judge.py                 # LLM-as-judge scoring
│   │   ├── grounding_rate.py            # Evidence grounding metrics
│   │   └── run_eval.py                  # Evaluation harness
│   ├── shared/
│   │   ├── schema.py                    # FactCheckResult output schema
│   │   ├── llm.py                       # Multi-provider LLM client
│   │   ├── vector_store.py              # ChromaDB setup & search
│   │   ├── embeddings.py                # Embedding model config
│   │   └── corpus_loader.py             # SciFact corpus loader
│   └── experiment_runner.py             # Batch execution with resume support
├── presentation_content.md              # Slide content (19 slides)
├── pyproject.toml
├── .env.example
└── uv.lock
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- API keys (see below)

### Installation

```bash
git clone <repo-url>
cd CS614-Group-Project
uv sync
```

### Environment Variables

```bash
cp .env.example .env
```

| Variable | Required For | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | E1-E8, E9c | Anthropic Claude API |
| `AWS_ACCESS_KEY_ID` | E7, E9c | AWS Bedrock (Strands agents) |
| `AWS_SECRET_ACCESS_KEY` | E7, E9c | AWS Bedrock |
| `SEMANTIC_SCHOLAR_API_KEY` | E9c, E9d | Semantic Scholar external search |
| `OLLAMA_MODEL` | E11 | Local Llama 3.1 8B via Ollama |

## Experiments

| ID | Description | Chunking | Retrieval | Architecture | Model | Accuracy |
|----|-------------|----------|-----------|-------------|-------|----------|
| E1 | Fixed chunking | fixed | naive | single_pass | Claude Sonnet 4 | 79.7% |
| E2 | Section-aware | section_aware | naive | single_pass | Claude Sonnet 4 | 77.3% |
| E3 | Semantic | semantic | naive | single_pass | Claude Sonnet 4 | 76.3% |
| E4 | Recursive (best RAG) | recursive | naive | single_pass | Claude Sonnet 4 | **82.0%** |
| E5 | Hybrid retrieval | recursive | hybrid | single_pass | Claude Sonnet 4 | 79.0% |
| E6 | Hybrid + reranking | recursive | hybrid_reranked | single_pass | Claude Sonnet 4 | 81.3% |
| E7 | Strands multi-agent | recursive | naive | strands_multi | Claude Sonnet 4 | 65.3% |
| E8 | LangGraph multi-agent | recursive | naive | langgraph_multi | Claude Sonnet 4 | 66.0% |
| E9c | Smart rerouting + S2 | recursive | naive | strands_rerouting_ext_v2 | Claude Sonnet 4 | 84.7%* |
| E9d | E9c on out-of-corpus | recursive | naive | strands_rerouting_ext_v2 | Claude Sonnet 4 | 66.7% |
| E4b | RAG on out-of-corpus | recursive | naive | single_pass | Claude Sonnet 4 | 22.2% |
| E11 | Llama 3.1 8B | recursive | naive | single_pass | Llama 3.1 8B | 62.7% |

*Projected from 79-claim targeted evaluation (54 E4 failures + 25 control)

## Usage

### Run an experiment

```bash
uv run python -m src.experiment_runner E4
```

### Run with custom claims

```bash
uv run python -m src.experiment_runner E9d --claims=data/out_of_corpus_claims.json
```

### Run analysis notebooks

```bash
uv run jupyter notebook notebooks/experiment_analysis.ipynb
uv run jupyter notebook notebooks/e9c_analysis.ipynb
```

## Key Findings

1. **Chunking > retrieval**: Recursive (82.0%) vs semantic (76.3%) = 5.7pp; naive vs hybrid retrieval = 3.0pp
2. **RAG wins on closed corpora**: 82.0% at 8.4s/$0.012 vs agents at 65-66% at 32-68s/$0.10-$0.30
3. **Agents excel at contradiction detection**: UNSUPPORTED recall 83-84% (matches RAG), SUPPORTED recall drops to 49%
4. **External search is context-dependent**: Hurts on in-corpus (noise), transformative on out-of-corpus (+44.5pp)
5. **Model quality > architecture**: Claude vs Llama gap (19.3pp) exceeds RAG vs agents gap (16-17pp)
6. **Corpus ceiling**: 97.6% of persistent E9c failures match E4 errors exactly

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store | ChromaDB (persistent) |
| LLM | Claude Sonnet 4 (Anthropic API / AWS Bedrock) |
| LLM (local) | Llama 3.1 8B via Ollama |
| Agent Frameworks | [Strands SDK](https://github.com/strands-agents/sdk-python), LangGraph |
| External Search | Semantic Scholar API (200M+ papers) |
| Dataset | SciFact (5,183 abstracts, 300 claims) |
