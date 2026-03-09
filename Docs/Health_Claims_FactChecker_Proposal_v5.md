# Comparing RAG and Agent Architectures for Health Claim Verification (v5)

## A Systematic Study of Chunking, Orchestration, and Model Strategies

---

## 1. Research Question

> **How do chunking strategies, agent architectures, and model choices affect health claim verification — and where do the gains justify the added complexity and cost?**

We build a **configurable pipeline** and systematically vary three axes — **RAG/chunking strategy**, **agent architecture**, and **LLM model** (proprietary and open-source) — measuring verdict accuracy, explanation quality, retrieval precision, latency, and cost across **120+ health claims**.

### Three Axes of Comparison

**RAG axis:** Does investing in sophisticated chunking (semantic, section-aware, recursive) and retrieval (hybrid search, re-ranking, claim decomposition, live PubMed API) improve the quality of evidence that reaches the LLM?

**Agent axis:** Does structured task decomposition — breaking a complex task into specialised sequential steps — produce better outcomes than a single monolithic prompt? Does the orchestration framework matter (Strands vs LangGraph)? Can adaptive rerouting improve on fixed sequential pipelines?

**Model axis:** How do proprietary frontier models (Claude Sonnet, GPT-4o-mini) compare to open-source alternatives (Llama 3, Mistral)? Can fine-tuning a small open-source model on domain-specific data close the gap with larger proprietary models?

**Interaction effects:** The configurable pipeline allows us to isolate each axis and measure interactions — e.g., do agents help more with a weaker model? Does better chunking matter more than a better model?

---

## 2. Problem Statement

Health misinformation is widespread — social media amplifies unverified claims, news articles misrepresent findings, and AI chatbots may hallucinate citations. Patients make treatment decisions based on inaccurate information.

### Why This Task Suits a Comparative Study

1. **Ground truth exists** — benchmark datasets (ANTi-Vax, PUBHEALTH) provide labeled claims for verdict accuracy
2. **Claims vary in difficulty** — from simple factual to nuanced claims requiring multi-step reasoning
3. **Evidence is publicly accessible** — PubMed, WHO guidelines, open-access medical literature
4. **Explanation quality matters** — a correct verdict with a poor explanation is insufficient; this motivates our LLM-as-judge evaluation which scores pipeline outputs independently of any reference explanation

### Scope

Health claims spanning vaccines, nutrition, supplements, and chronic disease — broadened from the POC's vaccine-only focus to support a 120+ claim evaluation set.

---

## 3. Study Design: Configurable Pipeline with Three Axes

Instead of building many separate pipelines, we build **one configurable pipeline** with interchangeable components along three axes. Each experiment is a specific combination of chunking strategy, agent architecture, and model.

```python
run_experiment(
    claim,
    chunking_strategy  = "fixed" | "semantic" | "section_aware" | "recursive",
    retrieval_method   = "naive" | "hybrid" | "hybrid_reranked",
    agent_architecture = "single_pass" | "strands_multi" | "langgraph_multi" | "strands_rerouting",
    model              = "claude-sonnet-4" | "gpt-4o-mini" | "claude-haiku" | "llama-3.1-8b" | "llama-3.1-8b-ft",
)
```

### 3.1 Chunking Strategies (RAG Pair — 2 Members)

| Strategy | Description | What It Tests |
|----------|-------------|---------------|
| **Fixed (baseline)** | 200-token chunks, 50-token overlap, word-boundary split | Control — current approach |
| **Semantic** | Split at cosine similarity drops between adjacent sentences (using PubMedBERT embeddings) | Does respecting topic boundaries improve retrieval? |
| **Section-aware** | Parse PubMed abstract sections (Background / Methods / Results / Conclusions), chunk per-section | Does keeping methodological context together help? |
| **Recursive + metadata** | LangChain `RecursiveCharacterTextSplitter` (paragraph → sentence → character fallback) enriched with study type, sample size, publication year | Does metadata enrichment improve evidence quality? |

Each strategy produces chunks in the same format (`pmid`, `title`, `chunk_index`, `text`, optional metadata) so ChromaDB ingestion works unchanged.

### 3.2 Retrieval Methods

| Method | Description |
|--------|-------------|
| **Naive** | Raw claim embedded → cosine similarity top-k from ChromaDB |
| **Hybrid** | BM25 (lexical) + dense retrieval (PubMedBERT) with reciprocal rank fusion |
| **Hybrid + reranking** | Hybrid retrieval → cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`) |
| **Advanced (agent-controlled)** | Agent-driven: claim decomposition into sub-queries + local corpus + live PubMed API per sub-claim |

### 3.3 Agent Architectures (Agent Pair — 2 Members)

| Architecture | Description | What It Tests |
|-------------|-------------|---------------|
| **Single-pass** | All retrieved evidence → single LLM call → verdict + explanation | Baseline — no orchestration overhead |
| **Strands sequential** | 4-agent pipeline: Claim Parser → Retrieval Agent → Evidence Reviewer → Verdict Agent (AWS Strands SDK) | Does structured multi-step reasoning improve quality? |
| **LangGraph graph-based** | Same 4-agent decomposition via LangGraph's node/edge/state orchestration | Does the framework matter? Graph-based vs event-driven |
| **Rerouting / adaptive** | Evidence Reviewer can loop back to Retrieval Agent if coverage is insufficient; confidence gate can short-circuit when evidence is strong | Can adaptive routing match full pipeline quality at lower cost? |

### 3.4 Models

| Model | Type | Access | Cost | What It Tests |
|-------|------|--------|------|---------------|
| **Claude Sonnet 4** | Proprietary frontier | Anthropic API / AWS Bedrock | $$$ | Quality ceiling |
| **GPT-4o-mini** | Proprietary efficient | OpenAI API | $$ | Does a cheaper proprietary model suffice? |
| **Claude Haiku** | Proprietary fast | Anthropic API / Bedrock | $ | Cost-quality frontier for Anthropic |
| **Llama 3.1 8B** | Open-source | Ollama (local) or Bedrock | Free / $ | Open-source baseline, zero API cost locally |
| **Llama 3.1 8B (fine-tuned)** | Open-source + fine-tuned | Local (LoRA) | Free | Can domain-specific fine-tuning close the gap? |
| **Llama 3.1 70B** | Open-source large | AWS Bedrock | $$ | Does scale compensate for no fine-tuning? |

### 3.5 Recommended Experiment Matrix

Running all combinations (4 × 3 × 4 × 6 = 288) is infeasible. We run a focused set of **10-12 experiments**, each over 120+ claims:

| # | Chunking | Retrieval | Agent | Model | What It Tests |
|---|----------|-----------|-------|-------|---------------|
| E1 | fixed | naive | single_pass | claude-sonnet | Baseline (current P1) |
| E2 | semantic | hybrid+rerank | single_pass | claude-sonnet | Best RAG, simple agent |
| E3 | section_aware | hybrid+rerank | single_pass | claude-sonnet | Alt chunking comparison |
| E4 | recursive | hybrid+rerank | single_pass | claude-sonnet | Metadata-enriched chunking |
| E5 | best_chunking | hybrid+rerank | strands_multi | claude-sonnet | Best RAG + Strands agents |
| E6 | best_chunking | hybrid+rerank | langgraph_multi | claude-sonnet | Strands vs LangGraph |
| E7 | best_chunking | hybrid+rerank | strands_rerouting | claude-sonnet | Rerouting value |
| E8 | best_chunking | hybrid+rerank | single_pass | gpt-4o-mini | Cheap proprietary model |
| E9 | best_chunking | hybrid+rerank | single_pass | llama-3.1-8b | Open-source baseline |
| E10 | best_chunking | hybrid+rerank | single_pass | llama-3.1-8b-ft | Fine-tuned open-source |
| E11 | best_chunking | hybrid+rerank | strands_multi | gpt-4o-mini | Cheap model + agents |
| E12 | fixed | naive | single_pass | gpt-4o-mini | Cheap everything |

**Note:** `best_chunking` is determined from E2-E4 results (whichever chunking strategy achieves highest Recall@5 in intrinsic retrieval evaluation). Experiments run sequentially: E1-E4 first (determine best chunking), then E5-E12.

**Estimated cost:** 12 experiments × 120 claims × ~$0.01-0.04/claim = **$15-$60 total**.

### 3.6 Key Comparisons

| Comparison | What It Reveals |
|-----------|----------------|
| E1 vs E2 | Does better chunking + retrieval improve results? |
| E2 vs E3 vs E4 | Which chunking strategy works best? |
| E2 vs E5 | Do agents add value when RAG is already good? |
| E5 vs E6 | Does the agent framework matter (Strands vs LangGraph)? |
| E5 vs E7 | Does rerouting improve on fixed sequential? |
| E2 vs E8 | Claude Sonnet vs GPT-4o-mini (same architecture) |
| E8 vs E9 | Proprietary vs open-source (same architecture) |
| E9 vs E10 | Does fine-tuning help? |
| E8 vs E11 | Do agents help more with a weaker model? |
| E1 vs E12 | Full quality vs full budget — how much do you lose? |

### 3.7 Detailed Pipeline Flows

#### Single-Pass Architecture

```
claim → [Chunking Strategy] → [Retrieval Method] → top-k passages
      → Single LLM call: "Given these passages, classify and explain"
      → Verdict + explanation
```

#### Multi-Agent Architecture (Strands / LangGraph)

```
claim → [Agent 1: Claim Parser]
          Decomposes into sub-claims with targeted queries
      → [Agent 2: Retrieval Agent]
          Per sub-claim: local corpus (chunked + hybrid search) + live PubMed API
      → [Agent 3: Evidence Reviewer]
          Flags contradictions, identifies gaps, assesses evidence quality
      → [Agent 4: Verdict Agent]
          Synthesises verdict addressing each sub-claim with citations
```

#### Rerouting Architecture

```
claim → [Agent 1: Claim Parser]
      → [Agent 2: Retrieval Agent]
      → [Agent 3: Evidence Reviewer]
            If coverage insufficient → loop back to Agent 2 with refined queries
            If coverage sufficient → proceed
      → [Confidence Gate]
            HIGH local evidence → skip to Agent 4 (short-circuit)
            LOW → continue full pipeline
      → [Agent 4: Verdict Agent]
```

---

## 4. Shared Components

All experiment configurations share the following to ensure fair comparison:

### 4.1 Input/Output Contract

**Input:** A health claim as text string.

**Output (identical schema across all configurations):**

```json
{
  "claim": "Intermittent fasting reverses Type 2 diabetes",
  "verdict": "OVERSTATED",
  "explanation": "While intermittent fasting shows promise for improving glycemic control...",
  "evidence": [
    {
      "source": "PMID:12345678",
      "passage": "Remission observed in 47% at 3 months but only 20% at 12 months...",
      "relevance_score": 0.92
    }
  ],
  "metadata": {
    "latency_seconds": 4.2,
    "total_tokens": 3850,
    "estimated_cost_usd": 0.012,
    "chunking_strategy": "semantic",
    "retrieval_method": "hybrid_reranked",
    "agent_architecture": "strands_multi",
    "model": "claude-sonnet-4"
  }
}
```

### 4.2 Verdict Taxonomy

| Verdict | Definition |
|---------|-----------|
| **Supported** | Well-supported by strong, consistent evidence |
| **Unsupported** | Contradicts available evidence or has no supporting evidence |
| **Overstated** | Contains a kernel of truth but exaggerates the evidence |
| **Insufficient Evidence** | Not enough quality evidence to determine |

### 4.3 Retrieval Corpus

- **Local corpus:** 150-200 health-related PubMed articles (expanded from POC's 36 vaccine articles), chunked using the strategy under test
- **Live PubMed API:** Available as a tool for agent-based architectures via Biopython E-utilities
- **Embedding model:** NeuML/pubmedbert-base-embeddings (local, no API key required)
- **Vector store:** ChromaDB (local, persistent), with separate collections per chunking strategy

### 4.4 LLM Providers

| Provider | Models | Access Method |
|----------|--------|---------------|
| Anthropic | Claude Sonnet 4, Claude Haiku | Anthropic API (single-pass), AWS Bedrock (agents) |
| OpenAI | GPT-4o-mini | OpenAI API |
| Open-source | Llama 3.1 8B/70B, Mistral 7B | Ollama (local) or AWS Bedrock |

`src/shared/llm.py` provides a unified `call_llm(prompt, system, model, provider)` interface that routes to the appropriate API.

---

## 5. Evaluation Framework

### 5.1 Two Tracks of Evaluation

The evaluation has two distinct tracks with different ground truth requirements:

| Track | What It Measures | Ground Truth Needed | Source |
|-------|-----------------|-------------------|--------|
| **Verdict accuracy** | Is the classification correct? | Claim + correct label | PUBHEALTH, ANTi-Vax datasets (labels map to our 4-verdict taxonomy) |
| **Explanation quality** | Is the explanation faithful, specific, complete, nuanced? | None — the LLM judge evaluates the pipeline's explanation against its own retrieved evidence | LLM-as-judge scores; calibrated against human annotation |

This is a critical distinction: **explanation quality does not require gold-standard reference explanations.** The LLM judge assesses properties of the output (Does it cite studies? Does it acknowledge limitations?) rather than comparing against a "correct" explanation. Human calibration validates the judge itself, not the explanations.

### 5.2 Quantitative Metrics (120+ claims)

| Metric | What It Measures | Method |
|--------|-----------------|--------|
| **Verdict Accuracy** | Correctness of classification | Macro-F1, per-class precision/recall against benchmark labels |
| **Explanation Quality** | Faithfulness, specificity, completeness, nuance | LLM-as-judge rubric scoring (1-5 per dimension) |
| **Grounding Rate** | % of factual statements traceable to retrieved evidence | Automated — LLM checks each statement against retrieved passages |
| **Retrieval Quality** | Are the right passages being retrieved? | Recall@5, Recall@10, MRR against gold PMIDs (intrinsic eval) |
| **Latency** | End-to-end time per claim | Wall-clock seconds |
| **Cost** | Token usage and API costs | Estimated USD per claim |

### 5.3 LLM-as-Judge

A frontier model scores each explanation on a structured rubric across four dimensions:

| Dimension | 1 (Low) | 5 (High) |
|-----------|---------|----------|
| **Faithfulness** | Makes claims not in retrieved evidence | All claims grounded in evidence |
| **Specificity** | Vague, no citations | Cites specific studies, sample sizes, dates |
| **Completeness** | Misses key aspects of the claim | Addresses all relevant dimensions |
| **Nuance** | Binary verdict with no caveats | Acknowledges limitations, populations, evidence strength |

**Calibration:** 2-3 team members independently score 30-50 pipeline-generated explanations on the same rubric. Measure inter-annotator agreement (target: Cohen's κ > 0.6) and LLM judge–human correlation to validate the judge. This validates the judge methodology — it does not require ground-truth explanations from external datasets.

**Reliability:** Run the judge 3 times per explanation, report variance to ensure scoring stability.

**Pairwise comparison:** For each claim, present two explanations from different configurations to the judge. Compute win rates across configuration pairs.

### 5.4 Statistical Significance

With 120+ claims, we can apply proper statistical tests:
- **McNemar's test** for paired verdict accuracy comparisons (same claims, two systems)
- **Bootstrap confidence intervals** for accuracy and quality score differences
- **Wilcoxon signed-rank test** for LLM-as-judge score comparisons
- Report p-values and effect sizes; the professor specifically requires statistical significance

### 5.5 Error Analysis

- Claims where simple configurations fail but advanced ones succeed (and vice versa)
- Claims where all configurations fail — corpus coverage gaps vs reasoning failures
- Cases where added complexity introduces errors (agent hallucination, rerouting loops)
- Breakdown by claim difficulty tier and topic domain

---

## 6. POC Results

A proof-of-concept comparing P1 (Naive RAG + Single-Pass) vs P6 (Advanced RAG + Multi-Agent) was built on 7 test claims. Additionally, P7 (confidence gating) was implemented as an adaptive optimisation of P6.

### 6.1 Verdict Accuracy

| Pipeline | Correct | Accuracy |
|----------|---------|----------|
| P1 (Naive + Single-Pass) | 4/7 | 57% |
| P6 (Advanced + Multi-Agent) | 4/7 | 57% |

Both achieve similar accuracy on this small test set. P1 struggles with nuanced claims (e.g., "Vitamin D prevents COVID" → UNSUPPORTED instead of OVERSTATED). P6 handles nuance better but occasionally over-qualifies mechanistic claims. The small sample size (n=7) means this comparison lacks statistical power — motivating the scale-up to 120+ claims.

### 6.2 Explanation Quality (LLM-as-Judge, 1-5 scale)

| Dimension | P1 | P6 | Delta |
|-----------|----|----|-------|
| Faithfulness | 4.86 | 5.00 | +0.14 |
| Specificity | 3.43 | 4.43 | +1.00 |
| Completeness | 3.43 | 4.71 | +1.29 |
| Nuance | 3.00 | 4.71 | +1.71 |
| **Overall** | **3.68** | **4.71** | **+1.03** |

P6 substantially outperforms P1 on explanation quality. The multi-agent pipeline produces richer explanations that acknowledge limitations and cite specific studies. Note: these judge scores are unvalidated against human ratings — human calibration is planned for the full study.

### 6.3 Cost & Latency

| Metric | P1 | P6 | Ratio |
|--------|----|----|-------|
| Total latency (7 claims) | 36s | 888s | 24.4× |
| Avg latency per claim | 5.2s | 126.8s | 24.4× |
| Total cost | $0.055 | $0.279 | 5.1× |

### 6.4 Key Takeaway

The POC validates the study premise: **there are real, measurable tradeoffs between pipeline complexity and output quality.** P6 produces significantly better explanations at substantially higher cost. The questions now are: (1) which specific components drive the improvement (chunking? agents? model?), (2) can cheaper models or open-source alternatives preserve quality, and (3) does this hold at scale on 120+ claims?

---

## 7. Datasets

### 7.1 Retrieval Corpus (150-200 articles)

| Source | Content | Size |
|--------|---------|------|
| PubMed Open Access (PMC) | Vaccine, nutrition, supplement, chronic disease research papers | ~150 articles |
| WHO Position Papers | Official WHO guidance on vaccines and public health | ~20 documents |
| CDC Fact Sheets | Public-facing health information | ~20 documents |

The corpus is expanded from the POC's 36 vaccine-only articles to cover the broader claim topics in the evaluation set. Articles are fetched via Biopython Entrez API and stored in `data/corpus.json`.

### 7.2 Evaluation Claims (120+ claims)

| Source | Size | Labels Available | How Used |
|--------|------|-----------------|----------|
| **PUBHEALTH** | ~11K claims | supported / refuted / mixed / unproven | Filter ~60-80 health-relevant claims; map labels to our 4-verdict taxonomy |
| **ANTi-Vax** | ~300 claims | misinformation / factual | Sample ~30-40 vaccine-specific claims; map to UNSUPPORTED / SUPPORTED |
| **Custom curated** | 20-30 claims | Team-annotated | Edge cases (nuanced, mechanistic, multi-hop) not well-represented in public datasets |

**Label mapping:**

| Source Label | Our Verdict |
|-------------|-------------|
| PUBHEALTH: supported | SUPPORTED |
| PUBHEALTH: refuted | UNSUPPORTED |
| PUBHEALTH: mixed | OVERSTATED |
| PUBHEALTH: unproven | INSUFFICIENT_EVIDENCE |
| ANTi-Vax: misinformation | UNSUPPORTED |
| ANTi-Vax: factual | SUPPORTED |

**Important:** These datasets provide **verdict labels only** — they do not include gold-standard explanations. This is by design: our explanation quality evaluation uses LLM-as-judge scoring, which assesses properties of the pipeline's own output (faithfulness to retrieved evidence, specificity, completeness, nuance) rather than comparing against a reference explanation. See Section 5.1 for details.

### 7.3 Fine-Tuning Data (Stretch Goal)

For the open-source fine-tuning experiment:
- **Training set:** ~80 claims with verdict labels + P6-generated explanations as silver-standard training signal
- **Test set:** ~40 held-out claims (never seen during fine-tuning)
- Fine-tune Llama 3.1 8B using LoRA via `unsloth` or `peft` on a free Colab T4 GPU
- The hypothesis: a fine-tuned 8B model can approach Claude Sonnet quality on this specific task at zero API cost

---

## 8. Demo Design

### Interactive Streamlit Application

1. Enter a health claim (free text) or select from curated examples
2. Choose configuration: chunking strategy, agent architecture, model
3. View results: verdict, explanation, evidence passages, latency, cost
4. Side-by-side comparison of any two configurations
5. Aggregate results dashboard with evaluation metrics

### Demo Script

| Step | Claim | Purpose |
|------|-------|---------|
| 1 | "Vaccines cause autism" | Easy claim — show all configs agree |
| 2 | "Vitamin D supplements prevent COVID infection" | Nuanced claim — show chunking/agent impact on OVERSTATED detection |
| 3 | Compare Claude Sonnet vs Llama 8B on same claim | Model comparison in real-time |
| 4 | Aggregate results dashboard | Full comparison across all metrics and configurations |

---

## 9. Workload Division (6 Members)

| Member | Role | Deliverables |
|--------|------|-------------|
| **Member 1** | Input Data | 120+ labeled claims from PUBHEALTH + ANTi-Vax + custom. Expanded corpus (150-200 articles). Label mapping and quality checks. Gold PMIDs for intrinsic retrieval eval. Fine-tuning data split. |
| **Members 2 & 3** | RAG / Chunking | 4 chunking strategies (fixed, semantic, section-aware, recursive). Hybrid retrieval (BM25 + dense fusion). Cross-encoder reranker. Intrinsic retrieval evaluation (Recall@k, MRR per strategy). Separate ChromaDB collections per strategy. |
| **Members 4 & 5** | Agent Architectures | 4 architectures (single-pass, Strands sequential, LangGraph graph, rerouting). Multi-provider `llm.py` (Anthropic, OpenAI, Ollama/Bedrock for open-source). Model integration for 3-5 models. Open-source fine-tuning experiment (LoRA on Llama 8B). |
| **Member 6** | Evaluation & Demo | Experiment runner with resumption. Statistical tests (McNemar, bootstrap CIs). `metrics.py` (F1, kappa, confusion matrix). `pairwise.py` (head-to-head comparison). Human annotation protocol + judge calibration. Visualisations (heatmaps, Pareto frontier, box plots). Streamlit demo. |

### Integration Points

- Member 1 delivers claims + corpus by week 2 so all others can run experiments
- Members 2-3 deliver chunked collections; Members 4-5 consume them via the same `vector_store.py` API
- Member 6 provides the experiment runner that all configurations plug into via the shared I/O contract
- All members contribute to human annotation of 30-50 explanations for judge calibration

### Execution Order

```
Week 1-2: Member 1 delivers data → Members 2-3 start chunking experiments
Week 2-3: Members 4-5 deliver multi-model llm.py → all can test model variations
Week 3-4: Members 2-3 determine best chunking (E2-E4) → feeds into E5-E12
Week 4-5: Member 6 runs full evaluation across all configurations
```

---

## 10. Timeline (5 Weeks Remaining)

| Week | Data (M1) | RAG (M2, M3) | Agents (M4, M5) | Eval (M6) |
|------|-----------|-------------|-----------------|-----------|
| **1** | Curate 120+ claims from PUBHEALTH/ANTi-Vax; expand corpus to 150+ articles | Implement semantic chunking; implement hybrid retrieval (BM25 + dense) | Refactor `llm.py` for multi-provider; add GPT-4o-mini + Ollama support | Build experiment runner with config + resumption |
| **2** | Finalise claims JSON with label mapping; build gold PMIDs for retrieval eval | Implement section-aware + recursive chunking; implement cross-encoder reranker | Implement LangGraph graph + 4 nodes; test open-source models | Implement `metrics.py` (F1, kappa, confusion matrix); design human annotation protocol |
| **3** | Prepare fine-tuning data split; support retrieval eval | Intrinsic retrieval eval (Recall@5, MRR); determine best chunking → publish results | Implement rerouting orchestrator; begin LoRA fine-tuning on Llama 8B | Implement `pairwise.py`; begin running E1-E4 with Member 2-3 |
| **4** | Support full experiment runs; fix data issues | Run experiments E5-E7 with eval team | Run experiments E8-E12 with eval team; evaluate fine-tuned model | Run all experiments; compute all metrics; human annotation (30-50 claims) |
| **5** | Report: data section | Report: RAG/chunking section | Report: agent/model section | Visualisations; statistical significance tests; report: evaluation section |

**What's already done (POC):**
- P1 (Naive RAG + Single-Pass) — working end-to-end
- P6 (Advanced RAG + Multi-Agent via Strands) — working end-to-end
- P7 confidence gating — implemented
- Evaluation framework — LLM-as-judge + grounding rate working
- Corpus — 36 articles, 84 chunks indexed in ChromaDB
- 7 test claims with POC comparison results

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM-as-judge unreliable | Calibrate against human annotations (30-50 claims); use grounding rate as backup metric; report judge variance |
| Chunking strategies show no difference | Valid finding — document with Recall@k numbers; indicates retrieval is not the bottleneck |
| Open-source models too poor for multi-agent | Use larger open-source model (70B via Bedrock) as fallback; document quality gap as a finding |
| Fine-tuning fails to improve base model | Report negative result with analysis; fine-tuning is a stretch goal, not core |
| PubMed API rate limits | Pre-cache responses; limit API-dependent experiments to agent architectures only |
| Agent overhead without quality gain | Document overhead-quality tradeoff as a key finding |
| 120 claims insufficient for statistical significance | Run power analysis; increase to 150+ if needed; use non-parametric tests |
| Cost overrun on experiments | Start with 30-claim pilot per experiment; estimate cost before full run; cut experiment matrix if needed |

---

## 12. Ethics & Safety

- All outputs include disclaimer: "Not medical advice. Consult a healthcare professional."
- System evaluates claims; it does not generate medical recommendations.
- Acknowledge potential selection bias in curated corpus as a limitation.
- Grounding rate metric directly measures hallucination risk.
- Open-source model outputs may have lower safety guardrails — monitor for harmful content in explanations.

---

## 13. Course Requirements Mapping

| Requirement | How Met |
|-------------|---------|
| Real-world problem | Health misinformation |
| Substantial GenAI component | 4 chunking strategies + 4 agent architectures + 5 models + fine-tuning + LLM-as-judge |
| At least one class component | RAG (4 chunking strategies, hybrid retrieval, reranking) and agentic orchestration (2 frameworks + rerouting) |
| More than a single assignment | 12 experiment configurations + evaluation framework + demo |
| Quantitative metrics | Macro-F1, Recall@k, grounding rate, LLM-as-judge scores, latency, cost |
| Qualitative analysis | Error analysis, pairwise comparisons, demo walkthrough |
| Compare 2+ alternatives | 12 configurations compared systematically across 3 axes |
| Reflective analysis | Where do gains plateau? When does complexity hurt? Can open-source + fine-tuning compete? |
| Risks & mitigations | 8 risks with specific mitigations |
| Reproducibility | Public benchmarks, shared evaluation harness, configurable experiments, README |
| Good teamwork | Shared I/O contract, configurable pipeline, clear role boundaries |
| Compare agent platforms (Prof Wynter) | LangGraph vs Strands Agent SDK |
| Advanced RAG (Prof Wynter) | Sophisticated chunking (4 strategies) + hybrid retrieval + reranking |
| Statistical significance (Prof Wynter) | 120+ claims, McNemar's test, bootstrap CIs |
| Different models (Prof Wynter) | Proprietary (Claude, GPT-4o-mini) + open-source (Llama, Mistral) + fine-tuning |

---

## 14. Summary

This study builds a configurable health claim verification pipeline and systematically varies chunking strategies, agent architectures, and LLM models — including open-source alternatives and domain-specific fine-tuning — across 120+ labeled health claims.

A proof-of-concept comparing the simplest (naive RAG + single-pass) vs most complex (advanced RAG + multi-agent) configurations has validated the study premise: multi-agent pipelines produce significantly richer explanations (+1.03 on a 5-point scale) at 24× the latency and 5× the cost. The full study will determine which specific components drive this improvement and whether cheaper alternatives can preserve quality.

Deliverables:
- Quantitative comparison across 12 experiment configurations with statistical significance testing
- Calibrated evaluation methodology combining LLM-as-judge with human validation and automated grounding rate
- Comparison of 4 chunking strategies with intrinsic retrieval evaluation
- Comparison of proprietary vs open-source models, including fine-tuning experiment
- Analysis of 4 agent architectures across 2 orchestration frameworks
- Interactive Streamlit demo visualising tradeoffs in real-time
- Findings on when simple approaches suffice vs when advanced ones are warranted
