# Comparing RAG and Agent Architectures for Health Claim Verification (v4)

## A Systematic Study of Retrieval and Orchestration Strategies

---

## 1. Research Question

> **How do increasingly sophisticated RAG strategies and agent architectures compare for health claim verification — and where do the gains justify the added complexity?**

We compare **3 RAG tiers × 2 agent approaches** (6 pipeline variants) plus an **adaptive gated variant** (7 pipelines total) on the same task, measuring verdict accuracy, explanation quality, retrieval precision, latency, and cost.

### Two Axes of Complexity

**RAG axis:** Does investing in better retrieval (query rewriting, hybrid search, re-ranking, claim decomposition) improve the quality of evidence that reaches the LLM?

**Agent axis:** Does structured task decomposition — breaking a complex task into specialised sequential steps, each with a focused prompt — produce better outcomes than a single monolithic prompt given the same retrieved evidence? Each agent uses the same underlying LLM; the difference is how the task is structured, not model capability.

**Interaction effect:** Does the benefit of agent orchestration depend on retrieval quality? The 3×2 matrix allows us to isolate and measure this interaction.

**Efficiency axis (P7):** Can we get multi-agent quality at single-pass speed by adaptively short-circuiting expensive agents when local evidence is already sufficient? This tests whether confidence-based gating can eliminate unnecessary work without sacrificing verdict quality.

---

## 2. Problem Statement

Health misinformation is widespread — social media amplifies unverified claims, news articles misrepresent findings, and AI chatbots may hallucinate citations. Patients make treatment decisions based on inaccurate information.

### Why This Task Suits a Comparative Study

1. **Ground truth exists** — benchmark datasets (ANTi-Vax, PUBHEALTH) provide labeled claims
2. **Claims vary in difficulty** — from simple factual to nuanced claims requiring multi-step reasoning
3. **Evidence is publicly accessible** — PubMed, WHO guidelines, open-access medical literature
4. **Explanation quality matters** — a correct verdict with a poor explanation is insufficient

### Scope

Vaccine-related claims, due to high prevalence of both misinformation and quality counter-evidence. May be adjusted based on dataset availability.

---

## 3. Study Design: 3 × 2 Comparison Matrix + Adaptive Gating

### 3.1 RAG Tiers

| Tier | Key Techniques |
|------|----------------|
| **Naive RAG** | Chunk → embed (`text-embedding-3-small`) → cosine similarity top-k → stuff into prompt |
| **Intermediate RAG** | Query rewriting/expansion + hybrid search (BM25 + dense) + cross-encoder re-ranking |
| **Advanced RAG** | Claim decomposition into sub-queries + multi-source retrieval (PubMed API + local corpus) + evidence hierarchy ranking |

### 3.2 Agent Approaches

| Approach | Description |
|----------|-------------|
| **Single-Pass** | Retrieved evidence passed to a single LLM call that generates verdict + explanation |
| **Multi-Agent** | Claim Parser → Retrieval Agent → Verdict Agent, orchestrated via agent framework |

### 3.3 The 7 Pipeline Variants

| Pipeline | RAG Tier | Agent Approach | What It Tests |
|----------|----------|----------------|--------------|
| P1 | Naive | Single-Pass | Baseline |
| P2 | Naive | Multi-Agent | Does orchestration help when retrieval is weak? |
| P3 | Intermediate | Single-Pass | How much does better retrieval alone improve results? |
| P4 | Intermediate | Multi-Agent | Do agents add value on top of good retrieval? |
| P5 | Advanced | Single-Pass | Can sophisticated retrieval compensate for single-pass reasoning? |
| P6 | Advanced | Multi-Agent | Does full complexity justify its cost? |
| P7 | Advanced | Adaptive Multi-Agent | Can confidence gating match P6 quality at lower cost? |

### 3.4 Detailed Pipeline Flows

#### P1 — Naive RAG + Single-Pass

```
User claim → Embed claim (text-embedding-3-small)
           → Cosine similarity search against pre-chunked corpus → top-k passages
           → Single LLM prompt: "Given these passages, classify this claim and explain"
           → Verdict + explanation
```

---

#### P2 — Naive RAG + Multi-Agent

```
User claim → [Agent 1: Claim Parser]
               Extracts key entities, reformulates claim for retrieval
           → Embed refined query (text-embedding-3-small)
           → Cosine similarity search, top-k (same naive retrieval as P1)
           → [Agent 2: Retrieval Agent]
               Reviews retrieved passages, can request second retrieval with refined query
           → [Agent 3: Verdict Agent]
               Generates verdict + explanation from curated evidence
```

---

#### P3 — Intermediate RAG + Single-Pass

```
User claim → LLM rewrites claim into optimised search query
               (expand with medical synonyms e.g. "cholecalciferol", "SARS-CoV-2")
           → Hybrid search: BM25 + dense retrieval (reciprocal rank fusion)
           → Cross-encoder re-ranking of top candidates → top-k passages
           → Single LLM prompt with re-ranked evidence
           → Verdict + explanation
```

---

#### P4 — Intermediate RAG + Multi-Agent

```
User claim → [Agent 1: Claim Parser]
               Extracts entities, reformulates query, identifies implicit sub-claims
           → Hybrid search (BM25 + dense) on agent-refined query
           → Cross-encoder re-ranking
           → [Agent 2: Retrieval Agent]
               Reviews evidence quality, can trigger second retrieval if coverage insufficient
           → [Agent 3: Verdict Agent]
               Generates verdict + explanation from curated, re-ranked evidence
```

---

#### P5 — Advanced RAG + Single-Pass

```
User claim → LLM decomposes claim into verifiable sub-claims
               e.g. ① "Does Vitamin D affect COVID outcomes?"
                    ② "Is the effect prevention vs severity reduction?"
                    ③ "Is this specific to deficient populations?"
           → Per sub-claim: PubMed API + local corpus, hybrid search, re-ranking
           → Evidence hierarchy ranking (systematic reviews > RCTs > observational)
           → Deduplicate across sub-queries
           → Single LLM prompt with decomposed, hierarchy-ranked evidence
           → Verdict + explanation
```

---

#### P6 — Advanced RAG + Multi-Agent

```
User claim → [Agent 1: Claim Parser]
               Decomposes into sub-claims, generates targeted queries per sub-claim
           → [Agent 2: Retrieval Agent]
               Per sub-claim: PubMed API + local corpus, hybrid search, re-ranking,
               evidence hierarchy ranking. Can iteratively refine queries.
           → [Agent 3: Evidence Reviewer]
               Cross-checks evidence across sub-claims, flags contradictions, identifies gaps
           → [Agent 4: Verdict Agent]
               Synthesises across sub-claims into nuanced verdict with per-sub-claim evidence
```

---

#### P7 — Advanced RAG + Adaptive Multi-Agent (Confidence Gating)

P7 is a latency-optimised variant of P6. It runs the Claim Parser first, then searches the local ChromaDB corpus directly (fast, ~100ms) to score how well the local evidence covers each sub-claim. If coverage is high enough, it skips the Retrieval Agent and Evidence Reviewer entirely — saving ~40-50% latency per claim.

```
User claim → [Agent 1: Claim Parser]
               Decomposes into sub-claims with targeted queries
           → [Local ChromaDB search per sub-claim — no LLM call]
           → [Confidence Gate: score local evidence quality]
               Score = 50% avg sub-claim quality + 50% coverage ratio
               HIGH (score ≥ 0.7 AND coverage ≥ 75%):
                   → Format local evidence + synthetic review
                   → [Agent 4: Verdict Agent] → output     (~50s, 2 agent calls)
               LOW:
                   → [Full P6 pipeline: Agents 2-4] → output   (~90s, 4 agent calls)
```

**Confidence scoring per sub-claim:**
- Search ChromaDB top-5 for each sub-claim query
- Count hits with L2 distance < 0.45 ("relevant" threshold)
- Compute average distance of top-3 hits (quality signal)
- Coverage ratio = fraction of sub-claims with ≥ 1 relevant hit

**What P7 tests:** Whether adaptive orchestration — running fewer agents when evidence is already strong — can preserve P6-level verdict quality while significantly reducing latency and cost. This is relevant for production deployment where per-claim latency matters.

### 3.5 Key Cross-Pipeline Comparisons

| Comparison | What It Reveals |
|-----------|----------------|
| P1 vs P2 | Value of agents when retrieval is weak |
| P3 vs P4 | Value of agents when retrieval is good |
| P5 vs P6 | Value of agents when retrieval is excellent |
| P1 vs P3 vs P5 | Impact of RAG sophistication (single-pass held constant) |
| P2 vs P4 vs P6 | Impact of RAG sophistication (multi-agent held constant) |
| P2 vs P3 | Agents vs better retrieval — which investment yields more? |
| P3 vs P6 | Intermediate single-pass vs full complexity — diminishing returns? |
| P6 vs P7 | Does adaptive gating preserve quality while cutting latency? |
| P5 vs P7 | Single-pass vs adaptive multi-agent — what's the best quality/cost tradeoff? |

### 3.6 Agent Framework Comparison

The multi-agent pipelines (P2, P4, P6) will be implemented on **two agent frameworks** — **LangGraph** and **AWS Strands Agent SDK** — to compare two different orchestration paradigms: graph-based (explicit nodes, edges, state passing) vs event-driven (tool-use loops with autonomous agent control).

---

## 4. Shared Components

All 7 pipelines share the following to ensure fair comparison:

### 4.1 Input/Output Contract

**Input:** A health claim as text string.

**Output (identical schema across all pipelines):**

```json
{
  "claim": "Intermittent fasting reverses Type 2 diabetes",
  "verdict": "OVERSTATED",
  "explanation": "While intermittent fasting shows promise for improving glycemic control...",
  "evidence": [
    {
      "source": "Liu et al., Lancet 2023",
      "passage": "Remission observed in 47% at 3 months but only 20% at 12 months...",
      "relevance_score": 0.92
    }
  ],
  "metadata": {
    "latency_seconds": 4.2,
    "total_tokens": 3850,
    "estimated_cost_usd": 0.012,
    "retrieval_method": "advanced_rag",
    "agent_type": "multi_agent"
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

- **Local corpus:** Curated vaccine-related articles from PubMed Open Access, WHO guidelines, CDC fact sheets (pre-downloaded and chunked)
- **API-based retrieval (Advanced RAG only):** Live PubMed E-utilities search, supplementing the local corpus

### 4.4 LLM

All pipelines use **Claude Sonnet 4** as the base LLM — via the Anthropic API for single-pass pipelines and via AWS Bedrock for multi-agent pipelines (Strands Agent SDK). Using the same underlying model isolates the effect of RAG and agent architecture from model capability.

---

## 5. Evaluation Framework

### 5.1 Metrics

| Metric | What It Measures | Method |
|--------|-----------------|--------|
| **Verdict Accuracy** | Correctness of classification | Macro-F1 against benchmark labels |
| **Explanation Quality** | Faithfulness, specificity, completeness, nuance | LLM-as-judge rubric scoring (1-5 per dimension) |
| **Grounding Rate** | % of factual statements in explanation traceable to retrieved evidence | Automated — LLM checks each statement against retrieved passages |
| **Latency** | End-to-end time per claim | Wall-clock seconds |
| **Cost** | Token usage and API costs | Estimated USD per claim |

### 5.2 LLM-as-Judge

A frontier model scores each explanation on a structured rubric across four dimensions:

| Dimension | 1 (Low) | 5 (High) |
|-----------|---------|----------|
| **Faithfulness** | Makes claims not in retrieved evidence | All claims grounded in evidence |
| **Specificity** | Vague, no citations | Cites specific studies, sample sizes, dates |
| **Completeness** | Misses key aspects of the claim | Addresses all relevant dimensions |
| **Nuance** | Binary verdict with no caveats | Acknowledges limitations, populations, evidence strength |

**Calibration:** 2-3 team members independently score 30-50 explanations on the same rubric. Measure inter-annotator agreement (target: Cohen's κ > 0.6) and LLM judge–human correlation to validate the judge.

**Pairwise comparison:** For each claim, present two explanations from different pipelines to the judge. Compute win rates across pipeline pairs.

### 5.3 Grounding Rate

Automated complement to the LLM judge. For each factual statement in an explanation, check whether it traces to a specific passage in the retrieved evidence. Produces a percentage per pipeline — quantifiable without subjective scoring.

### 5.4 Error Analysis

- Claims where simple pipelines fail but advanced ones succeed (and vice versa)
- Claims where all pipelines fail
- Cases where added complexity introduces errors

---

## 6. POC Results

A proof-of-concept was built with P1, P6, and P7 on 7 test claims (vaccines, mRNA, Vitamin D, fasting, flu). Key findings:

### 6.1 Verdict Accuracy

| Pipeline | Correct | Accuracy |
|----------|---------|----------|
| P1 (Naive + Single-Pass) | 4/7 | 57% |
| P6 (Advanced + Multi-Agent) | 4/7 | 57% |

Both achieve similar accuracy on this small test set. P1 struggles with nuanced claims (e.g., "Vitamin D prevents COVID" → UNSUPPORTED instead of OVERSTATED). P6 handles nuance better but occasionally over-qualifies mechanistic claims.

### 6.2 Explanation Quality (LLM-as-Judge, 1-5 scale)

| Dimension | P1 | P6 | Delta |
|-----------|----|----|-------|
| Faithfulness | 4.86 | 5.00 | +0.14 |
| Specificity | 3.43 | 4.43 | +1.00 |
| Completeness | 3.43 | 4.71 | +1.29 |
| Nuance | 3.00 | 4.71 | +1.71 |
| **Overall** | **3.68** | **4.71** | **+1.03** |

P6 substantially outperforms P1 on explanation quality, especially on nuance (+1.71) and completeness (+1.29). The multi-agent pipeline produces richer, more detailed explanations that acknowledge limitations and cite specific studies.

### 6.3 Cost & Latency

| Metric | P1 | P6 | Ratio |
|--------|----|----|-------|
| Total latency (7 claims) | 36s | 888s | 24.4× |
| Avg latency per claim | 5.2s | 126.8s | 24.4× |
| Total cost | $0.055 | $0.279 | 5.1× |

### 6.4 Key Takeaway

The POC validates the study premise: **there are real, measurable tradeoffs between pipeline complexity and output quality.** P6 produces significantly better explanations at substantially higher cost. The full 3×2+1 build is worth pursuing to map the full tradeoff curve and determine where the inflection points lie.

P7 (confidence gating) is expected to close the latency gap for claims where the local corpus already has strong coverage — roughly 40-50% savings on well-covered claims with no accuracy loss.

---

## 7. Demo Design

### Interactive Streamlit Application

1. Enter a health claim (free text) or select from curated examples
2. Choose which pipelines to run (any combination, or all at once)
3. View side-by-side results: verdict, explanation, evidence, latency, cost
4. Comparison panel showing agreement/disagreement and evaluation scores

### Demo Script

| Step | Claim | Purpose |
|------|-------|---------|
| 1 | "Vaccines cause autism" | Easy claim — all pipelines agree |
| 2 | "Vitamin D supplements prevent COVID infection" | Nuanced claim — tests overstated detection |
| 3 | "mRNA vaccines alter your DNA" | Tests mechanistic reasoning in explanation |
| 4 | Aggregate results dashboard | Full comparison across all metrics |

---

## 8. Datasets

### Retrieval Corpus

| Source | Content |
|--------|---------|
| PubMed Open Access (PMC) | Vaccine-related research papers |
| WHO Vaccine Position Papers | Official WHO guidance |
| CDC Vaccine Information Statements | Public-facing vaccine facts |

### Evaluation

| Dataset | Size | Labels |
|---------|------|--------|
| ANTi-Vax | ~300 claims | Misinformation / factual |
| PUBHEALTH | ~11K claims | Supported / refuted / mixed / unproven |
| Custom Curated | 50-100 claims | Team-annotated for qualitative analysis + judge calibration |

---

## 9. Workload Division (6 Members)

| Member | Responsibility | Deliverables |
|--------|---------------|-------------|
| **Member 1** | Data & preprocessing | Retrieval corpus, evaluation datasets, preprocessing pipeline |
| **Member 2** | Naive RAG pipelines (P1, P2) | Two working pipelines with shared I/O contract |
| **Member 3** | Intermediate RAG pipelines (P3, P4) | Two pipelines with query rewriting, hybrid search, re-ranking |
| **Member 4** | Advanced RAG pipelines (P5, P6, P7) | Three pipelines with decomposition, multi-source retrieval, hierarchy ranking, and adaptive confidence gating |
| **Member 5** | Agent frameworks | LangGraph vs Strands Agent SDK implementation, shared agent templates for Members 2-4 |
| **Member 6** | Evaluation & demo | Evaluation harness, LLM-as-judge, Streamlit app, results analysis |

### Integration Points

- Members 2-4 consume the corpus from Member 1 and agent templates from Member 5
- Member 6 provides the evaluation harness that all pipelines plug into via the shared I/O contract
- All members contribute to human annotation of the calibration subset

---

## 10. Timeline (8 Weeks)

| Week | Phase | Deliverables |
|------|-------|-------------|
| **1-2** | Foundation (done) | Corpus curated and chunked (36 articles, 84 chunks). I/O contract finalised. P1 and P6 working end-to-end. POC comparison complete on 7 test claims. P7 (confidence gating) implemented. |
| **3-4** | Core Pipelines | Remaining pipelines (P2-P5) built. LangGraph agent templates. Evaluation harness operational. LLM-as-judge calibrated. Initial results on ~50 claims. |
| **5-6** | Evaluation | Full evaluation run across all 7 pipelines. Human annotation. Grounding rate computation. Pairwise comparisons. P7 threshold sensitivity analysis. Error analysis begun. |
| **7** | Analysis & Demo | Results analysis and visualisation. Streamlit demo functional. Agent framework comparison written up. |
| **8** | Report & Polish | Written report. Demo rehearsed. Code cleanup. Reproducibility check. |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM-as-judge unreliable | Calibrate against human annotations; use grounding rate as backup metric |
| Advanced RAG shows no improvement | Null result is valid; analyse why and for which claim types |
| PubMed API rate limits | Pre-cache responses; smaller claim subset for API-dependent pipelines |
| Agent overhead without accuracy gain | Document overhead-accuracy tradeoff as a finding |
| Unfair comparison across pipelines | Shared I/O contract, same LLM, same corpus, same evaluation harness |
| P7 gate threshold too aggressive/conservative | Threshold is configurable; tune on dev set before evaluation; report sensitivity analysis |
| Scope too large for 8 weeks | P1, P6, P7 already built as POC; intermediate pipelines are additive |

---

## 12. Ethics & Safety

- All outputs include disclaimer: "Not medical advice. Consult a healthcare professional."
- System evaluates claims; it does not generate medical recommendations.
- Acknowledge potential selection bias in curated corpus as a limitation.
- Grounding rate metric directly measures hallucination risk.

---

## 13. Course Requirements Mapping

| Requirement | How Met |
|-------------|---------|
| Real-world problem | Health misinformation |
| Substantial GenAI component | 3 RAG strategies + 2 agent architectures + adaptive gating + LLM-as-judge |
| At least one class component | Both RAG (3 tiers) and agentic orchestration (2 frameworks) |
| More than a single assignment | 7 pipeline variants + evaluation framework + demo |
| Quantitative metrics | Macro-F1, grounding rate, LLM-as-judge scores, latency, cost |
| Qualitative analysis | Error analysis, pairwise comparisons, demo walkthrough |
| Compare 2+ alternatives | 7 pipeline variants compared systematically |
| Reflective analysis | Where do gains plateau? When does complexity hurt? Can adaptive gating recover cost? |
| Risks & mitigations | 7 risks with specific mitigations |
| Reproducibility | Public benchmarks, shared evaluation harness, fixed seeds, README |
| Good teamwork | Shared I/O contract, corpus, and evaluation harness ensure integration |
| Compare agent platforms (Prof Wynter) | LangGraph vs Strands Agent SDK |
| Advanced RAG (Prof Wynter) | 3 RAG tiers from naive to advanced |

---

## 14. Summary

This study compares 7 pipeline variants (3 RAG tiers × 2 agent approaches + 1 adaptive gated variant) on health claim verification to determine at what point additional retrieval and orchestration complexity stops improving results — and whether adaptive orchestration can recover the cost without sacrificing quality.

A proof-of-concept comparing P1 (simplest) vs P6 (most complex) has already validated the study premise: P6 produces significantly richer explanations (+1.03 on a 5-point quality scale) at 24× the latency and 5× the cost. P7 (confidence gating) targets closing this efficiency gap for claims with strong local corpus coverage.

Deliverables:
- Quantitative comparison across accuracy, explanation quality, retrieval precision, latency, and cost
- Calibrated evaluation methodology combining LLM-as-judge with human validation and automated grounding rate
- Analysis of adaptive orchestration (P7) as a production-viable optimisation strategy
- Interactive demo visualising tradeoffs in real-time
- Findings on when simple RAG suffices vs when advanced approaches are warranted
