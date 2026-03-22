# Automated Health Claim Fact-Checking: RAG vs Multi-Agent Approaches
## CS614 Generative AI — Group Project Presentation (15 min)

---

## Slide 1: Title Slide
**Automated Health Claim Fact-Checking: RAG vs Multi-Agent Approaches**
- CS614 Generative AI — Group Project
- Team members: [names]
- Date: March 2026

---

## Slide 2: The Problem — Health Misinformation at Scale
- WHO declared health misinformation an "infodemic" alongside COVID-19
- Claims like "Vitamin D prevents COVID" or "Vaccines cause autism" spread faster than fact-checkers can respond
- Manual fact-checking: a single claim takes a trained reviewer 15-30 minutes to verify against scientific literature
- **Our goal**: Build and compare automated pipelines that verify health claims against scientific evidence, and evaluate the tradeoff between simple RAG and complex agentic architectures

**Speaker notes**: Frame the real-world urgency. Mention that platforms like Meta/X rely on third-party fact-checkers who are overwhelmed. Our system targets the retrieval + reasoning bottleneck.

---

## Slide 3: Dataset — SciFact Benchmark
**Why SciFact (Allen AI, EMNLP 2020)?**
- Gold-standard benchmark for scientific claim verification — expert-annotated by scientists
- Allows rigorous, reproducible evaluation (unlike scraping social media claims)

**Corpus**: 5,183 scientific abstracts
- Average abstract length: 1,404 characters
- 19% have structured sections (Background, Methods, Results, Conclusion)
- Covers biomedical topics: genetics, immunology, oncology, epidemiology

**Test Claims**: 300 claims, perfectly balanced:
- 100 SUPPORTED — evidence directly confirms the claim
- 100 UNSUPPORTED — evidence directly contradicts the claim
- 100 INSUFFICIENT_EVIDENCE — corpus doesn't address the claim

**Easy vs Hard split** (important for later analysis):
- 200 "easy" claims — gold evidence document exists in the corpus
- 100 "hard" claims — no gold evidence document, system must reason from tangential evidence

**Speaker notes**: The easy/hard split is key — it explains why agents underperform on this dataset but may excel in real-world scenarios where evidence isn't neatly pre-indexed.

---

## Slide 4: System Architecture — Three Experimental Axes

**Our system is modular along 3 axes, allowing controlled experiments:**

```
Health Claim
    ↓
[Axis 1: CHUNKING] — How we split 5,183 abstracts into searchable passages
    ↓
[ChromaDB Vector Index] — all-MiniLM-L6-v2 embeddings (384-dim)
    ↓
[Axis 2: RETRIEVAL] — How we find the most relevant evidence passages
    ↓
[Axis 3: REASONING] — How the LLM interprets evidence and produces a verdict
    ↓
Verdict: SUPPORTED / UNSUPPORTED / INSUFFICIENT_EVIDENCE
```

**Tech stack**:
- Embedding: Sentence-Transformers (all-MiniLM-L6-v2), ChromaDB persistent vector store
- LLM: Claude Sonnet 4 (via Anthropic API and AWS Bedrock)
- Agent frameworks: Strands SDK (AWS), LangGraph (LangChain)
- External search: Semantic Scholar API (200M+ papers)
- App: Streamlit demo with 3 tabs

**Speaker notes**: Emphasize the controlled variable design — we change one axis at a time to isolate the impact of each component. This is what makes our results scientifically meaningful rather than anecdotal.

---

## Slide 5: Axis 1 — Chunking Strategies (Implementation Detail)

**The problem**: Abstracts average 1,404 chars. We need to split them into passages small enough for precise retrieval but large enough to preserve context.

**Four strategies implemented** (`src/chunking/`):

| Strategy | How it works | Chunk count | Avg chunk size |
|----------|-------------|-------------|---------------|
| **Fixed** | Split every 500 chars with 50-char overlap. Min 10 words filter to prevent fragments. | ~14,000 | ~500 chars |
| **Section-Aware** | Split on abstract section boundaries (Background/Methods/Results/Conclusion). Falls back to fixed for unstructured abstracts (81% of corpus). | ~13,000 | varies |
| **Semantic** | Use sentence-transformer embeddings to detect topic shifts between sentences. Split when cosine similarity drops below threshold. | ~11,000 | varies |
| **Recursive** | LangChain RecursiveCharacterTextSplitter — tries paragraph → sentence → word boundaries, chunk_size=800 chars. | 13,580 | ~800 chars |

**Key implementation decisions**:
- Fixed: Added `MIN_CHUNK_WORDS=10` after finding tiny 2-3 word fragments at abstract boundaries
- Section-aware: Parsed `Label` attributes from PubMed XML — only 19% of abstracts had them, so 81% fell back to fixed chunking
- Recursive: Increased chunk_size from 500→800 after analysis showed 500 produced too-small chunks that lost inter-sentence context

**Speaker notes**: Walk through why recursive won — it adapts to content structure (splits at paragraphs when possible, sentences when needed) while maintaining consistent size. Section-aware was limited by the corpus having mostly unstructured abstracts.

---

## Slide 6: Axis 1 Results — Chunking Comparison

| Strategy | Accuracy | SUPPORTED | UNSUPPORTED | INSUFF_EVIDENCE |
|----------|----------|-----------|-------------|-----------------|
| Fixed | 79.7% | 80% | 84% | 75% |
| Section-aware | 77.3% | — | — | — |
| Semantic | 76.3% | — | — | — |
| **Recursive** | **82.0%** | **88%** | **83%** | **75%** |

**Accuracy ranking**: Recursive (82.0%) > Fixed (79.7%) > Section-aware (77.3%) > Semantic (76.3%)

**Analysis**:
- Recursive's 800-char chunks capture enough context for the LLM to reason about cause-effect relationships
- Semantic chunking underperformed because scientific abstracts already have clear sentence-level structure — the embedding-based boundary detection added noise rather than insight
- Section-aware suffered from 81% fallback rate — most abstracts weren't structured
- Fixed is a strong baseline — simplicity shouldn't be underestimated

**Speaker notes**: The 82% vs 76% gap (5.7 percentage points) is statistically significant across 300 claims. Recursive chunking is our winner and becomes the fixed choice for all subsequent experiments.

---

## Slide 7: Axis 2 — Retrieval Methods (Implementation Detail)

**Three retrieval approaches** (`src/retrieval/`):

**1. Naive (Dense Embedding Search)**
- Query the claim directly against ChromaDB using cosine similarity
- Return top-5 most similar passages
- Simple, fast (~100ms per query)

**2. Hybrid (Dense + Sparse)**
- Combine embedding similarity with BM25 keyword matching (TF-IDF based)
- BM25 handles exact term matches (e.g., specific gene names, drug names)
- Reciprocal Rank Fusion to merge the two ranked lists
- Return top-10, truncate to top-5

**3. Hybrid + Cross-Encoder Reranking**
- Same as hybrid, but pass top-10 candidates through a cross-encoder model
- Cross-encoder scores each (query, passage) pair jointly — more accurate but slower
- Return top-5 after reranking

**Speaker notes**: The intuition for hybrid was that BM25 would catch exact medical terms that embedding similarity might miss. Reranking was meant to fix hybrid's ranking quality. In practice, neither helped on this corpus.

---

## Slide 8: Axis 2 Results — Retrieval Comparison

All using recursive chunking (best from Axis 1):

| Method | Accuracy | Avg Latency | Cost (300 claims) |
|--------|----------|-------------|-------------------|
| **Naive** | **82.0%** | **8.4s** | **$3.46** |
| Hybrid | 79.0% | 6.1s | $3.60 |
| Hybrid + Rerank | 81.3% | 6.0s | $3.72 |

**Why did naive win?**
- SciFact corpus is topically focused (all biomedical). Dense embeddings capture semantic similarity well in this domain.
- BM25 introduced false positives: keyword matches on common medical terms (e.g., "patients", "treatment") that weren't semantically relevant to the specific claim.
- Reranking recovered 2.3 percentage points (79.0% → 81.3%) by filtering BM25 noise, but didn't surpass pure embedding search.

**Insight**: Hybrid retrieval is designed for broad, heterogeneous corpora. For a focused scientific corpus where the embedding model is well-suited, simpler is better.

**Speaker notes**: Note that latency is slightly lower for hybrid/reranked — this is because the LLM receives different evidence passages, not because retrieval itself is faster. The LLM call dominates latency.

---

## Slide 9: Axis 3 — Agent Architectures (Implementation Detail)

**Architecture A: Single-Pass RAG (E1-E6)**
```
Claim → Retrieve top-5 passages from ChromaDB → Single LLM call → Verdict
```
- 1 LLM call per claim
- System prompt: "You are a scientific claim fact-checker. Given evidence passages, determine whether each claim is supported, contradicted, or lacks sufficient evidence."
- LLM sees all evidence at once and makes a direct judgment

**Architecture B: Multi-Agent Pipeline (E7 Strands / E8 LangGraph)**
```
Claim → [Agent 1: Claim Parser]
           Decomposes into 2-4 verifiable sub-claims with targeted search queries
       → [Agent 2: Retrieval Agent]
           Searches ChromaDB separately for each sub-claim query
       → [Agent 3: Evidence Reviewer]
           Flags contradictions, evidence gaps, quality issues
           Rates overall evidence strength: STRONG / MODERATE / WEAK / MIXED
       → [Agent 4: Verdict Agent]
           Synthesizes review into final verdict with cited evidence
```
- 4 LLM calls per claim (minimum)
- Each agent has a specialized system prompt and structured JSON output schema

**Speaker notes**: Walk through a concrete example. For the claim "Vitamin D supplementation prevents COVID-19 infection", the Claim Parser might produce: (1) "Vitamin D has immunomodulatory effects on respiratory infections", (2) "Supplementation reduces COVID-19 incidence in clinical trials", (3) "The effect is dose-dependent". Each gets its own retrieval query.

---

## Slide 10: Strands vs LangGraph — Framework Comparison

**E7: Strands SDK (AWS)**
- Each agent is a Strands `Agent()` with tool-calling capability
- Retrieval Agent has a `@tool` decorated function `search_local_corpus()` — the LLM decides when/how to call it
- Uses AWS Bedrock as the LLM backend with built-in retry and exponential backoff
- Structured output via Pydantic models (`RetrievalOutput`, `ReviewedEvidence`, etc.)

**E8: LangGraph (LangChain)**
- Pipeline defined as a `StateGraph` DAG with 4 nodes
- Each node is a pure Python function that reads shared state and returns updates
- Retrieval node calls ChromaDB directly (no LLM tool-calling for retrieval)
- Uses Anthropic API via our `call_llm()` utility for the 3 LLM nodes
- State flows through a typed `PipelineState` dictionary

**Key architectural difference**:
- Strands: LLM has agency over retrieval (decides what to search, how many times)
- LangGraph: Retrieval is deterministic (always searches each sub-claim query once)

**Infrastructure difference**:
- Strands/Bedrock: Built-in retry with exponential backoff → 0 connection errors in E7
- LangGraph/Anthropic API: No built-in retry → 51 BrokenPipeError in E8 (fixed with retry wrapper)

**Speaker notes**: The BrokenPipe errors are NOT a LangGraph limitation — they're an API client issue we fixed. The fair comparison is on accuracy of successfully completed claims.

---

## Slide 11: Axis 3 Results — RAG vs Agents (300 Claims)

| Architecture | Accuracy | SUP | UNS | INS | Avg Latency | LLM Calls |
|-------------|----------|-----|-----|-----|-------------|-----------|
| **Single-pass RAG (E4)** | **82.0%** | **88%** | **83%** | **75%** | **8.4s** | **1** |
| Strands Agents (E7) | 65.3% | 49% | 84% | 63% | 68s median | 4-5 |
| LangGraph Agents (E8) | 66.0% | 49% | 83% | 66% | 32s median | 3 |
| Llama 3.1 8B RAG (E11) | 62.7% | 61% | 68% | 59% | 98s median | 1 |

**Diagram**: `results/accuracy_comparison.png` — bar chart of all experiments

**Key observations**:
- RAG outperforms both agent frameworks by 16-17 percentage points on closed corpus
- E7 and E8 converge to similar accuracy (65-66%) — the large gap in preliminary results disappeared after E8 bug fixes
- Both agents share the same weakness: SUPPORTED recall drops from 88% (RAG) to 49% (agents)
- Both agents are strong on UNSUPPORTED (83-84%) — close to RAG's 83%
- Llama 3.1 8B (E11) at 62.7% shows model quality matters: Claude Sonnet RAG beats Llama RAG by 19.3pp
- E11's high latency (98s median) is due to local Ollama inference — not a fair speed comparison, but cost is ~$0

**Speaker notes**: The 82% vs 65% gap is the central result. But the story doesn't end here — agents have a fundamentally different failure mode that becomes an advantage in different settings.

---

## Slide 12: Root Cause Analysis — Why Agents Underperform on Closed Corpus

**Diagram**: `results/confusion_matrices.png` — E4/E7/E8 side-by-side

**Root Cause 1: Query Dilution**
- Claim: "Aspirin inhibits platelet aggregation" (concise, directly searchable)
- After decomposition: sub-claims like "mechanism of aspirin on platelets" — more generic, retrieve less targeted evidence
- SciFact claims are already well-scoped — decomposition hurts when the original claim IS the best query

**Root Cause 2: Error Accumulation**
- 4 sequential LLM calls, each with ~5-10% error rate
- Compounded: 0.95^4 = 0.81 — up to 19% accuracy loss from chaining alone
- Single-pass RAG: 1 call, 1 chance to get it right

**Root Cause 3: SUPPORTED Recall Collapse**
- Both E7 and E8 drop SUPPORTED recall from 88% to 49% — the biggest single failure mode
- Diluted queries retrieve weaker evidence → Evidence Reviewer flags gaps → Verdict Agent defaults to cautious verdicts
- UNSUPPORTED is preserved (83-84%) because contradiction detection benefits from multi-step review

**Root Cause 4: Framework differences are minor**
- Strands (65.3%) vs LangGraph (66.0%) — effectively identical after bug fixes
- Strands has tool-calling agency; LangGraph uses deterministic graph execution
- Both converge because the bottleneck is upstream (query dilution), not the framework

**Speaker notes**: The insight is NOT "agents are bad" — it's that agents add value when they can take actions simple RAG cannot. On a closed corpus with well-scoped claims, extra complexity hurts. Next slides show where agents shine.

---

## Slide 13: Easy vs Hard Claims + Cross-Model Analysis

**Diagram**: `results/easy_vs_hard_claims.png`

**Using SciFact's gold evidence annotations to classify claims:**
- **Easy** (200 claims): Gold evidence document exists in corpus
- **Hard** (100 claims): No gold evidence — system must reason from tangential papers

| Architecture | Easy Claims | Hard Claims | Gap |
|-------------|-------------|-------------|-----|
| RAG (E4) | 85.5% | 75.0% | -10.5pp |
| Strands (E7) | 66.5% | 63.0% | -3.5pp |
| LangGraph (E8) | 66.0% | 66.0% | 0pp |

**Key insight**: RAG has a 10.5pp drop from easy→hard, while agents show minimal or no drop (E8: 0pp gap). Agents are equally mediocre on both, but RAG's hard-claim gap is where external search can add value.

**Cross-model comparison (all single-pass RAG)**:
| Model | Accuracy | Note |
|-------|----------|------|
| Claude Sonnet 4 (E4) | 82.0% | Best overall |
| Llama 3.1 8B (E11) | 62.7% | -19.3pp vs Claude |

Model quality is a larger factor (19pp gap) than architecture choice (17pp gap), suggesting investment in stronger models may yield higher returns than architectural complexity.

**Speaker notes**: The hard claims motivate our next experiment — what happens when the system can go beyond the local corpus?

---

## Slide 14: E9c — Smart Rerouting with Semantic Scholar Fallback

**The hypothesis**: Agents justify their cost when they can go BEYOND the local corpus.

**E9c implementation** — three key improvements over basic agents (`src/agents/strands/orchestrator_rerouting_ext_v2.py`):

```
Claim → [Smart Gate]
  SIMPLE (≤25 words, no conjunctions) →
     [Direct ChromaDB search] → [Optional S2 if weak] → [Verdict Agent]     (~20s, 2 calls)
  COMPLEX →
     [Claim Parser] → [Original-claim-first retrieval + sub-claim supplement]
     → [Evidence Reviewer] → [Optional S2 if weak/moderate] → [Verdict Agent] (~60-80s, 4-5 calls)
```

**Three innovations**:
1. **Smart gating**: Simple claims skip decomposition entirely — avoids query dilution for straightforward claims
2. **Original-claim-first retrieval**: Always search with the original claim first, sub-claims supplement — fixes the query dilution problem
3. **Aggressive S2 trigger**: External Semantic Scholar search fires on "moderate" evidence too, not just "weak" — catches borderline cases

**Semantic Scholar integration**: 200M+ papers, free API, rate-limited to 1 req/1.5s with exponential backoff on 429 errors

**Speaker notes**: Each innovation targets a specific root cause identified in the previous analysis. Smart gating reduces error accumulation, original-claim-first fixes query dilution, and S2 fallback extends coverage beyond the local corpus.

---

## Slide 15: E9c/E9d Results — Where Agents Shine

**In-corpus analysis (E9c vs E4 on 79 targeted claims)**:
- Tested on 54 claims E4 got wrong + 25 E4 successes (control)
- E9c recovered 12/54 E4 failures (22.2% recovery rate)
- E9c regressed on 4/25 E4 successes (16% regression rate)
- 97.6% of persistent failures had identical errors to E4 — a **corpus-level ceiling**, not a pipeline failure

**Out-of-corpus analysis (E9d vs E4b on 18 novel claims)**:

| Pipeline | Accuracy | SUP (11) | UNS (5) | INS (2) |
|----------|----------|----------|---------|---------|
| RAG baseline (E4b) | 22.2% | 9% (1/11) | 20% (1/5) | 100% (2/2) |
| **E9c + S2 (E9d)** | **66.7%** | **73% (8/11)** | **60% (3/5)** | **50% (1/2)** |
| | **+44.5pp** | | | |

**The S2 insight — context-dependent value**:
- On in-corpus claims: S2 adds noise (local evidence already strong) — agents with S2 slightly hurt accuracy
- On out-of-corpus claims: S2 is transformative — **+44.5pp accuracy gain**, RAG baseline collapses to predicting INSUFFICIENT_EVIDENCE for 89% of claims

**Speaker notes**: This is the key finding. RAG is strictly better on closed corpora, but agents with external search are dramatically better on open-domain claims. A production system should use RAG first, then escalate to agents when local evidence is weak.

---

## Slide 16: Cost, Latency & Practical Tradeoffs

**Diagram**: `results/cost_latency_analysis.png`

| Pipeline | Avg Latency | Cost/300 | Accuracy | Use Case |
|----------|-------------|----------|----------|----------|
| Single-pass RAG (E4) | 8.4s | $3.46 | 82.0% | Closed corpus, fast path |
| Llama 3.1 8B (E11) | 98s median* | ~$0 (local) | 62.7% | Budget/offline |
| Strands Agents (E7) | 68s median | ~$1.28** | 65.3% | Multi-agent baseline |
| LangGraph Agents (E8) | 32s median | ~$0.86** | 66.0% | Multi-agent baseline |
| Smart Rerouting (E9c) | 28s avg | — | 41.8%*** | Recovery of RAG failures |
| E9c + S2 out-of-corpus (E9d) | 22s avg | — | 66.7% | When corpus coverage is weak |

*E11 latency is local Ollama inference (no API cost)
**E7/E8 cost tracking is approximate — Bedrock costs may be underreported
***E9c tested on skewed set (54 E4 failures + 25 control); projects to 84.7% on full 300

**Recommended three-tier production architecture**:
1. **Tier 1 — RAG** (8s, ~$0.01/claim): Fast path for all claims. Best on well-indexed corpora.
2. **Tier 2 — Smart Agent** (20-80s, ~$0.02-0.05/claim): Triggered when RAG confidence is low. Uses claim decomposition + evidence review.
3. **Tier 3 — External Search** (~60s extra): Triggered when local evidence is weak/moderate. Queries Semantic Scholar for 200M+ papers.

**Speaker notes**: The key design insight is that no single architecture wins everywhere. A tiered system combines the strengths of each: RAG for speed, agents for depth, external search for coverage.

---

## Slide 17: Key Findings Summary

**Finding 1: Chunking matters more than retrieval method**
- Recursive chunking (82.0%) vs semantic chunking (76.3%) = 5.7pp gap
- Naive retrieval (82.0%) vs hybrid (79.0%) = 3.0pp gap
- Invest in how you split documents before optimizing how you search them

**Finding 2: Simplicity wins on closed corpora**
- Best pipeline: recursive chunking + naive retrieval + single-pass RAG = 82.0%
- Simplest pipeline is also the fastest (8.4s) and cheapest
- Adding multi-agent complexity reduced accuracy by 16-17pp on this benchmark

**Finding 3: Agents excel at contradiction detection**
- UNSUPPORTED recall: E7 84%, E8 83%, RAG 83% — agents match or exceed RAG
- SUPPORTED recall: E7 49%, E8 49%, RAG 88% — agents lose on confirmation
- Multi-step review acts as a "devil's advocate" — strong at finding counterevidence, weak at affirming

**Finding 4: External search is context-dependent**
- On in-corpus claims: Semantic Scholar adds noise (local evidence already strong)
- On out-of-corpus claims: S2 provides **+44.5pp** accuracy gain (22% → 67%)
- The value of external tools depends entirely on corpus coverage

**Finding 5: Model quality > architectural complexity**
- Claude Sonnet RAG (82.0%) vs Llama 3.1 8B RAG (62.7%) = 19.3pp gap (model)
- Claude Sonnet RAG (82.0%) vs Claude Sonnet Agents (65-66%) = 16-17pp gap (architecture)
- A stronger model with simple RAG outperforms a weaker architecture with the same model

**Finding 6: Corpus-level ceiling limits all architectures**
- 97.6% of E9c's persistent failures had identical errors to E4
- When the corpus doesn't contain relevant evidence, no amount of architectural sophistication helps
- This motivates the tiered approach: exhaust local evidence first, then go external

---

## Slide 18: Limitations & Future Work

**Limitations**:
- SciFact claims are concise, well-scoped — real misinformation is vague and emotional ("Big Pharma hides cures")
- Out-of-corpus evaluation is small (18 claims) — directionally strong but needs larger validation
- No evaluation of explanation quality — correct verdict could cite wrong evidence
- Single embedding model (all-MiniLM-L6-v2) — larger models may improve retrieval

**Future work**:
- **E10/E12/E13**: Cross-model experiments (GPT-4o-mini, GPT-4o-mini + agents) — configured but not yet run
- **Confidence gating**: Automatically route claims between tiers based on retrieval confidence scores
- **LLM-as-judge**: Score explanation quality and evidence grounding, not just verdict accuracy
- **Real-world claims**: Test on PUBHEALTH or social media health misinformation
- **Fine-tuned retrieval**: Domain-adapted embedding model for biomedical text

---

## Slide 19: Demo & Questions
- **Live demo**: Streamlit app with 3 tabs
  - Tab 1: Enter a claim → get verdict with evidence citations
  - Tab 2: Run batch experiments across configurations
  - Tab 3: Explore results, compare pipelines
- **Code**: Modular Python package — swap chunking/retrieval/agent with config change
- **Reproducibility**: All experiments save full provenance (config, per-claim results, latency, cost)

**Questions?**

---

## Appendix A: System Prompts (Backup Slide)

**Single-pass RAG prompt:**
"You are a scientific claim fact-checker. Given evidence passages from scientific abstracts, determine whether each claim is supported, contradicted, or lacks sufficient evidence. Base your verdict ONLY on the provided evidence."

**Claim Parser prompt:**
"You are a medical claim decomposition specialist. Break down health claims into 2-4 specific, verifiable sub-claims. For each, generate a targeted PubMed search query."

**Evidence Reviewer prompt:**
"Flag contradictions between evidence passages. Identify gaps. Note evidence quality (systematic review, RCT, observational). Assess whether evidence collectively supports, refutes, or partially supports the main claim."

**Verdict Agent prompt:**
"Based on the evidence review, generate a final verdict. SUPPORTED if evidence directly supports. UNSUPPORTED if evidence contradicts. INSUFFICIENT_EVIDENCE only when evidence truly does not address the claim."

---

## Appendix B: Detailed Per-Class Metrics (Backup Slide)

**E4 (Best RAG) — Confusion Matrix:**
```
                  Predicted:  SUP    UNS    INS
Expected SUP:                 88      3      9
Expected UNS:                  5     83     12
Expected INS:                 10     15     75
```
- Strong balanced classification across all three classes
- Most common error: INS misclassified as UNS (15/100)

**E7 (Strands Agents) — Confusion Matrix:**
```
                  Predicted:  SUP    UNS    INS    ERR
Expected SUP:                 49      8     36      7
Expected UNS:                  2     84     10      4
Expected INS:                  2     28     63      7
```
- 18/300 claims errored (Bedrock timeouts) — rows sum to 100 including ERR column
- Strong on UNSUPPORTED (84%) — agents excel at contradiction detection
- Weak on SUPPORTED (49%) — query dilution loses supporting evidence
- Over-predicts UNSUPPORTED for INSUFFICIENT_EVIDENCE claims (28/100)

**E8 (LangGraph Agents) — Confusion Matrix:**
```
                  Predicted:  SUP    UNS    INS
Expected SUP:                 49     20     31    ← 31 misclassified as INS
Expected UNS:                  0     83     17
Expected INS:                  3     31     66    ← 31 misclassified as UNS
```
- Same SUPPORTED weakness as E7 (49%)
- Similar pattern: strong UNSUPPORTED, weak SUPPORTED — confirms this is architectural, not framework-specific

**E9d vs E4b (Out-of-Corpus, 18 claims) — Comparison:**
```
E4b: Predicts INSUFFICIENT_EVIDENCE for 89% of claims (corpus has no coverage)
     Only 4/18 correct — 2 genuine INS + 1 SUP + 1 UNS by luck

E9d: 12/18 correct — recovers 8 SUP, 3 UNS via Semantic Scholar evidence
     S2 transforms out-of-corpus performance: +44.5pp over RAG baseline
```

---

## Appendix C: Experiment Configurations (Backup Slide)

| Exp | Name | Chunking | Retrieval | Architecture | Model |
|-----|------|----------|-----------|-------------|-------|
| E1 | Fixed + Naive RAG | fixed | naive | single_pass | Claude Sonnet 4 |
| E2 | Section-aware + Naive RAG | section_aware | naive | single_pass | Claude Sonnet 4 |
| E3 | Semantic + Naive RAG | semantic | naive | single_pass | Claude Sonnet 4 |
| E4 | Recursive + Naive RAG | recursive | naive | single_pass | Claude Sonnet 4 |
| E5 | Recursive + Hybrid RAG | recursive | hybrid | single_pass | Claude Sonnet 4 |
| E6 | Recursive + Hybrid Reranked | recursive | hybrid_reranked | single_pass | Claude Sonnet 4 |
| E7 | Strands Multi-Agent | recursive | naive | strands_multi | Claude Sonnet 4 |
| E8 | LangGraph Multi-Agent | recursive | naive | langgraph_multi | Claude Sonnet 4 |
| E9c | Smart Rerouting + S2 v2 | recursive | naive | strands_rerouting_ext_v2 | Claude Sonnet 4 |
| E9d | E9c on out-of-corpus claims | recursive | naive | strands_rerouting_ext_v2 | Claude Sonnet 4 |
| E4b | RAG on out-of-corpus claims | recursive | naive | single_pass | Claude Sonnet 4 |
| E11 | Llama 3.1 8B baseline | recursive | naive | single_pass | Llama 3.1 8B |

---

## Appendix D: Diagrams Inventory (Backup Slide)

**Available from notebooks** (saved to `results/`):
1. `accuracy_comparison.png` — Bar chart of all experiment accuracies → Slides 11, 16
2. `confusion_matrices.png` — Side-by-side confusion matrices for E4/E7/E8 → Slide 12, Appendix B
3. `cost_latency_analysis.png` — Scatter plots of cost vs accuracy, latency vs accuracy → Slide 16
4. `easy_vs_hard_claims.png` — Grouped bar chart by claim difficulty → Slide 13

**Generated in E9c notebook (saved after re-run)**:
5. `e9c_accuracy_comparison.png` — E9c vs E4 on 79-claim subset → Slide 15
6. `e9c_recovery_analysis.png` — Recovery pie + verdict bar chart → Slide 15
7. `e9c_confusion_matrices.png` — E4 vs E9c confusion matrices → Slide 15
8. `e9d_out_of_corpus_comparison.png` — E9d vs E4b bar chart → Slide 15
9. `s2_impact_comparison.png` — S2 value in-corpus vs out-of-corpus → Slide 15
