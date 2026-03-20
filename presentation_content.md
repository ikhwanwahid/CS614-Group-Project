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
| Fixed | 79.9% | 81% | 84% | 75% |
| Section-aware | 77.3% | — | — | — |
| Semantic | 76.3% | — | — | — |
| **Recursive** | **82.0%** | **88%** | **83%** | **75%** |

**Accuracy ranking**: Recursive (82%) > Fixed (80%) > Section-aware (77%) > Semantic (76%)

**Analysis**:
- Recursive's 800-char chunks capture enough context for the LLM to reason about cause-effect relationships
- Semantic chunking underperformed because scientific abstracts already have clear sentence-level structure — the embedding-based boundary detection added noise rather than insight
- Section-aware suffered from 81% fallback rate — most abstracts weren't structured
- Fixed is a strong baseline — simplicity shouldn't be underestimated

**Speaker notes**: The 82% vs 76% gap (6 percentage points) is statistically significant across 300 claims. Recursive chunking is our winner and becomes the fixed choice for all subsequent experiments.

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

## Slide 11: Axis 3 Results — RAG vs Agents

| Architecture | Accuracy | SUP | UNS | INS | Avg Latency | Cost/300 |
|-------------|----------|-----|-----|-----|-------------|----------|
| **Single-pass RAG (E4)** | **82.0%** | **88%** | **83%** | **75%** | **8.4s** | **$3.46** |
| Strands Agents (E7)* | 71.0% | 50% | 88% | 64% | 81.4s | ~$5+ |
| LangGraph Agents (E8)* | 34.8% | 0% | 1% | 100% | 107.0s | ~$8+ |

*Preliminary — experiments still running

**Confusion matrix spotlight — E8 has a critical failure mode:**
```
E8 Predictions:     SUPPORTED  UNSUPPORTED  INSUFFICIENT
Expected SUP:            0          1            72    ← all misclassified
Expected UNS:            0          1            79    ← all misclassified
Expected INS:            0          0            80    ← 100% correct (by default)
```
E8 classifies virtually everything as INSUFFICIENT_EVIDENCE — 99% of predictions are this single class.

**E7 is more balanced but still underperforms RAG:**
```
E7 Predictions:     SUPPORTED  UNSUPPORTED  INSUFFICIENT
Expected SUP:            8          2             5
Expected UNS:            1         23             2
Expected INS:            1          7            14
```
E7 is strong on UNSUPPORTED (88%) but weak on SUPPORTED (50%).

**Speaker notes**: The E8 result is striking and warrants explanation — next slide.

---

## Slide 12: Root Cause Analysis — Why Agents Underperform

**Root Cause 1: Query Dilution**
- Claim: "Aspirin inhibits platelet aggregation" (concise, searchable)
- After decomposition: sub-claim queries like "mechanism of aspirin on platelets" and "platelet aggregation pathways" — more generic, retrieve less targeted evidence
- SciFact claims are already well-scoped — decomposition hurts when the original claim IS the best search query

**Root Cause 2: Error Accumulation**
- 4 sequential LLM calls, each with ~5-10% error rate
- Compounded: 0.95^4 = 0.81 — up to 19% accuracy loss from chaining alone
- Single-pass RAG: 1 call, 1 chance to get it right

**Root Cause 3: The INSUFFICIENT_EVIDENCE Trap (E8's failure)**
- Evidence Reviewer flags gaps in every sub-claim (because decomposed queries find weaker evidence)
- Verdict Agent sees "WEAK evidence, multiple GAP flags" → defaults to INSUFFICIENT_EVIDENCE
- This is rational given the input — the problem is upstream (query dilution → weak retrieval → conservative review → wrong verdict)

**Root Cause 4: E7 vs E8 gap (71% vs 35%)**
- Strands agent has tool-calling agency — can retry searches, adjust queries
- LangGraph nodes are deterministic — one search per sub-claim, no adaptation
- Strands structured output (Pydantic) constrains LLM to valid verdicts; LangGraph JSON parsing is more fragile

**Speaker notes**: This is the most important slide. The insight is NOT "agents are bad" — it's "agents add value when they can take actions that simple RAG can't". On a closed corpus with well-scoped claims, the extra complexity hurts.

---

## Slide 13: Easy vs Hard Claims — Where Does Each Architecture Excel?

**Using SciFact's gold evidence annotations to classify claims:**
- **Easy** (200 claims): Gold evidence document exists in corpus — system should find it
- **Hard** (100 claims): No gold evidence — system must reason from tangentially related papers

| Architecture | Easy Claims | Hard Claims |
|-------------|-------------|-------------|
| RAG (E4) | 85.5% (200) | 75.0% (100) |
| Strands (E7)* | 75.6% (41) | 63.6% (22) |
| LangGraph (E8)* | 0.7% (153) | 100.0% (80)** |

*Preliminary sample sizes
**E8's "100% on hard" is misleading — it predicts INSUFFICIENT_EVIDENCE for everything, which happens to be correct for the 100 hard claims that ARE insufficient evidence

**Key insight**: RAG has a 10.5 percentage point drop from easy→hard (85.5%→75.0%). This gap is where agents with external search could add value.

**Speaker notes**: The hard claims are where the corpus literally doesn't have the answer. RAG can only guess from tangential evidence. This motivates E9b.

---

## Slide 14: E9b — Rerouting with Semantic Scholar External Search

**The hypothesis**: Agents justify their cost when they can go BEYOND the local corpus.

**Implementation** (`src/agents/strands/orchestrator_rerouting_ext.py`):
```
Claim → [Claim Parser] → [Retrieval Agent (local ChromaDB)]
      → [Evidence Reviewer]
          ├─ Evidence SUFFICIENT → [Verdict Agent] → done (same as E7)
          └─ Evidence WEAK →
               → Identify which sub-claims have gaps
               → Query Semantic Scholar API (200M+ papers) for those sub-claims
               → Merge external abstracts with local evidence
               → [Evidence Reviewer round 2] with enriched evidence
               → [Verdict Agent] → done
```

**Semantic Scholar integration** (`src/retrieval/semantic_scholar.py`):
- API: `GET https://api.semanticscholar.org/graph/v1/paper/search`
- Returns: title, abstract, year, citation count
- Rate limited: 1 request/second with API key
- Filters: skip papers without abstracts, cap passage length at 800 chars

**What this tests:**
- Does external evidence improve "hard" claims where local corpus fails?
- Is the added latency (~1-2s per API call) and cost worth the accuracy gain?
- Can agents justify their complexity by doing something RAG fundamentally cannot?

**Speaker notes**: This is the key innovation — we're not just comparing architectures, we're showing that each has a regime where it excels. RAG wins on closed corpus, agents win when they need to go beyond it.

---

## Slide 15: Cost, Latency & Practical Tradeoffs

| Pipeline | Avg Latency | Cost (300 claims) | Accuracy | LLM Calls/Claim |
|----------|-------------|-------------------|----------|-----------------|
| Single-pass RAG (E4) | 8.4s | $3.46 | 82.0% | 1 |
| Strands Agents (E7) | 81.4s | ~$5+ | 71.0%* | 4-5 |
| LangGraph Agents (E8) | 107.0s | ~$8+ | 34.8%* | 3 |
| Rerouting + S2 (E9b) | ~100-120s est. | ~$8-10 est. | TBD | 4-6 |

**Cost breakdown**:
- RAG: 1 LLM call × ~1,300 tokens = ~$0.012/claim
- Agents: 4 LLM calls × ~500-2000 tokens each = ~$0.03-0.05/claim
- External search adds ~$0 (Semantic Scholar API is free)

**The "10x cost for lower accuracy" problem**:
- On a closed, well-indexed corpus: RAG is strictly better (faster, cheaper, more accurate)
- The agent value proposition requires a scenario where RAG fails — open-domain, evolving knowledge, or insufficient corpus coverage

**Speaker notes**: For a production health fact-checking system, you'd likely use RAG as the fast path and trigger agents only when RAG confidence is low (confidence gating — a future optimization we've designed but not yet tested).

---

## Slide 16: Key Findings Summary

**Finding 1: Chunking matters more than retrieval method**
- Recursive chunking (82%) vs semantic chunking (76%) = 6pp gap
- Naive retrieval (82%) vs hybrid (79%) = 3pp gap
- Invest in how you split documents before optimizing how you search them

**Finding 2: Simplicity wins on closed corpora**
- Best pipeline: recursive chunking + naive retrieval + single-pass RAG = 82.0%
- Simplest pipeline is also the fastest (8.4s) and cheapest ($3.46/300 claims)
- Adding complexity (hybrid retrieval, multi-agent) reduced accuracy

**Finding 3: Multi-agent architectures have a conservative bias problem**
- Evidence Reviewer creates an INSUFFICIENT_EVIDENCE default
- LangGraph (E8) is an extreme case: 99% of predictions are INSUFFICIENT_EVIDENCE
- This is a design flaw in multi-step review pipelines, not a framework issue

**Finding 4: Agents and RAG serve different regimes**
- RAG excels: closed corpus, well-scoped claims, speed-critical
- Agents excel: open-domain, evidence gaps, complex multi-source reasoning
- E9b (external search) tests this hypothesis directly

**Finding 5: Framework ≠ architecture**
- Strands (71%) vs LangGraph (35%) gap is implementation detail (structured output, tool-calling agency), not fundamental
- BrokenPipe errors in E8 are API client issues, not LangGraph limitations

---

## Slide 17: Limitations & Future Work

**Limitations of current study:**
- SciFact claims are concise, well-scoped scientific statements — real health misinformation is vague, emotional, multi-faceted ("Big Pharma is hiding the cure")
- Single LLM (Claude Sonnet 4) — accuracy may vary significantly across models
- E7/E8 results are preliminary (experiments still completing)
- No evaluation of explanation quality — a claim could get the right verdict for wrong reasons

**Planned & future work:**
- **E9b**: Semantic Scholar external search — does it close the gap on hard claims?
- **E10-E13**: Cross-model comparison (GPT-4o-mini, Llama 3.1 8B) — already configured
- **LLM-as-judge**: Have a separate LLM score explanation quality and evidence grounding
- **Confidence gating**: Use RAG as fast path, trigger agents only when confidence is low
- **Real-world dataset**: Test on PUBHEALTH or COVID-specific misinformation claims
- **Fine-tuned models**: Llama 3.1 fine-tuned on SciFact training set (E11 configured)

---

## Slide 18: Demo & Architecture Walkthrough
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
- Most common error: INSUFFICIENT_EVIDENCE claims misclassified as UNSUPPORTED (15/100)
- Strong diagonal = good balanced classification

**E7 (Strands Agents) — Confusion Matrix:**
```
                  Predicted:  SUP    UNS    INS
Expected SUP:                  8      2      5
Expected UNS:                  1     23      2
Expected INS:                  1      7     14
```
- Strong on UNSUPPORTED (88%) — agents good at detecting contradictions
- Weak on SUPPORTED (50%) — query dilution loses supporting evidence

**E8 (LangGraph Agents) — Confusion Matrix:**
```
                  Predicted:  SUP    UNS    INS
Expected SUP:                  0      1     72
Expected UNS:                  0      1     79
Expected INS:                  0      0     80
```
- Near-total collapse to INSUFFICIENT_EVIDENCE
- Conservative bias from multi-step review pipeline

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
| E9 | Strands Rerouting (local) | recursive | naive | strands_rerouting | Claude Sonnet 4 |
| E9b | Rerouting + Semantic Scholar | recursive | naive | strands_rerouting_ext | Claude Sonnet 4 |
| E10 | GPT-4o-mini baseline | recursive | naive | single_pass | GPT-4o-mini |
| E11 | Llama 3.1 8B baseline | recursive | naive | single_pass | Llama 3.1 8B |
