# RAG Pair Guide — Members 2 & 3 (Chunking + Retrieval)

You own the **chunking strategies** and **retrieval methods** — everything between the raw corpus and the evidence that reaches the LLM. Your work directly impacts experiments E1-E4 (chunking comparison) and all experiments that use hybrid/reranked retrieval.

---

## Your Deliverables

| # | Task | Files | Priority | Blocks |
|---|------|-------|----------|--------|
| 1 | Implement semantic chunking | `src/chunking/semantic.py` | **Critical** | E2, E5-E11 |
| 2 | Implement section-aware chunking | `src/chunking/section_aware.py` | **Critical** | E3 |
| 3 | Implement recursive chunking with metadata | `src/chunking/recursive.py` | **Critical** | E4 |
| 4 | Implement hybrid retrieval (BM25 + dense) | `src/retrieval/hybrid.py` | **Critical** | E2-E11 |
| 5 | Implement cross-encoder reranking | `src/retrieval/reranker.py` | **Critical** | E2-E11 |
| 6 | Implement query rewriter | `src/retrieval/query_rewriter.py` | Medium | Improves retrieval |
| 7 | Wire chunking + retrieval into configurable pipeline | `src/pipelines/configurable.py` | **Critical** | All experiments |

---

## Architecture Overview

```
corpus.json → [Chunking Strategy] → chunks → [ChromaDB Index] → [Retrieval Method] → evidence → LLM
```

You control the left side of this flow. The agent/LLM side is handled by the Agent pair.

### How your code gets called

The configurable pipeline calls your code like this:

```python
# 1. Chunking (called once during corpus preparation)
from src.chunking import chunk_corpus
chunks = chunk_corpus(corpus, strategy="semantic")  # dispatches to your function

# 2. Retrieval (called per claim during pipeline execution)
from src.retrieval.hybrid import retrieve_hybrid
hits = retrieve_hybrid(query, collection, top_k=10)

from src.retrieval.reranker import rerank
top_hits = rerank(query, hits, top_k=5)
```

---

## Task 1: Semantic Chunking

**File:** `src/chunking/semantic.py`

**Current state:** Stub with `NotImplementedError`

**What it does:** Split text at points where cosine similarity between adjacent sentences drops below a threshold, instead of at fixed token counts.

### Approach

1. Split the abstract into sentences (use `nltk.sent_tokenize` or simple regex)
2. Embed each sentence with PubMedBERT (same model we use for the vector store)
3. Compute cosine similarity between each adjacent pair of sentence embeddings
4. Split where similarity drops below the threshold (default 0.5)
5. Group consecutive sentences into chunks

### Implementation

```python
"""Semantic chunking strategy."""

import numpy as np
from sentence_transformers import SentenceTransformer

# Use the same embedding model as the vector store
MODEL_NAME = "NeuML/pubmedbert-base-embeddings"


def chunk_corpus_semantic(corpus: list[dict], similarity_threshold: float = 0.5) -> list[dict]:
    """Chunk all abstracts using semantic boundary detection."""
    model = SentenceTransformer(MODEL_NAME)
    all_chunks = []

    for article in corpus:
        abstract = article.get("abstract", "")
        if not abstract:
            continue

        chunks = _chunk_text_semantic(abstract, model, similarity_threshold)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "pmid": article["pmid"],
                "title": article["title"],
                "chunk_index": i,
                "text": chunk_text,
            })

    return all_chunks


def _chunk_text_semantic(text: str, model, threshold: float) -> list[str]:
    """Split text into chunks at semantic boundaries."""
    # Step 1: Sentence split
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    # Step 2: Embed sentences
    embeddings = model.encode(sentences)

    # Step 3: Compute adjacent cosine similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        similarities.append(sim)

    # Step 4: Split where similarity drops below threshold
    chunks = []
    current_chunk = [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])
    chunks.append(" ".join(current_chunk))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter (replace with nltk if preferred)."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]
```

### Key decisions to make
- **Threshold value:** 0.5 is a starting point. Try 0.3-0.7 range and pick the one that gives reasonable chunk sizes (50-300 tokens)
- **Minimum chunk size:** Consider merging very small chunks (< 30 tokens) with their neighbors
- **Model loading:** Load the model once, not per-article. The function signature accepts `corpus` so you can control this

---

## Task 2: Section-Aware Chunking

**File:** `src/chunking/section_aware.py`

**What it does:** Parse PubMed structured abstracts into sections (Background, Methods, Results, Conclusions) and chunk per-section.

### Approach

Many PubMed abstracts have section headers like:
```
BACKGROUND: Vaccines have been...
METHODS: We conducted a systematic...
RESULTS: Among 657,461 children...
CONCLUSIONS: The study strongly supports...
```

1. Detect section headers via regex
2. Split at section boundaries
3. If no sections detected, fall back to fixed chunking
4. Tag each chunk with its section name in metadata

### Implementation sketch

```python
"""Section-aware chunking strategy."""

import re

# Common PubMed abstract section headers
SECTION_PATTERNS = [
    r'\b(BACKGROUND|INTRODUCTION|CONTEXT)\s*:',
    r'\b(OBJECTIVE|AIM|PURPOSE)\s*:',
    r'\b(METHODS|MATERIALS AND METHODS|STUDY DESIGN)\s*:',
    r'\b(RESULTS|FINDINGS)\s*:',
    r'\b(DISCUSSION)\s*:',
    r'\b(CONCLUSION|CONCLUSIONS|SUMMARY)\s*:',
]
SECTION_RE = re.compile('|'.join(SECTION_PATTERNS), re.IGNORECASE)


def chunk_corpus_section_aware(corpus: list[dict]) -> list[dict]:
    """Chunk all abstracts by detected section boundaries."""
    all_chunks = []

    for article in corpus:
        abstract = article.get("abstract", "")
        if not abstract:
            continue

        sections = _split_into_sections(abstract)

        if not sections:
            # Fallback to fixed chunking for unstructured abstracts
            from src.chunking.fixed import chunk_text
            texts = chunk_text(abstract)
            for i, text in enumerate(texts):
                all_chunks.append({
                    "pmid": article["pmid"],
                    "title": article["title"],
                    "chunk_index": i,
                    "text": text,
                    "metadata": {"section": "unstructured"},
                })
        else:
            for i, (section_name, section_text) in enumerate(sections):
                all_chunks.append({
                    "pmid": article["pmid"],
                    "title": article["title"],
                    "chunk_index": i,
                    "text": section_text,
                    "metadata": {"section": section_name.lower()},
                })

    return all_chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_name, section_text) pairs.

    Returns empty list if no section headers detected.
    """
    # Find all section header positions
    matches = list(SECTION_RE.finditer(text))
    if len(matches) < 2:
        return []  # Not enough sections to be meaningful

    sections = []
    for i, match in enumerate(matches):
        section_name = match.group().rstrip(':').strip()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))

    return sections
```

### Key decisions
- **Large sections:** If a section is > 500 tokens, consider sub-chunking it with fixed chunking
- **Section metadata:** The `metadata.section` field allows analysis like "do Results sections retrieve better than Methods sections?"

---

## Task 3: Recursive Chunking with Metadata

**File:** `src/chunking/recursive.py`

**What it does:** Use LangChain's `RecursiveCharacterTextSplitter` with paragraph → sentence → character fallback, then enrich chunks with extracted metadata.

### Approach

```python
"""Recursive chunking strategy with metadata enrichment."""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_corpus_recursive(corpus: list[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """Chunk all abstracts using recursive splitting with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph → sentence → word → char
    )
    all_chunks = []

    for article in corpus:
        abstract = article.get("abstract", "")
        if not abstract:
            continue

        texts = splitter.split_text(abstract)
        metadata = _extract_metadata(abstract, article)

        for i, text in enumerate(texts):
            all_chunks.append({
                "pmid": article["pmid"],
                "title": article["title"],
                "chunk_index": i,
                "text": text,
                "metadata": metadata,
            })

    return all_chunks


def _extract_metadata(abstract: str, article: dict) -> dict:
    """Extract structured metadata from abstract text."""
    metadata = {"year": article.get("year", "")}

    # Study type detection
    study_type = "unknown"
    lower = abstract.lower()
    if "meta-analysis" in lower or "meta analysis" in lower:
        study_type = "meta-analysis"
    elif "systematic review" in lower:
        study_type = "systematic_review"
    elif "randomized" in lower or "randomised" in lower:
        study_type = "rct"
    elif "cohort" in lower:
        study_type = "cohort"
    elif "case-control" in lower or "case control" in lower:
        study_type = "case_control"
    elif "cross-sectional" in lower:
        study_type = "cross_sectional"
    metadata["study_type"] = study_type

    # Sample size extraction
    match = re.search(r'[Nn]\s*=\s*([\d,]+)', abstract)
    if match:
        metadata["sample_size"] = match.group(1).replace(",", "")

    return metadata
```

### Dependencies
You'll need to add `langchain-text-splitters` to `pyproject.toml`:
```bash
uv add langchain-text-splitters
```

---

## Task 4: Hybrid Retrieval

**File:** `src/retrieval/hybrid.py`

**What it does:** Combine BM25 (lexical) and dense (ChromaDB/PubMedBERT) retrieval using reciprocal rank fusion.

### Approach

```python
"""Hybrid retrieval — BM25 + dense with reciprocal rank fusion."""

from rank_bm25 import BM25Okapi


def retrieve_hybrid(query, collection, corpus_texts=None, top_k=10, bm25_weight=0.5):
    """Retrieve using BM25 + dense, fused with RRF."""

    # 1. Dense retrieval from ChromaDB
    dense_results = collection.query(query_texts=[query], n_results=top_k * 2)

    # 2. BM25 retrieval
    # corpus_texts should be pre-tokenized list of all chunk texts
    tokenized = [doc.split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())

    # 3. Reciprocal rank fusion
    # ... combine rankings from both methods
    # RRF score = sum(1 / (k + rank_i)) across all rankers

    # 4. Return top-k by fused score
    ...
```

### Dependencies
```bash
uv add rank-bm25
```

### Key considerations
- **BM25 indexing:** BM25 needs all corpus texts in memory. You can build this once at startup and cache it
- **RRF constant k:** Standard value is 60. `score = 1/(60 + rank)`
- **Dense result format:** ChromaDB returns `{'ids': [[...]], 'documents': [[...]], 'distances': [[...]], 'metadatas': [[...]]}`

---

## Task 5: Cross-Encoder Reranking

**File:** `src/retrieval/reranker.py`

**What it does:** Take the top-k results from hybrid retrieval and rerank them using a cross-encoder model that scores (query, passage) pairs jointly.

### Approach

```python
"""Cross-encoder re-ranking."""

from sentence_transformers import CrossEncoder


def rerank(query, passages, top_k=5, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Re-rank passages using a cross-encoder."""
    model = CrossEncoder(model_name)

    # Score each (query, passage) pair
    pairs = [(query, p["text"]) for p in passages]
    scores = model.predict(pairs)

    # Sort by score and return top-k
    for i, p in enumerate(passages):
        p["rerank_score"] = float(scores[i])

    ranked = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]
```

### Dependencies
```bash
uv add sentence-transformers
```

The model `cross-encoder/ms-marco-MiniLM-L-6-v2` downloads automatically on first use (~80MB). It runs locally, no API key needed.

---

## Task 6: Query Rewriter

**File:** `src/retrieval/query_rewriter.py`

**What it does:** Use an LLM to expand a health claim into a better search query with medical synonyms.

### Approach

```python
"""LLM-based query rewriting."""

from src.shared.llm import call_llm


def rewrite_query(claim, model=None):
    """Rewrite a claim into an optimised search query."""
    prompt = (
        f"Rewrite this health claim as a PubMed search query. "
        f"Add medical synonyms and MeSH terms. Return ONLY the query, nothing else.\n\n"
        f"Claim: {claim}"
    )
    response = call_llm(prompt, system="You are a medical librarian.", max_tokens=200)
    return response["content"].strip()
```

This is a nice-to-have enhancement. Lower priority than Tasks 1-5.

---

## Task 7: Wire Into Configurable Pipeline

**File:** `src/pipelines/configurable.py` (modify existing)

The `_run_single_pass()` function currently only handles `retrieval_method="naive"`. You need to add the hybrid and reranked paths.

### What to change in `configurable.py`

```python
def _run_single_pass(claim, retrieval_method, model):
    if retrieval_method == "naive":
        from src.pipelines.p1_naive_single.pipeline import run as run_p1
        result = run_p1(claim)
        return {
            "verdict": result["verdict"],
            "explanation": result["explanation"],
            "evidence": result["evidence"],
        }

    elif retrieval_method == "hybrid":
        # YOUR CODE: load chunks, run hybrid retrieval, pass to LLM
        ...

    elif retrieval_method == "hybrid_reranked":
        # YOUR CODE: hybrid retrieval → reranker → LLM
        ...
```

You'll also need to ensure that different chunking strategies produce ChromaDB collections. Consider a helper that:
1. Checks if a ChromaDB collection exists for the given strategy
2. If not, runs `chunk_corpus(corpus, strategy=...)` and indexes the chunks
3. Returns the collection for retrieval

---

## Suggested Division of Work (Members 2 & 3)

| Member | Tasks | Why |
|--------|-------|-----|
| **Member 2** | Semantic chunking + Hybrid retrieval + Query rewriter | Chunking and retrieval are tightly coupled for semantic |
| **Member 3** | Section-aware + Recursive chunking + Reranker + Pipeline wiring | Section-aware is regex-heavy; recursive uses LangChain |

Or split by layer:
- **Member 2:** All 3 chunking strategies (Tasks 1-3)
- **Member 3:** All retrieval methods + pipeline wiring (Tasks 4-7)

---

## Testing Your Code

### Quick test for a chunking strategy

```python
from src.shared.corpus_loader import load_corpus
from src.chunking import chunk_corpus

corpus = load_corpus("data/corpus.json")
chunks = chunk_corpus(corpus, strategy="semantic")  # or section_aware, recursive
print(f"Total chunks: {len(chunks)}")
print(f"Avg chunk length: {sum(len(c['text'].split()) for c in chunks) / len(chunks):.0f} tokens")
print(f"\nFirst 3 chunks:")
for c in chunks[:3]:
    print(f"  [{c['pmid']}] chunk {c['chunk_index']}: {c['text'][:100]}...")
```

### Quick test for retrieval

```python
from src.shared.vector_store import get_or_create_collection
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.reranker import rerank

collection = get_or_create_collection()
hits = retrieve_hybrid("vaccines autism", collection, top_k=10)
print(f"Hybrid hits: {len(hits)}")

reranked = rerank("vaccines autism", hits, top_k=5)
for h in reranked:
    print(f"  score={h['rerank_score']:.3f}: {h['text'][:80]}...")
```

### Run via experiment runner

Once wired into the configurable pipeline:

```bash
# This should work end-to-end
uv run python -m src.experiment_runner E2
```

---

## Output Format

All chunking functions must return `list[dict]` where each dict has:

```python
{
    "pmid": "30986133",           # Required
    "title": "Measles, Mumps...", # Required
    "chunk_index": 0,             # Required (integer, 0-based per article)
    "text": "The chunk text...",  # Required
    "metadata": {                 # Optional (section-aware and recursive add this)
        "section": "results",
        "study_type": "meta-analysis",
        "sample_size": "657461",
    }
}
```

All retrieval functions must return `list[dict]` where each dict has:

```python
{
    "id": "30986133_0",           # Chunk ID
    "text": "The passage text...",# Passage text
    "metadata": {...},            # ChromaDB metadata
    "score": 0.85,                # Relevance score (higher = better)
}
```

---

## Dependencies to Add

```bash
uv add rank-bm25 sentence-transformers langchain-text-splitters
```

`sentence-transformers` is likely already installed (used by the embedding model). Check `pyproject.toml`.

---

## Files Reference

| File | Status | Owner |
|------|--------|-------|
| `src/chunking/__init__.py` | Done — dispatcher | — |
| `src/chunking/fixed.py` | Done — baseline | — |
| `src/chunking/semantic.py` | **Stub → implement** | You |
| `src/chunking/section_aware.py` | **Stub → implement** | You |
| `src/chunking/recursive.py` | **Stub → implement** | You |
| `src/retrieval/naive.py` | Done — baseline | — |
| `src/retrieval/hybrid.py` | **Stub → implement** | You |
| `src/retrieval/reranker.py` | **Stub → implement** | You |
| `src/retrieval/query_rewriter.py` | **Stub → implement** | You |
| `src/pipelines/configurable.py` | Needs hybrid/reranked paths | You |
| `src/shared/vector_store.py` | Done — ChromaDB client | — |
| `src/shared/embeddings.py` | Done — PubMedBERT config | — |
