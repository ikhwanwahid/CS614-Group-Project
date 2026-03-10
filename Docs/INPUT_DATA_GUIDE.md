# Input Data Guide — Member 1 (Data & Ground Truth)

This guide covers everything you need to build the test dataset and corpus for the project. Follow the steps in order — the rest of the team is blocked until **Task 1** and **Task 2** are done.

---

## Your Deliverables

| # | Task | Output File | Priority | Blocks |
|---|------|------------|----------|--------|
| 1 | Build 120+ claim dataset with verdict labels | `data/test_claims.json` | **Critical** | Everyone |
| 2 | Expand PubMed corpus to cover new topics | `data/corpus.json` | **Critical** | RAG pair |
| 3 | Add difficulty/category tags to claims | (same file) | Medium | Analysis |
| 4 | Human calibration subset (~20 claims) | `data/human_calibration.json` | Lower | Final report |

---

## Task 1: Build the Claim Dataset (120+ claims)

### Data Sources

Two public datasets with verdict labels:

1. **PUBHEALTH** — https://huggingface.co/datasets/health_fact
   - ~12K claims with labels: `true`, `false`, `mixture`, `unproven`
   - Sourced from fact-checking sites (Snopes, PolitiFact health, etc.)
   - Download: `datasets` library or direct download from HuggingFace

2. **ANTi-Vax** — search for "ANTi-Vax dataset" on HuggingFace/GitHub
   - Vaccine-specific misinformation claims
   - Labels vary by version — map to our 4 verdicts

### Verdict Mapping

Map source labels to our 4 verdicts:

| Source Label | Our Verdict | Meaning |
|-------------|-------------|---------|
| `true` / `supported` | `SUPPORTED` | Evidence confirms the claim |
| `false` / `refuted` | `UNSUPPORTED` | Evidence contradicts the claim |
| `mixture` / `partially true` | `OVERSTATED` | Kernel of truth but misleading as stated |
| `unproven` / `not enough info` | `INSUFFICIENT_EVIDENCE` | Cannot determine from available evidence |

### Target Distribution

Aim for a balanced dataset across verdicts:

| Verdict | Target Count | Why |
|---------|-------------|-----|
| SUPPORTED | ~30-35 | Need enough to measure true positive rate |
| UNSUPPORTED | ~30-35 | Most common in misinformation datasets |
| OVERSTATED | ~25-30 | Hardest category — tests nuanced reasoning |
| INSUFFICIENT_EVIDENCE | ~20-25 | Tests whether pipeline avoids overconfidence |
| **Total** | **120+** | Enough for statistical significance (McNemar's test) |

### Topic Diversity

Try to cover these health topics (not just vaccines):

- Vaccines (COVID, MMR, flu, HPV)
- Nutrition (vitamins, supplements, diets)
- Exercise and fitness
- Mental health
- Chronic disease (diabetes, heart disease, cancer)
- Medications and treatments
- Public health measures (masking, hand washing)

### Output Format

Save to `data/test_claims.json`. Must match this exact format:

```json
[
  {
    "claim": "Vaccines cause autism",
    "expected_verdict": "UNSUPPORTED",
    "difficulty": "easy",
    "category": "vaccines",
    "source": "PUBHEALTH",
    "source_id": "pubhealth_1234"
  },
  {
    "claim": "Vitamin D supplements prevent COVID infection",
    "expected_verdict": "OVERSTATED",
    "difficulty": "nuanced",
    "category": "nutrition",
    "source": "PUBHEALTH",
    "source_id": "pubhealth_5678"
  }
]
```

**Required fields:**
- `claim` (string) — the health claim, as a single sentence
- `expected_verdict` (string) — one of: `SUPPORTED`, `UNSUPPORTED`, `OVERSTATED`, `INSUFFICIENT_EVIDENCE`

**Optional but helpful fields:**
- `difficulty` — one of: `easy`, `nuanced`, `mechanistic`, `mixed_evidence`, `simple_supported`
- `category` — topic area (e.g., `vaccines`, `nutrition`, `exercise`, `mental_health`, `chronic_disease`, `medications`)
- `source` — which dataset it came from (`PUBHEALTH`, `ANTIVAX`, `manual`)
- `source_id` — original ID in the source dataset (for traceability)

### Existing Claims (keep these)

The POC already has 7 claims in `data/test_claims.json`. Keep them and add new ones after. They serve as a baseline we've already tested.

### Quick Start with PUBHEALTH

```python
# Option 1: Using HuggingFace datasets library
from datasets import load_dataset

ds = load_dataset("health_fact")
train = ds["train"]

# Browse claims and labels
for i, row in enumerate(train):
    print(f"{row['label']}: {row['claim'][:80]}")
    if i > 20:
        break

# Option 2: Direct download
# Go to https://huggingface.co/datasets/health_fact
# Download the parquet/csv files
# Load with pandas
```

### Selection Criteria

When picking claims from PUBHEALTH/ANTi-Vax:

**Do pick:**
- Claims that are self-contained single sentences
- Claims about topics where PubMed abstracts would have relevant evidence
- Claims spanning different difficulty levels
- Claims where the verdict label seems correct/unambiguous

**Don't pick:**
- Claims that require very recent data (post-2023) — our corpus is PubMed abstracts
- Claims about specific people or political events (not generalizable)
- Claims that are too vague ("health is important")
- Claims where the source label seems wrong or debatable — skip these

---

## Task 2: Expand the Corpus

The current corpus has **36 PubMed abstracts** focused on vaccines. For 120+ claims across diverse topics, we need **150-300 abstracts**.

### How It Works

The script `scripts/fetch_corpus.py` searches PubMed and downloads abstracts. You just need to **add more search queries** to cover the new claim topics.

### Steps

1. **Edit `scripts/fetch_corpus.py`** — add search queries for your claim topics:

```python
SEARCH_QUERIES = [
    # Existing (keep these)
    "COVID-19 vaccine efficacy",
    "MMR vaccine autism",
    "Vitamin D COVID prevention",
    "mRNA vaccine DNA",
    "HPV vaccine safety",
    "influenza vaccine elderly hospitalisation",
    "COVID-19 vaccine variants effectiveness",

    # Add new queries based on your claims — examples:
    "intermittent fasting type 2 diabetes",
    "omega-3 heart disease prevention",
    "exercise depression anxiety treatment",
    "vitamin C common cold prevention",
    "turmeric anti-inflammatory evidence",
    "probiotics gut health systematic review",
    "meditation blood pressure reduction",
    "sleep duration cardiovascular risk",
    "sugar consumption obesity children",
    "antioxidant supplements cancer prevention",
    "yoga chronic pain management",
    "green tea weight loss",
    "statins side effects muscle pain",
    "zinc supplements immune function",
    "screen time children mental health",
    # ... add more to match your claims
]

MAX_RESULTS_PER_QUERY = 8  # increase from 6 to get more coverage
```

2. **Run the script:**

```bash
uv run python scripts/fetch_corpus.py
```

3. **Verify coverage** — after fetching, check that your claims have relevant abstracts:

```python
import json

with open("data/corpus.json") as f:
    corpus = json.load(f)

print(f"Total abstracts: {len(corpus)}")

# Search for a keyword to check coverage
keyword = "diabetes"
hits = [a for a in corpus if keyword.lower() in a["abstract"].lower()]
print(f"Abstracts mentioning '{keyword}': {len(hits)}")
```

4. **Re-index ChromaDB** — delete the old database so it rebuilds on next run:

```bash
rm -rf data/corpus/embeddings/chroma_db
```

The vector store will rebuild automatically when any pipeline runs.

### Coverage Check

For each claim topic, you want at least **3-5 relevant abstracts** in the corpus. If a topic has zero coverage, the pipeline will have nothing to retrieve and will likely return `INSUFFICIENT_EVIDENCE` regardless of its actual verdict — which pollutes the results.

**Important:** Not every claim needs perfect corpus coverage. Some claims *should* have weak coverage (to test how pipelines handle `INSUFFICIENT_EVIDENCE`). But the majority should have retrievable evidence.

---

## Task 3: Difficulty Tags

Tag each claim with a difficulty level. This lets us analyze questions like "does the multi-agent pipeline help more on nuanced claims?"

| Difficulty | Description | Example |
|-----------|-------------|---------|
| `easy` | Clear scientific consensus, straightforward | "Vaccines cause autism" → UNSUPPORTED |
| `simple_supported` | Well-established and supported | "Flu vaccines reduce hospitalisation in elderly" → SUPPORTED |
| `nuanced` | Partially true but overstated | "Vitamin D prevents COVID" → OVERSTATED |
| `mechanistic` | Requires understanding biological mechanism | "mRNA vaccines alter your DNA" → UNSUPPORTED |
| `mixed_evidence` | Conflicting studies, hard to judge | "COVID vaccines effective against all variants" → OVERSTATED |

Use your judgment. When unsure, `nuanced` is a safe default.

---

## Task 4: Human Calibration Subset

Pick ~20 claims (spread across verdicts and difficulties) and write a brief justification for each. This is used to check whether the LLM-as-Judge scoring is reasonable.

Save to `data/human_calibration.json`:

```json
[
  {
    "claim": "Vaccines cause autism",
    "expected_verdict": "UNSUPPORTED",
    "human_justification": "Multiple large-scale studies (e.g., Hviid et al. 2019, n=657,461) found no association between MMR vaccination and autism. The original Wakefield study was retracted for fraud. Scientific consensus is clear.",
    "key_evidence": ["PMID:30831578", "PMID:24814559"],
    "confidence": "high"
  }
]
```

**Fields:**
- `human_justification` — 2-3 sentences explaining why the verdict is correct
- `key_evidence` — PMIDs of relevant papers (optional, but helpful)
- `confidence` — `high`, `medium`, or `low` (how confident you are in the label)

This doesn't need to be perfect — it's a calibration tool, not a gold standard. Focus on the 20 most representative claims.

---

## Timeline

| Week | Deliverable |
|------|------------|
| **Week 1** | Task 1: First 60 claims selected and formatted. Task 2: Corpus expanded with new queries. |
| **Week 2** | Task 1: Remaining 60+ claims. Task 3: Difficulty tags added. Share `test_claims.json` with team. |
| **Week 3** | Task 4: Human calibration subset. Final cleanup based on team feedback. |

---

## FAQ

**Q: What if a claim from PUBHEALTH doesn't fit neatly into our 4 verdicts?**
Skip it. There are thousands of claims — pick ones with clear mappings.

**Q: How do I know if a claim has enough corpus coverage?**
Run the coverage check in Task 2. If a keyword search on the abstract text returns 0 hits, either add a PubMed query for that topic or skip the claim.

**Q: Should I include claims our pipeline will definitely get wrong?**
Yes — some hard claims are good. We want to measure where pipelines struggle, not just confirm they work on easy ones. Aim for ~60% claims we expect to get right, ~40% that are challenging.

**Q: Can I add claims manually (not from PUBHEALTH/ANTi-Vax)?**
Yes, but mark them with `"source": "manual"`. Manual claims need extra care to ensure the verdict label is correct — get a second opinion.

**Q: What if PUBHEALTH labels seem wrong?**
Trust your judgment. If a label seems wrong, either skip the claim or correct the label and note it with `"source": "PUBHEALTH_corrected"`.

---

## Files Reference

| File | What | Status |
|------|------|--------|
| `data/test_claims.json` | Main claim dataset | Has 7 POC claims, needs 120+ |
| `data/corpus.json` | PubMed abstracts | Has 36, needs 150-300 |
| `data/human_calibration.json` | Human justifications | Does not exist yet |
| `scripts/fetch_corpus.py` | PubMed fetcher | Working, needs more queries |
| `data/corpus/embeddings/chroma_db/` | Vector store | Auto-rebuilds from corpus |
| `data/corpus/processed/chunks.json` | Chunked corpus | Auto-rebuilds from corpus |
