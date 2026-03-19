# SciFact Migration — Branch `feat/scifact-claims`

## Why

The PUBHEALTH test claims were mostly **news headlines** (policy, politics, statistics), not medical/scientific assertions. Our PubMed corpus couldn't verify them, resulting in **~33% accuracy** with the model defaulting to INSUFFICIENT_EVIDENCE for 72% of claims.

SciFact (AllenAI, EMNLP 2020) contains **expert-written scientific claims** paired with PubMed abstracts — exactly what our RAG pipeline is designed for.

**Result: accuracy jumped from 33% to 81%** with the same pipeline (E1 config).

## What Changed

### Data
| File | Before (main) | After (this branch) |
|------|--------------|---------------------|
| `data/corpus.json` | 511 PubMed articles (hand-picked queries) | **5,183 PubMed abstracts** (SciFact corpus) |
| `data/test_claims.json` | 179 PUBHEALTH claims (4 classes) | **300 SciFact claims** (3 classes, 100 each) |

### Verdict Classes
| Before | After |
|--------|-------|
| SUPPORTED | SUPPORTED |
| UNSUPPORTED | UNSUPPORTED |
| OVERSTATED | *(removed — SciFact doesn't have this)* |
| INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE |

### Code Changes
| File | Change |
|------|--------|
| `src/pipelines/configurable.py` | Simplified system prompt for 3-class scientific claims (removed OVERSTATED detection, entity/intensity matching) |
| `src/shared/schema.py` | Removed OVERSTATED from valid verdicts |
| `src/shared/vector_store.py` | Fixed batch size limit (ChromaDB crashes on >5,461 chunks at once) |
| `scripts/build_scifact_dataset.py` | **New** — pulls SciFact from HuggingFace, converts to our format |

### Removed
- `results/figures/` — old POC figures (no longer relevant)

## E1 Results Comparison

| Metric | PUBHEALTH (main) | SciFact (this branch) |
|--------|-----------------|----------------------|
| Overall accuracy | 33.0% | **81.0%** |
| SUPPORTED | 17.1% | **85.0%** |
| UNSUPPORTED | 17.8% | **84.0%** |
| INSUFFICIENT_EVIDENCE | 84.1% (biased) | **74.0%** |

## How to Run

```bash
# Switch to this branch
git checkout feat/scifact-claims

# Install dependencies (if needed)
pip install rank_bm25 anthropic datasets

# Run an experiment (first run auto-indexes corpus into ChromaDB, ~2-5 min)
python3 -m src.experiment_runner E1    # fixed chunking
python3 -m src.experiment_runner E2    # semantic chunking
python3 -m src.experiment_runner E3    # section-aware chunking
python3 -m src.experiment_runner E4    # recursive chunking

# Results saved to results/experiments/E1.json etc.
```

## Notes
- ChromaDB index auto-builds on first run per chunking strategy — no manual indexing needed
- Index is in `.gitignore` (~300-400MB per strategy), not pushed to repo
- Experiment runner auto-resumes if interrupted — safe to Ctrl+C and re-run
- To regenerate the dataset: `python3 scripts/build_scifact_dataset.py --claims-per-class 100`
