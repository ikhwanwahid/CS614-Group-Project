"""Build test claims and corpus from SciFact dataset.

SciFact (AllenAI, EMNLP 2020) contains expert-written scientific claims
paired with scientific abstracts as evidence. Labels: SUPPORT, CONTRADICT, or
no evidence (NEI).

Mapping to our verdict schema:
    SUPPORT    -> SUPPORTED
    CONTRADICT -> UNSUPPORTED
    '' (NEI)   -> INSUFFICIENT_EVIDENCE

Outputs:
    data/test_claims.json   — balanced claims (sampled per class)
    data/corpus.json        — SciFact corpus (5,183 abstracts)

Usage:
    python scripts/build_scifact_dataset.py
    python scripts/build_scifact_dataset.py --claims-per-class 100
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42

LABEL_MAP = {
    "SUPPORT": "SUPPORTED",
    "CONTRADICT": "UNSUPPORTED",
    "": "INSUFFICIENT_EVIDENCE",
}

def build_corpus(corpus_ds) -> list[dict]:
    """Convert SciFact corpus to our format."""
    articles = []
    for doc in corpus_ds["train"]:
        abstract_text = " ".join(doc["abstract"])
        articles.append({
            "doc_id": str(doc["doc_id"]),
            "title": doc["title"],
            "abstract": abstract_text,
            "structured": doc["structured"],
        })
    return articles


def build_claims(claims_ds, claims_per_class: int | None = None) -> list[dict]:
    """Convert SciFact claims to our format, deduplicated and balanced."""
    # Combine train + validation (test has no labels)
    all_entries = list(claims_ds["train"]) + list(claims_ds["validation"])

    # Deduplicate by claim text (prefer labeled over unlabeled)
    unique = {}
    for c in all_entries:
        text = c["claim"]
        if text not in unique:
            unique[text] = c
        elif not unique[text]["evidence_label"] and c["evidence_label"]:
            unique[text] = c

    # Convert to our format
    claims_by_verdict = {"SUPPORTED": [], "UNSUPPORTED": [], "INSUFFICIENT_EVIDENCE": []}
    for c in unique.values():
        verdict = LABEL_MAP[c["evidence_label"]]
        claims_by_verdict[verdict].append({
            "claim": c["claim"],
            "expected_verdict": verdict,
            "source": "scifact",
            "source_id": str(c["id"]),
            "evidence_doc_id": c["evidence_doc_id"] if c["evidence_doc_id"] else None,
            "evidence_sentences": c["evidence_sentences"] if c["evidence_sentences"] else None,
            "cited_doc_ids": c["cited_doc_ids"] if c["cited_doc_ids"] else None,
        })

    # Balance classes
    if claims_per_class is None:
        # Use the smallest class size
        claims_per_class = min(len(v) for v in claims_by_verdict.values())

    random.seed(SEED)
    balanced = []
    for verdict, pool in claims_by_verdict.items():
        n = min(claims_per_class, len(pool))
        sampled = random.sample(pool, n)
        balanced.extend(sampled)

    random.shuffle(balanced)
    return balanced


def main():
    parser = argparse.ArgumentParser(description="Build SciFact dataset for our pipeline")
    parser.add_argument(
        "--claims-per-class",
        type=int,
        default=100,
        help="Claims per verdict class (default: 100, yielding 300 total across 3 classes)",
    )
    args = parser.parse_args()

    print("Loading SciFact from HuggingFace...")
    claims_ds = load_dataset("allenai/scifact", "claims", trust_remote_code=True)
    corpus_ds = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)

    # Build corpus
    print("Building corpus...")
    corpus = build_corpus(corpus_ds)
    corpus_path = PROJECT_ROOT / "data" / "corpus.json"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"  Saved {len(corpus)} articles to {corpus_path.name}")

    # Build claims
    print("Building claims...")
    claims = build_claims(claims_ds, args.claims_per_class)
    claims_path = PROJECT_ROOT / "data" / "test_claims.json"
    with open(claims_path, "w") as f:
        json.dump(claims, f, indent=2)

    # Report
    verdicts = Counter(c["expected_verdict"] for c in claims)
    print(f"  Saved {len(claims)} claims to {claims_path.name}")
    print(f"  Verdict distribution:")
    for v, n in sorted(verdicts.items()):
        print(f"    {v}: {n}")

    # Show sample claims per class
    for v in ["SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"]:
        samples = [c for c in claims if c["expected_verdict"] == v][:3]
        print(f"\n  Sample {v} claims:")
        for s in samples:
            print(f"    - {s['claim'][:100]}")

    print(f"\n  Corpus: {len(corpus)} abstracts")
    print(f"  Claims: {len(claims)} ({len(verdicts)} verdict classes)\n")

if __name__ == "__main__":
    main()
