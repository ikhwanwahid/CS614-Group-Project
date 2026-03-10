"""Recursive chunking strategy with metadata enrichment.

Uses LangChain's RecursiveCharacterTextSplitter with paragraph -> sentence ->
character fallback, then enriches each chunk with extracted metadata
(study type, sample size, publication year).
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _extract_metadata(abstract: str, article: dict) -> dict:
    """Extract structured metadata from abstract text."""
    metadata = {"year": article.get("year", "")}

    # Study type detection
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
    elif "review" in lower:
        study_type = "review"
    else:
        study_type = "unknown"
    metadata["study_type"] = study_type

    # Sample size extraction
    match = re.search(r"[Nn]\s*=\s*([\d,]+)", abstract)
    if match:
        metadata["sample_size"] = match.group(1).replace(",", "")

    return metadata


def chunk_corpus_recursive(
    corpus: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """Chunk all abstracts using recursive splitting with metadata enrichment.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.
        chunk_size: Target character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
        Chunks include additional metadata: 'study_type', 'sample_size', 'year'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
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
