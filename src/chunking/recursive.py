"""Recursive chunking strategy with metadata enrichment.

Uses LangChain's RecursiveCharacterTextSplitter with paragraph -> sentence ->
character fallback, then enriches each chunk with extracted metadata
(study type, sample size, publication year).
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Separators: paragraph break, sentence end, space, then individual characters
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

# Patterns for study-type classification (order matters – first match wins)
_STUDY_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("meta-analysis",       re.compile(r"meta[\-\s]?analysis|systematic review", re.I)),
    ("randomized controlled trial", re.compile(r"randomi[sz]ed.{0,20}(controlled|clinical)?\s*trial|RCT\b", re.I)),
    ("clinical trial",      re.compile(r"clinical\s+trial|phase\s+[I1-4]+\b", re.I)),
    ("cohort study",        re.compile(r"cohort\s+stud|longitudinal\s+stud|prospective\s+stud|retrospective\s+stud", re.I)),
    ("case-control study",  re.compile(r"case[\-\s]control", re.I)),
    ("cross-sectional",     re.compile(r"cross[\-\s]sectional", re.I)),
    ("in vitro",            re.compile(r"\bin\s+vitro\b", re.I)),
    ("animal study",        re.compile(r"\bin\s+vivo\b|\bmice\b|\bmouse\b|\bmurine\b|\brat\b", re.I)),
    ("review",              re.compile(r"\breview\b", re.I)),
]

# Patterns for sample-size extraction (returns first numeric match)
_SAMPLE_SIZE_RE = re.compile(
    r"(?:n\s*=\s*|sample\s+(?:size|of)\s+|(\d[\d,]+)\s+participants"
    r"|(\d[\d,]+)\s+patients|(\d[\d,]+)\s+subjects"
    r"|enrolled\s+(\d[\d,]+))"
    r"[\s,]*(\d[\d,]*)?",
    re.I,
)


def _extract_study_type(text: str) -> str:
    """Return the best-matching study type label, or 'unknown'."""
    for label, pattern in _STUDY_TYPE_PATTERNS:
        if pattern.search(text):
            return label
    return "unknown"


def _extract_sample_size(text: str) -> int | None:
    """Return the first plausible sample size found, or None."""
    m = _SAMPLE_SIZE_RE.search(text)
    if m:
        # Pick the first captured group that is not None, or the tail group
        for g in m.groups():
            if g is not None:
                return int(g.replace(",", ""))
    return None


def chunk_corpus_recursive(
    corpus: list[dict], chunk_size: int = 200, overlap: int = 50
) -> list[dict]:
    """Chunk all abstracts using recursive splitting.

    Each chunk is returned as a dict enriched with metadata extracted from the
    source article: *study_type*, *sample_size*, and *publication_year*.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract',
                and optionally 'year'.
        chunk_size: Approximate token budget per chunk.
        overlap: Approximate token overlap between consecutive chunks.

    Returns:
        List of dicts with 'pmid', 'title', 'chunk_index', 'text', 'metadata'.
    """

    # Keep API in token-like units for consistency with other chunkers.
    char_chunk_size = int(chunk_size * 4)
    char_overlap = int(overlap * 4)

    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=char_chunk_size,
        chunk_overlap=char_overlap,
        length_function=len,
    )

    chunks: list[dict] = []
    for article in corpus:
        abstract = article.get("abstract", "") or ""
        title = article.get("title", "")
        combined_text = f"{title} {abstract}"

        study_type = _extract_study_type(combined_text)
        sample_size = _extract_sample_size(combined_text)
        publication_year = article.get("year")

        raw_chunks = splitter.split_text(abstract)
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append({
                "pmid": article["pmid"],
                "title": title,
                "chunk_index": i,
                "text": chunk_text,
                "metadata": {
                    "study_type": study_type,
                    "sample_size": sample_size,
                    "publication_year": publication_year,
                },
            })
    return chunks
