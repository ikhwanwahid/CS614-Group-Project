"""Section-aware chunking strategy.

Parses PubMed abstract sections (Background, Methods, Results, Conclusions)
and chunks per-section, keeping methodological context together.
"""

import re
from src.chunking.fixed import chunk_text

# Define known section labels and their common variants in PubMed abstracts.
# https://www.researchgate.net/figure/Headings-of-Structured-Abstracts_tbl1_23998674
_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "AIM": (
        "aim",
        "aims",
        "goal",
        "goals",
        "objective",
        "objectives",
        "purpose",
        "hypothesis",
        "hypotheses",
        "introduction",
        "background",
        "context",
        "rationale",
    ),
    "INTERVENTION": (
        "intervention",
        "interventions",
        "interventions of the study",
    ),
    "PARTICIPANTS": (
        "participant",
        "participants",
        "population",
        "patients",
        "subjects",
        "sample",
    ),
    "OUTCOME_MEASURES": (
        "outcome measures",
        "outcome measure",
        "primary outcome parameters",
        "main variables",
        "measures",
        "measurements",
        "assessments",
    ),
    "METHOD": (
        "method",
        "methods",
        "materials",
        "materials and methods",
        "study design",
        "design",
        "setting",
        "procedures",
        "process",
        "methodology",
        "research design",
    ),
    "RESULTS": (
        "results",
        "findings",
        "outcomes",
        "main outcomes and results",
    ),
    "CONCLUSION": (
        "conclusion",
        "conclusions",
        "conclusion and clinical relevance",
        "clinical implications",
        "discussion",
        "interpretation",
        "significance",
    ),
}

_LABEL_BY_VARIANT = {
    variant.casefold(): label
    for label, variants in _SECTION_ALIASES.items()
    for variant in variants
}

_VARIANT_PATTERN = "|".join(
    re.escape(variant)
    for variant in sorted(_LABEL_BY_VARIANT, key=len, reverse=True)
)

_HTML_TAG_PATTERN = re.compile(r"</?[^>]+>")

# Match known structured-abstract headings whether they appear on a new line,
# inline, or wrapped in simple HTML formatting tags.
_SECTION_PATTERN = re.compile(
    rf"(?<!\w)(?P<label>{_VARIANT_PATTERN})\s*:\s*",
    re.IGNORECASE,
)


def _normalize_abstract(abstract: str) -> str:
    """Remove simple HTML tags that commonly wrap PubMed section headers."""
    return _HTML_TAG_PATTERN.sub("", abstract)


def split_into_sections(abstract: str) -> list[tuple[str, str]]:
    """Split a structured abstract into (section_label, text) pairs.

    Returns a list of (label, text) tuples. If no section markers are found,
    returns a single ('', abstract) tuple so callers can fall back to
    fixed chunking.
    """
    normalized_abstract = _normalize_abstract(abstract)
    matches = list(_SECTION_PATTERN.finditer(normalized_abstract))
    if not matches:
        return [("", normalized_abstract)]

    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        raw_label = match.group("label")
        label = _LABEL_BY_VARIANT[raw_label.casefold()]
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized_abstract)
        text = normalized_abstract[start:end].strip()
        if text:
            sections.append((label, text))
    return sections


def chunk_corpus_section_aware(corpus: list[dict], chunk_size: int = 200, overlap: int = 50) -> list[dict]:
    """Chunk all abstracts by detected section boundaries.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
    """
    chunked: list[dict] = []
    for article in corpus:
        abstract = article.get("abstract", "") or ""
        sections = split_into_sections(abstract)

        chunk_index = 0
        for section_label, section_text in sections:
            for text in chunk_text(section_text, chunk_size=chunk_size, overlap=overlap):
                chunked.append({
                    "pmid": article["pmid"],
                    "title": article["title"],
                    "chunk_index": chunk_index,
                    "text": text,
                    "metadata": {"section": section_label.lower() if section_label else "unstructured"},
                })
                chunk_index += 1
    return chunked
