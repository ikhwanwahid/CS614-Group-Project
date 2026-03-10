"""Section-aware chunking strategy.

Parses PubMed abstract sections (Background, Methods, Results, Conclusions)
and chunks per-section, keeping methodological context together.
"""

import re

from src.chunking.fixed import chunk_text

# Common PubMed abstract section headers
SECTION_RE = re.compile(
    r"\b("
    r"BACKGROUND|INTRODUCTION|CONTEXT|"
    r"OBJECTIVE|OBJECTIVES|AIM|AIMS|PURPOSE|"
    r"METHODS|MATERIALS AND METHODS|STUDY DESIGN|DESIGN|SETTING|PARTICIPANTS|"
    r"RESULTS|FINDINGS|OUTCOMES|"
    r"DISCUSSION|"
    r"CONCLUSION|CONCLUSIONS|SUMMARY|IMPLICATIONS"
    r")\s*:",
    re.IGNORECASE,
)

# If a section exceeds this many words, sub-chunk it with fixed strategy
MAX_SECTION_WORDS = 300


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_name, section_text) pairs.

    Returns empty list if fewer than 2 section headers detected.
    """
    matches = list(SECTION_RE.finditer(text))
    if len(matches) < 2:
        return []

    sections = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip().upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))

    return sections


def chunk_corpus_section_aware(corpus: list[dict]) -> list[dict]:
    """Chunk all abstracts by detected section boundaries.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
        Chunks include 'section' in metadata when detected.
    """
    all_chunks = []

    for article in corpus:
        abstract = article.get("abstract", "")
        if not abstract:
            continue

        sections = _split_into_sections(abstract)

        if not sections:
            # Fallback to fixed chunking for unstructured abstracts
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
            chunk_idx = 0
            for section_name, section_text in sections:
                # Sub-chunk large sections
                if len(section_text.split()) > MAX_SECTION_WORDS:
                    sub_chunks = chunk_text(section_text, chunk_size=200, overlap=50)
                    for sc in sub_chunks:
                        all_chunks.append({
                            "pmid": article["pmid"],
                            "title": article["title"],
                            "chunk_index": chunk_idx,
                            "text": sc,
                            "metadata": {"section": section_name.lower()},
                        })
                        chunk_idx += 1
                else:
                    all_chunks.append({
                        "pmid": article["pmid"],
                        "title": article["title"],
                        "chunk_index": chunk_idx,
                        "text": section_text,
                        "metadata": {"section": section_name.lower()},
                    })
                    chunk_idx += 1

    return all_chunks
