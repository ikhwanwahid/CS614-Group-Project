"""Section-aware chunking strategy.

Parses PubMed abstract sections (Background, Methods, Results, Conclusions)
and chunks per-section, keeping methodological context together.
"""


def chunk_corpus_section_aware(corpus: list[dict]) -> list[dict]:
    """Chunk all abstracts by detected section boundaries.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
        Chunks include 'section' in metadata when detected.
    """
    raise NotImplementedError(
        "Section-aware chunking not yet implemented — RAG pair (Members 2 & 3).\n"
        "Approach: detect section markers in PubMed abstracts (BACKGROUND:, "
        "METHODS:, RESULTS:, CONCLUSIONS:, OBJECTIVE:, etc.) and split at "
        "those boundaries. Abstracts without markers fall back to fixed chunking."
    )
