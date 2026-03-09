"""Recursive chunking strategy with metadata enrichment.

Uses LangChain's RecursiveCharacterTextSplitter with paragraph -> sentence ->
character fallback, then enriches each chunk with extracted metadata
(study type, sample size, publication year).
"""


def chunk_corpus_recursive(corpus: list[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """Chunk all abstracts using recursive splitting with metadata enrichment.

    Args:
        corpus: List of article dicts with 'pmid', 'title', 'abstract'.
        chunk_size: Target character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk dicts with 'pmid', 'title', 'chunk_index', 'text'.
        Chunks include additional metadata: 'study_type', 'sample_size', 'year'.
    """
    raise NotImplementedError(
        "Recursive chunking not yet implemented — RAG pair (Members 2 & 3).\n"
        "Approach: use LangChain RecursiveCharacterTextSplitter with "
        "paragraph/sentence/character fallback. Enrich each chunk with metadata "
        "extracted via regex: study type (RCT, meta-analysis, observational), "
        "sample size (n=...), publication year."
    )
