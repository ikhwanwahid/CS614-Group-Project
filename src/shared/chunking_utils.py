"""Shared helpers for chunk generation, metadata, and artifact export."""

from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

PROCESSED_CORPUS_DIR = Path("data/corpus/processed")


def _stringify_authors(authors: Any) -> str:
    """Convert a corpus authors field into a readable scalar value."""
    if isinstance(authors, list):
        return ", ".join(str(author).strip() for author in authors if str(author).strip())
    if authors is None:
        return ""
    return str(authors).strip()


def build_base_metadata(article: dict, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build shared, Chroma-safe metadata from the raw corpus article."""
    authors_text = _stringify_authors(article.get("authors", []))
    metadata: dict[str, Any] = {
        "source": str(article.get("source", "") or ""),
        "year": str(article.get("year", "") or ""),
        "authors": authors_text,
        "author_count": len(article.get("authors", [])) if isinstance(article.get("authors"), list) else 0,
    }
    if extra:
        metadata.update(extra)
    return metadata


def build_chunk_record(
    article: dict,
    chunk_index: int,
    text: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standard chunk dict used by all chunking strategies."""
    return {
        "pmid": str(article["pmid"]),
        "title": article.get("title", ""),
        "chunk_index": chunk_index,
        "text": text,
        "metadata": build_base_metadata(article, extra=extra_metadata),
    }


def get_chunk_artifact_paths(strategy: str) -> dict[str, Path]:
    """Return the output paths used for persisted chunk artifacts."""
    base = PROCESSED_CORPUS_DIR / strategy
    return {
        "dir": base,
        "json": base / "chunks.json",
        "csv": base / "chunks.csv",
        "manifest": base / "manifest.json",
    }


def chunk_artifacts_exist(strategy: str) -> bool:
    """Whether persisted chunk artifacts exist for the strategy."""
    paths = get_chunk_artifact_paths(strategy)
    return paths["json"].exists() and paths["csv"].exists() and paths["manifest"].exists()


def clear_chunk_artifacts(strategy: str) -> None:
    """Delete persisted chunk artifacts for a strategy."""
    paths = get_chunk_artifact_paths(strategy)
    for key in ("json", "csv", "manifest"):
        if paths[key].exists():
            paths[key].unlink()
    if paths["dir"].exists() and not any(paths["dir"].iterdir()):
        paths["dir"].rmdir()


def export_chunk_artifacts(
    strategy: str,
    chunks: list[dict],
    corpus_size: int,
    parameters: dict[str, Any] | None = None,
) -> None:
    """Persist chunk outputs in JSON and CSV for easy inspection."""
    paths = get_chunk_artifact_paths(strategy)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    with open(paths["json"], "w", encoding="utf-8") as handle:
        json.dump(chunks, handle, indent=2, ensure_ascii=False)

    with open(paths["csv"], "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pmid", "title", "chunk_index", "text", "source", "year", "authors", "author_count", "section"],
        )
        writer.writeheader()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            writer.writerow(
                {
                    "pmid": chunk.get("pmid", ""),
                    "title": chunk.get("title", ""),
                    "chunk_index": chunk.get("chunk_index", ""),
                    "text": chunk.get("text", ""),
                    "source": metadata.get("source", ""),
                    "year": metadata.get("year", ""),
                    "authors": metadata.get("authors", ""),
                    "author_count": metadata.get("author_count", 0),
                    "section": metadata.get("section", ""),
                }
            )

    manifest = {
        "strategy": strategy,
        "corpus_size": corpus_size,
        "chunk_count": len(chunks),
        "parameters": parameters or {},
        "artifact_files": {
            "json": str(paths["json"]),
            "csv": str(paths["csv"]),
        },
        "content_hash": sha256(json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
    }
    with open(paths["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
