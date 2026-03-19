"""Shared helpers for chunk generation, metadata, and artifact export."""

from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

PROCESSED_CORPUS_DIR = Path("data/corpus/processed")


def abstract_to_text(abstract: Any) -> str:
    """Convert a SciFact abstract field into plain text for chunking."""
    if isinstance(abstract, list):
        return " ".join(str(sentence).strip() for sentence in abstract if str(sentence).strip())
    if abstract is None:
        return ""
    return str(abstract)


def build_base_metadata(article: dict, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build shared, Chroma-safe metadata from the SciFact corpus article."""
    doc_id = str(article["doc_id"])
    metadata: dict[str, Any] = {
        "doc_id": doc_id,
        "structured": bool(article.get("structured", False)),
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
    doc_id = str(article["doc_id"])
    return {
        "doc_id": doc_id,
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
            fieldnames=[
                "doc_id",
                "title",
                "chunk_index",
                "text",
                "structured",
                "section",
            ],
        )
        writer.writeheader()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            writer.writerow(
                {
                    "doc_id": chunk.get("doc_id", metadata.get("doc_id", "")),
                    "title": chunk.get("title", ""),
                    "chunk_index": chunk.get("chunk_index", ""),
                    "text": chunk.get("text", ""),
                    "structured": metadata.get("structured", False),
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
