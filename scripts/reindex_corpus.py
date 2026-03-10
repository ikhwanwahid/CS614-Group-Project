"""Re-index the corpus into ChromaDB for all chunking strategies.

Deletes existing collections and re-creates them from scratch.
Run this after updating the corpus or changing chunking parameters.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.shared.vector_store import get_chroma_client, get_or_create_collection, add_documents, delete_collection
from src.shared.corpus_loader import load_corpus
from src.chunking import chunk_corpus

STRATEGIES = ["fixed", "semantic", "section_aware", "recursive"]

COLLECTION_NAMES = {
    "fixed": "health_corpus",
    "semantic": "health_corpus_semantic",
    "section_aware": "health_corpus_section_aware",
    "recursive": "health_corpus_recursive",
}


def main():
    strategies = sys.argv[1:] if len(sys.argv) > 1 else STRATEGIES

    corpus = load_corpus(str(project_root / "data" / "corpus.json"))
    print(f"Loaded {len(corpus)} articles from corpus")

    client = get_chroma_client(persist_dir=str(project_root / "data" / "corpus" / "embeddings" / "chroma_db"))

    for strategy in strategies:
        if strategy not in COLLECTION_NAMES:
            print(f"Unknown strategy: {strategy}, skipping")
            continue

        name = COLLECTION_NAMES[strategy]
        print(f"\n--- {strategy} (collection: {name}) ---")

        # Delete old collection
        delete_collection(client, name)
        print(f"  Deleted old collection")

        # Chunk corpus
        chunks = chunk_corpus(corpus, strategy=strategy)
        print(f"  Created {len(chunks)} chunks")

        # Create fresh collection and index
        collection = get_or_create_collection(client, collection_name=name)
        add_documents(collection, chunks)
        print(f"  Indexed {collection.count()} documents")

    print("\nDone!")


if __name__ == "__main__":
    main()
