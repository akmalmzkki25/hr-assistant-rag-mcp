"""Ingest dokumen ke vector store dan retrieval top-k."""

from pathlib import Path

from hr_assistant.chunking import chunk_markdown, load_documents
from hr_assistant.embeddings import Embedder
from hr_assistant.vector_store import ChromaStore, RetrievedChunk

BATCH_SIZE = 64


def build_index(docs_dir: Path, store: ChromaStore, embedder: Embedder) -> int:
    chunks = [chunk for name, text in load_documents(docs_dir) for chunk in chunk_markdown(text, source=name)]
    store.reset()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        store.upsert(batch, embedder.embed_documents([c.text for c in batch]))
    return len(chunks)


class Retriever:
    def __init__(self, store: ChromaStore, embedder: Embedder, top_k: int = 4, min_score: float = 0.45):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        return self.store.query(self.embedder.embed_query(query), k or self.top_k)

    def relevant(self, query: str) -> list[RetrievedChunk]:
        return [hit for hit in self.search(query) if hit.score >= self.min_score]
