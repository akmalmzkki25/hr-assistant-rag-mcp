"""Penyimpanan vektor lokal berbasis Chroma."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb

from hr_assistant.chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class ChromaStore:
    def __init__(self, path: Path, collection: str):
        self._client = chromadb.PersistentClient(path=str(path))
        self._name = collection
        self._collection = self._open()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "section": c.section} for c in chunks],
        )

    def query(self, vector: Sequence[float], k: int) -> list[RetrievedChunk]:
        available = self.count()
        if available == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=min(k, available),
            include=["documents", "metadatas", "distances"],
        )
        hits = zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        return [
            # Chroma reports cosine distance, so similarity is 1 - distance.
            RetrievedChunk(Chunk(id=i, text=doc, source=meta["source"], section=meta["section"]), 1.0 - dist)
            for i, doc, meta, dist in hits
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._open()

    def _open(self):
        return self._client.get_or_create_collection(self._name, configuration={"hnsw": {"space": "cosine"}})
