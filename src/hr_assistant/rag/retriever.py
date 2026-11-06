"""Retrieval hybrid: kandidat dari kata kunci + vektor, lalu diurutkan ulang berdasarkan kemiripan vektor."""

from langchain_core.embeddings import Embeddings

from hr_assistant.rag.store import SearchHit, TypesenseStore

CANDIDATE_MULTIPLIER = 5


class HRRetriever:
    def __init__(self, store: TypesenseStore, embeddings: Embeddings, top_k: int = 4, min_score: float = 0.7):
        self.store = store
        self.embeddings = embeddings
        self.top_k = top_k
        self.min_score = min_score

    def search(self, query: str, k: int | None = None) -> list[SearchHit]:
        k = k or self.top_k
        candidates = self.store.search(query, self.embeddings.embed_query(query), k * CANDIDATE_MULTIPLIER)
        # Keyword matching widens recall, but Indonesian question words ("berapa", "apa") match almost every
        # chunk, so the final order comes from semantic similarity rather than Typesense's rank fusion.
        return sorted(candidates, key=lambda hit: hit.similarity, reverse=True)[:k]

    def relevant(self, query: str) -> list[SearchHit]:
        return [hit for hit in self.search(query) if hit.similarity >= self.min_score]
