from langchain_core.documents import Document

from hr_assistant.rag.retriever import HRRetriever
from hr_assistant.rag.store import SearchHit


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, vector, k):
        self.calls.append(k)
        return self.hits


class FakeEmbeddings:
    def embed_query(self, text):
        return [1.0]


def hit(section: str, similarity: float, text_match: int = 0) -> SearchHit:
    return SearchHit(Document(page_content=section, metadata={"section": section, "source": "a.md", "chunk_id": section}), similarity, text_match)


def test_keyword_only_noise_is_reranked_below_semantic_matches():
    store = FakeStore([hit("Cuti Sakit", 0.0, text_match=10**18), hit("Cuti Tahunan", 0.79), hit("Cuti Khusus", 0.70)])
    retriever = HRRetriever(store, FakeEmbeddings(), top_k=2, min_score=0.6)
    assert [h.document.metadata["section"] for h in retriever.search("berapa jatah cuti tahunan")] == ["Cuti Tahunan", "Cuti Khusus"]


def test_candidate_pool_is_wider_than_top_k():
    store = FakeStore([])
    HRRetriever(store, FakeEmbeddings(), top_k=4).search("x")
    assert store.calls == [20]


def test_relevant_applies_min_score():
    store = FakeStore([hit("A", 0.72), hit("B", 0.69)])
    retriever = HRRetriever(store, FakeEmbeddings(), top_k=4, min_score=0.70)
    assert [h.document.metadata["section"] for h in retriever.relevant("q")] == ["A"]
