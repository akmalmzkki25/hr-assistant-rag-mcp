from fakes import HashEmbedder
from hr_assistant.rag import build_index
from hr_assistant.vector_store import ChromaStore


def test_build_index_counts_chunks(tmp_path, sample_docs):
    store = ChromaStore(tmp_path / "idx", "hr_docs")
    n = build_index(sample_docs, store, HashEmbedder())
    assert n == store.count() > 0


def test_build_index_is_idempotent(tmp_path, sample_docs):
    store = ChromaStore(tmp_path / "idx", "hr_docs")
    first = build_index(sample_docs, store, HashEmbedder())
    assert build_index(sample_docs, store, HashEmbedder()) == first == store.count()


def test_search_returns_top_k_sorted(retriever):
    hits = retriever.search("berapa hari cuti tahunan", k=3)
    assert len(hits) == 3
    assert hits[0].chunk.section.endswith("Cuti Tahunan")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_keeps_chunk_metadata(retriever):
    top = retriever.search("kerja remote per minggu", k=1)[0]
    assert top.chunk.source == "kerja_remote.md"
    assert "2 hari per minggu" in top.chunk.text


def test_relevant_filters_by_min_score(retriever):
    assert retriever.relevant("zzqx kuantum nebula") == []
