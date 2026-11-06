import pytest

from fakes import HashEmbedder
from hr_assistant.config import get_settings
from hr_assistant.rag.ingest import build_index
from hr_assistant.rag.retriever import HRRetriever
from hr_assistant.rag.store import TypesenseStore, make_typesense_client

pytestmark = pytest.mark.integration
TEST_COLLECTION = "hr_docs_test"


@pytest.fixture
def store():
    store = TypesenseStore(make_typesense_client(get_settings()), TEST_COLLECTION, dim=768)
    if not store.is_reachable():
        pytest.skip("Typesense tidak bisa dihubungi")
    yield store
    store.drop()


@pytest.fixture
def retriever(store, sample_docs):
    embedder = HashEmbedder()
    build_index(sample_docs, store, embedder)
    return HRRetriever(store, embedder, top_k=4, min_score=0.3)


def test_build_index_counts_chunks(store, sample_docs):
    n = build_index(sample_docs, store, HashEmbedder())
    assert n == store.count() == 3


def test_build_index_is_idempotent(store, sample_docs):
    first = build_index(sample_docs, store, HashEmbedder())
    assert build_index(sample_docs, store, HashEmbedder()) == first == store.count()


def test_search_ranks_matching_section_first(retriever):
    hits = retriever.search("berapa hari cuti tahunan")
    assert hits[0].document.metadata["section"] == "Kebijakan Cuti > Cuti Tahunan"
    assert hits[0].document.metadata["source"] == "kebijakan_cuti.md"
    assert 0.0 <= hits[-1].similarity <= hits[0].similarity <= 1.0


def test_search_respects_k(retriever):
    assert len(retriever.search("cuti", k=1)) == 1


def test_relevant_filters_by_min_score(retriever):
    assert retriever.relevant("zzqx kuantum nebula") == []


def test_keyword_match_found_by_hybrid_search(retriever):
    hits = retriever.search("dokter")
    assert hits[0].document.metadata["section"] == "Kebijakan Cuti > Cuti Sakit"
    assert hits[0].text_match > 0
