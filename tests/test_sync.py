"""Ingest inkremental: hanya dokumen baru atau berubah yang di-embed ulang."""

import pytest

from fakes import HashEmbedder
from hr_assistant.config import get_settings
from hr_assistant.rag.embeddings import EmbeddingUnavailableError
from hr_assistant.rag.ingest import build_index, sync_index
from hr_assistant.rag.store import TypesenseStore, make_typesense_client

pytestmark = pytest.mark.integration
TEST_COLLECTION = "hr_docs_sync_test"


class CountingEmbedder(HashEmbedder):
    def __init__(self, fail_after: int | None = None):
        super().__init__()
        self.embedded: list[str] = []
        self.fail_after = fail_after

    def embed_documents(self, texts):
        if self.fail_after is not None and len(self.embedded) >= self.fail_after:
            raise EmbeddingUnavailableError("429 RESOURCE_EXHAUSTED: kuota habis")
        self.embedded += texts
        return super().embed_documents(texts)


@pytest.fixture
def store():
    store = TypesenseStore(make_typesense_client(get_settings()), TEST_COLLECTION, dim=768)
    if not store.is_reachable():
        pytest.skip("Typesense tidak bisa dihubungi")
    store.drop()
    yield store
    store.drop()


def sources_in(store) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in store.indexed_chunks():
        counts[record.source] = counts.get(record.source, 0) + 1
    return counts


def test_first_sync_adds_everything(store, sample_docs):
    embedder = CountingEmbedder()
    report = sync_index(sample_docs, store, embedder)
    assert (report.added, report.updated, report.removed, report.unchanged) == (
        ["kebijakan_cuti.md", "kerja_remote.md"], [], [], []
    )
    assert report.embedded_chunks == len(embedder.embedded) == store.count() == 3


def test_second_sync_without_changes_embeds_nothing(store, sample_docs):
    sync_index(sample_docs, store, CountingEmbedder())
    embedder = CountingEmbedder()
    report = sync_index(sample_docs, store, embedder)
    assert report.unchanged == ["kebijakan_cuti.md", "kerja_remote.md"]
    assert embedder.embedded == [] and report.embedded_chunks == 0


def test_changed_file_is_reembedded_and_stale_chunks_removed(store, sample_docs):
    sync_index(sample_docs, store, CountingEmbedder())
    (sample_docs / "kebijakan_cuti.md").write_text("# Kebijakan Cuti\n\n## Cuti Tahunan\nJatah cuti 14 hari.\n", encoding="utf-8")
    embedder = CountingEmbedder()
    report = sync_index(sample_docs, store, embedder)
    assert (report.updated, report.unchanged) == (["kebijakan_cuti.md"], ["kerja_remote.md"])
    assert len(embedder.embedded) == 1
    assert sources_in(store) == {"kebijakan_cuti.md": 1, "kerja_remote.md": 1}
    hits = store.search("cuti", HashEmbedder().embed_query("cuti"), k=4)
    assert any("14 hari" in hit.document.page_content for hit in hits)
    assert not any("12 hari" in hit.document.page_content for hit in hits)


def test_deleted_file_is_removed_from_index(store, sample_docs):
    sync_index(sample_docs, store, CountingEmbedder())
    (sample_docs / "kerja_remote.md").unlink()
    report = sync_index(sample_docs, store, CountingEmbedder())
    assert report.removed == ["kerja_remote.md"]
    assert sources_in(store) == {"kebijakan_cuti.md": 2}


def test_new_file_is_added_without_touching_others(store, sample_docs):
    sync_index(sample_docs, store, CountingEmbedder())
    (sample_docs / "lembur.md").write_text("# Lembur\n\n## Hari Libur\nUpah lembur 2 kali.\n", encoding="utf-8")
    embedder = CountingEmbedder()
    report = sync_index(sample_docs, store, embedder)
    assert report.added == ["lembur.md"] and len(embedder.embedded) == 1


def test_changed_embedding_fingerprint_reembeds_everything(store, sample_docs):
    sync_index(sample_docs, store, CountingEmbedder(), fingerprint="model-a:768")
    report = sync_index(sample_docs, store, CountingEmbedder(), fingerprint="model-b:768")
    assert report.updated == ["kebijakan_cuti.md", "kerja_remote.md"]


def test_quota_error_leaves_index_as_it_was(store, sample_docs, monkeypatch):
    monkeypatch.setattr("hr_assistant.rag.ingest.BATCH_SIZE", 1)
    sync_index(sample_docs, store, CountingEmbedder())
    (sample_docs / "kebijakan_cuti.md").write_text("# Kebijakan Cuti\n\n## Cuti Tahunan\nJatah 14 hari.\n", encoding="utf-8")
    (sample_docs / "kerja_remote.md").write_text("# Remote\n\n## Umum\nRemote 3 hari.\n", encoding="utf-8")
    with pytest.raises(EmbeddingUnavailableError):
        sync_index(sample_docs, store, CountingEmbedder(fail_after=1))
    assert sources_in(store) == {"kebijakan_cuti.md": 2, "kerja_remote.md": 1}
    # Nothing was written, so the next run still sees both files as changed.
    assert sorted(sync_index(sample_docs, store, CountingEmbedder()).updated) == ["kebijakan_cuti.md", "kerja_remote.md"]


def test_index_from_old_schema_is_rebuilt(store, sample_docs):
    store._client.collections.create(
        {"name": TEST_COLLECTION, "fields": [{"name": "content", "type": "string"}, {"name": "embedding", "type": "float[]", "num_dim": 768}]}
    )
    report = sync_index(sample_docs, store, CountingEmbedder())
    assert report.rebuilt and report.added == ["kebijakan_cuti.md", "kerja_remote.md"]
    assert store.count() == 3


def test_full_build_then_sync_sees_no_changes(store, sample_docs):
    build_index(sample_docs, store, CountingEmbedder())
    report = sync_index(sample_docs, store, CountingEmbedder())
    assert report.unchanged == ["kebijakan_cuti.md", "kerja_remote.md"]


def test_deleted_file_in_subfolder_is_really_removed(store, sample_docs):
    (sample_docs / "sub").mkdir()
    (sample_docs / "sub" / "plain.md").write_text("# Sub\n\n## Bagian\nIsi subfolder.\n", encoding="utf-8")
    sync_index(sample_docs, store, CountingEmbedder())
    (sample_docs / "sub" / "plain.md").unlink()
    report = sync_index(sample_docs, store, CountingEmbedder())
    assert report.removed == ["sub/plain.md"]
    assert "sub/plain.md" not in sources_in(store)
    assert sync_index(sample_docs, store, CountingEmbedder()).removed == []


def test_shrunk_file_in_subfolder_drops_old_chunks(store, sample_docs):
    (sample_docs / "sub").mkdir()
    path = sample_docs / "sub" / "dua.md"
    path.write_text("# Dua\n\n## A\nSatu.\n\n## B\nDua.\n", encoding="utf-8")
    sync_index(sample_docs, store, CountingEmbedder())
    path.write_text("# Dua\n\n## A\nSatu saja.\n", encoding="utf-8")
    sync_index(sample_docs, store, CountingEmbedder())
    assert sources_in(store)["sub/dua.md"] == 1


def test_quota_error_during_schema_rebuild_keeps_old_collection(store, sample_docs):
    store._client.collections.create(
        {"name": TEST_COLLECTION, "fields": [{"name": "content", "type": "string"}, {"name": "embedding", "type": "float[]", "num_dim": 768}]}
    )
    store._client.collections[TEST_COLLECTION].documents.create({"id": "lama", "content": "lama", "embedding": [0.1] * 768})
    with pytest.raises(EmbeddingUnavailableError):
        sync_index(sample_docs, store, CountingEmbedder(fail_after=0))
    assert store.count() == 1 and store.schema_state() == "outdated"


def test_changed_chunk_size_reembeds_everything(store, sample_docs, monkeypatch):
    sync_index(sample_docs, store, CountingEmbedder())
    monkeypatch.setattr("hr_assistant.rag.chunking.CHUNK_SIZE", 900)
    report = sync_index(sample_docs, store, CountingEmbedder())
    assert report.updated == ["kebijakan_cuti.md", "kerja_remote.md"]
