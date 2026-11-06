"""Jalur gagal yang harus berhenti dengan pesan jelas, tanpa merusak data yang sudah ada."""

import pytest

from hr_assistant.config import Settings
from hr_assistant.rag.embeddings import EmbeddingUnavailableError
from hr_assistant.rag.ingest import build_index
from hr_assistant.rag.loader import DocumentLoadError, load_documents
from hr_assistant.rag.store import SearchUnavailableError, TypesenseStore, make_typesense_client


class RecordingStore:
    def __init__(self):
        self.events: list[str] = []

    def recreate(self):
        self.events.append("recreate")

    def upsert(self, docs, vectors):
        self.events.append(f"upsert:{len(docs)}")


class QuotaAfterFirstBatch:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls > 1:
            raise EmbeddingUnavailableError("429 RESOURCE_EXHAUSTED: kuota habis")
        return [[0.1] * 3 for _ in texts]


def test_ingest_keeps_old_index_when_embedding_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("hr_assistant.rag.ingest.BATCH_SIZE", 1)
    for name in ("a", "b"):
        (tmp_path / f"{name}.md").write_text(f"# Judul {name}\n\nIsi dokumen {name}.", encoding="utf-8")
    store = RecordingStore()
    with pytest.raises(EmbeddingUnavailableError):
        build_index(tmp_path, store, QuotaAfterFirstBatch())
    assert store.events == []


def test_search_with_typesense_down_raises_search_unavailable():
    client = make_typesense_client(Settings(typesense_url="http://127.0.0.1:9", typesense_api_key="x"))
    with pytest.raises(SearchUnavailableError, match="Typesense"):
        TypesenseStore(client, "hr_docs", dim=3).search("cuti", [0.1, 0.2, 0.3], k=4)


def test_corrupt_pdf_names_the_file(tmp_path):
    (tmp_path / "rusak.pdf").write_bytes(b"bukan pdf")
    with pytest.raises(DocumentLoadError, match="rusak.pdf"):
        load_documents(tmp_path)


def test_non_utf8_text_names_the_file(tmp_path):
    (tmp_path / "latin.txt").write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(DocumentLoadError, match="latin.txt"):
        load_documents(tmp_path)


