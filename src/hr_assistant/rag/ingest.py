"""Ingest: muat dokumen, pecah, embed, lalu simpan ke Typesense, penuh atau hanya yang berubah."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hr_assistant.rag import chunking
from hr_assistant.rag.chunking import split_documents
from hr_assistant.rag.loader import load_documents
from hr_assistant.rag.store import ChunkRecord, TypesenseStore, document_id

BATCH_SIZE = 64
# Bump when chunking or the stored text format changes, so every document is re-embedded once.
INDEX_VERSION = "2"


@dataclass(frozen=True)
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    embedded_chunks: int = 0
    rebuilt: bool = False


def build_index(docs_dir: Path, store: TypesenseStore, embeddings: Embeddings, fingerprint: str = "") -> int:
    """Bangun ulang seluruh index."""
    chunks = _hashed_chunks(load_documents(docs_dir), fingerprint)
    # Embed everything before touching the store, so a quota error mid-run leaves the old index intact.
    vectors = _embed(chunks, embeddings)
    store.recreate()
    _upsert(store, chunks, vectors)
    return len(chunks)


def sync_index(docs_dir: Path, store: TypesenseStore, embeddings: Embeddings, fingerprint: str = "") -> SyncReport:
    """Embed ulang hanya dokumen yang baru atau berubah, dan hapus chunk dari dokumen yang sudah tidak ada.

    fingerprint menandai model dan dimensi embedding; kalau berubah, semua dokumen dianggap berubah.
    """
    chunks = _hashed_chunks(load_documents(docs_dir), fingerprint)
    wanted = _by_source(chunks, lambda c: c.metadata["source"])
    state = store.schema_state()
    if state != "current":
        # Embed before dropping the old collection, so a quota error leaves it as it was.
        vectors = _embed(chunks, embeddings)
        store.recreate()
        _upsert(store, chunks, vectors)
        return SyncReport(added=sorted(wanted), embedded_chunks=len(chunks), rebuilt=state == "outdated")
    indexed = _by_source(store.indexed_chunks(), lambda r: r.source)

    added, updated, unchanged = [], [], []
    for source, source_chunks in sorted(wanted.items()):
        records = indexed.get(source)
        if not records:
            added.append(source)
        elif _is_current(records, source_chunks):
            unchanged.append(source)
        else:
            updated.append(source)
    removed = sorted(set(indexed) - set(wanted))

    to_embed = [chunk for source in added + updated for chunk in wanted[source]]
    vectors = _embed(to_embed, embeddings)
    _upsert(store, to_embed, vectors)
    # Upsert first, delete after: an interrupted run leaves mixed hashes, which the next run treats as changed.
    kept = {document_id(chunk.metadata["chunk_id"]) for chunk in to_embed}
    stale = [record.id for source in updated + removed for record in indexed[source] if record.id not in kept]
    store.delete_ids(stale)
    return SyncReport(added, updated, removed, unchanged, len(to_embed))


def _hashed_chunks(documents: list[Document], fingerprint: str) -> list[Document]:
    chunks = split_documents(documents)
    hashes = {doc.metadata["source"]: _content_hash(doc.page_content, fingerprint) for doc in documents}
    for chunk in chunks:
        chunk.metadata["content_hash"] = hashes[chunk.metadata["source"]]
    return chunks


def _content_hash(text: str, fingerprint: str) -> str:
    # Chunk settings are read at call time, so changing them re-embeds every document.
    settings = f"{INDEX_VERSION}:{chunking.CHUNK_SIZE}:{chunking.CHUNK_OVERLAP}"
    return hashlib.sha256(f"{settings}\n{fingerprint}\n{text}".encode()).hexdigest()


def _is_current(records: list[ChunkRecord], chunks: list[Document]) -> bool:
    expected_hash = chunks[0].metadata["content_hash"]
    expected_ids = {document_id(chunk.metadata["chunk_id"]) for chunk in chunks}
    return {r.content_hash for r in records} == {expected_hash} and {r.id for r in records} == expected_ids


def _by_source(items, source_of) -> dict[str, list]:
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(source_of(item), []).append(item)
    return groups


def _embed(chunks: Sequence[Document], embeddings: Embeddings) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        vectors += embeddings.embed_documents([c.page_content for c in chunks[start : start + BATCH_SIZE]])
    return vectors


def _upsert(store: TypesenseStore, chunks: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
    for start in range(0, len(chunks), BATCH_SIZE):
        store.upsert(chunks[start : start + BATCH_SIZE], vectors[start : start + BATCH_SIZE])
