"""Ingest: muat dokumen, pecah, embed, lalu simpan ulang ke Typesense."""

from pathlib import Path

from langchain_core.embeddings import Embeddings

from hr_assistant.rag.chunking import split_documents
from hr_assistant.rag.loader import load_documents
from hr_assistant.rag.store import TypesenseStore

BATCH_SIZE = 64


def build_index(docs_dir: Path, store: TypesenseStore, embeddings: Embeddings) -> int:
    chunks = split_documents(load_documents(docs_dir))
    store.recreate()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        store.upsert(batch, embeddings.embed_documents([c.page_content for c in batch]))
    return len(chunks)
