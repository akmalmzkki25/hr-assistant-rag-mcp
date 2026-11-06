"""Vector store Typesense dengan hybrid search (kata kunci + vektor)."""

from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

import hashlib
import json
from typing import Literal

import httpx
import typesense
from langchain_core.documents import Document
from typesense.exceptions import ObjectNotFound, TypesenseClientError

from hr_assistant.config import Settings

VECTOR_POOL = 100


def document_id(chunk_id: str) -> str:
    """ID dokumen Typesense yang aman di URL: chunk_id berisi path berkas, dan "/" membuat hapus per ID salah alamat."""
    return hashlib.sha1(chunk_id.encode()).hexdigest()


class SearchUnavailableError(RuntimeError):
    """Typesense mati atau menolak query saat agent mencari dokumen."""


@dataclass(frozen=True)
class ChunkRecord:
    """Ringkasan satu chunk yang sudah ter-index, tanpa teks dan vektornya."""

    id: str
    source: str
    content_hash: str


@dataclass(frozen=True)
class SearchHit:
    document: Document
    similarity: float
    text_match: int


def make_typesense_client(settings: Settings) -> typesense.Client:
    url = urlparse(settings.typesense_url)
    node = {"host": url.hostname, "port": str(url.port or 8108), "protocol": url.scheme or "http"}
    return typesense.Client(
        {
            "nodes": [node],
            "api_key": settings.typesense_api_key or "",
            "connection_timeout_seconds": 5,
            # One quick retry: a stopped container should surface in seconds, not after the client's default backoff.
            "num_retries": 1,
            "retry_interval_seconds": 0.2,
        }
    )


class TypesenseStore:
    def __init__(self, client: typesense.Client, collection: str, dim: int):
        self._client = client
        self.collection = collection
        self.dim = dim

    def is_reachable(self) -> bool:
        try:
            return bool(self._client.operations.is_healthy())
        except Exception:
            return False

    def recreate(self) -> None:
        self.drop()
        self._client.collections.create(
            {
                "name": self.collection,
                "fields": [
                    {"name": "content", "type": "string"},
                    {"name": "source", "type": "string", "facet": True},
                    {"name": "section", "type": "string"},
                    {"name": "chunk_id", "type": "string"},
                    {"name": "content_hash", "type": "string"},
                    {"name": "embedding", "type": "float[]", "num_dim": self.dim},
                ],
            }
        )

    def schema_state(self) -> Literal["missing", "current", "outdated"]:
        """outdated: collection dari versi lama (tanpa content_hash) atau dimensi embedding berbeda."""
        try:
            fields = {f["name"]: f for f in self._client.collections[self.collection].retrieve()["fields"]}
        except ObjectNotFound:
            return "missing"
        embedding = fields.get("embedding", {})
        return "current" if "content_hash" in fields and embedding.get("num_dim") == self.dim else "outdated"

    def indexed_chunks(self) -> list[ChunkRecord]:
        try:
            exported = self._client.collections[self.collection].documents.export({"include_fields": "id,source,content_hash"})
        except ObjectNotFound:
            return []
        rows = [json.loads(line) for line in exported.splitlines() if line.strip()]
        return [ChunkRecord(row["id"], row.get("source", ""), row.get("content_hash", "")) for row in rows]

    def delete_ids(self, ids: Sequence[str]) -> None:
        # Deleting by ID avoids filter_by quoting, which breaks on file names containing a backtick.
        documents = self._client.collections[self.collection].documents
        for chunk_id in ids:
            try:
                documents[chunk_id].delete()
            except ObjectNotFound:
                pass

    def drop(self) -> None:
        try:
            self._client.collections[self.collection].delete()
        except ObjectNotFound:
            pass

    def upsert(self, docs: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
        records = [
            {
                "id": document_id(doc.metadata["chunk_id"]),
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "section": doc.metadata["section"],
                "chunk_id": doc.metadata["chunk_id"],
                "content_hash": doc.metadata.get("content_hash", ""),
                "embedding": list(vector),
            }
            for doc, vector in zip(docs, vectors, strict=True)
        ]
        results = self._client.collections[self.collection].documents.import_(records, {"action": "upsert"})
        failed = [r for r in results if not r.get("success")]
        if failed:
            raise RuntimeError(f"Typesense menolak {len(failed)} dokumen: {failed[0].get('error')}")

    def search(self, query: str, vector: Sequence[float], k: int) -> list[SearchHit]:
        vector_literal = ",".join(f"{v:.6f}" for v in vector)
        search = {
            "collection": self.collection,
            "q": query,
            "query_by": "content",
            # A wide neighbour pool gives keyword hits a vector distance too, so they can be reranked.
            "vector_query": f"embedding:([{vector_literal}], k:{max(k, VECTOR_POOL)})",
            "exclude_fields": "embedding",
            "per_page": k,
        }
        # multi_search sends the vector in the POST body; a 768-dim vector is too long for a GET query string.
        try:
            result = self._client.multi_search.perform({"searches": [search]}, {})["results"][0]
        except httpx.TransportError as error:
            raise SearchUnavailableError(f"Typesense tidak bisa dihubungi: {error}. Nyalakan container: docker start typesense") from error
        except TypesenseClientError as error:
            raise SearchUnavailableError(f"Pencarian Typesense gagal: {error}") from error
        if "error" in result:
            raise SearchUnavailableError(f"Pencarian Typesense gagal: {result['error']}")
        return [_to_hit(hit) for hit in result.get("hits", [])]

    def count(self) -> int:
        try:
            return self._client.collections[self.collection].retrieve()["num_documents"]
        except ObjectNotFound:
            return 0


def _to_hit(hit: dict) -> SearchHit:
    doc = hit["document"]
    metadata = {"source": doc["source"], "section": doc["section"], "chunk_id": doc["chunk_id"]}
    # Keyword-only matches can arrive without a vector distance; treat them as dissimilar.
    similarity = 1.0 - hit["vector_distance"] if "vector_distance" in hit else 0.0
    return SearchHit(Document(page_content=doc["content"], metadata=metadata), similarity, int(hit.get("text_match", 0)))
