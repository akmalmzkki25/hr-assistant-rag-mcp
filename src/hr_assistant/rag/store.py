"""Vector store Typesense dengan hybrid search (kata kunci + vektor)."""

from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

import typesense
from langchain_core.documents import Document
from typesense.exceptions import ObjectNotFound

from hr_assistant.config import Settings

VECTOR_POOL = 100


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
                    {"name": "embedding", "type": "float[]", "num_dim": self.dim},
                ],
            }
        )

    def drop(self) -> None:
        try:
            self._client.collections[self.collection].delete()
        except ObjectNotFound:
            pass

    def upsert(self, docs: Sequence[Document], vectors: Sequence[Sequence[float]]) -> None:
        records = [
            {
                "id": doc.metadata["chunk_id"],
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "section": doc.metadata["section"],
                "chunk_id": doc.metadata["chunk_id"],
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
        result = self._client.multi_search.perform({"searches": [search]}, {})["results"][0]
        if "error" in result:
            raise RuntimeError(f"Pencarian Typesense gagal: {result['error']}")
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
