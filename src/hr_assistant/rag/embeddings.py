"""Embedding Gemini (gemini-embedding-2) sebagai LangChain Embeddings."""

import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from langchain_core.embeddings import Embeddings

_TRANSPORT_ERRORS = (httpx.TransportError,)


class EmbeddingUnavailableError(Exception):
    """Embedding tidak bisa dibuat: kuota habis, key salah, atau jaringan bermasalah."""


class GeminiEmbeddings(Embeddings):
    """gemini-embedding-2 tidak menerima task_type, jadi jenis tugas ditulis sebagai prefix teks."""

    def __init__(
        self,
        model: str,
        dimensions: int,
        api_key: str | None = None,
        client: Any = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        batch_size: int = 32,
    ):
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._sleep = sleep
        self._client = client or genai.Client(api_key=api_key)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [_document_text(text) for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            vectors += self._embed(prepared[start : start + self.batch_size])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"task: search result | query: {text}"])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # A list of plain strings is embedded as one combined input, so each text gets its own Content.
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
        config = types.EmbedContentConfig(output_dimensionality=self.dimensions)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.embed_content(model=self.model, contents=contents, config=config)
                return [list(e.values) for e in response.embeddings]
            except (errors.ClientError, errors.ServerError) as error:
                retryable = isinstance(error, errors.ServerError) or error.code == 429
                if not retryable or attempt == self.max_retries:
                    raise EmbeddingUnavailableError(f"{error.code} {error.status}: {error.message}") from error
            except _TRANSPORT_ERRORS as error:
                if attempt == self.max_retries:
                    raise EmbeddingUnavailableError(f"Tidak bisa terhubung ke Gemini: {error}") from error
            self._sleep(2**attempt)
        raise AssertionError("unreachable")


def _document_text(text: str) -> str:
    title, _, body = text.partition("\n")
    return f"title: {title} | text: {body.strip()}" if body else f"title: none | text: {title}"
