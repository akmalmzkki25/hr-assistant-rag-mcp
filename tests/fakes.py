"""Pengganti dependensi eksternal untuk test: embedder dan chat model tanpa jaringan."""

import math
import re
import zlib
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings


class HashEmbedder(Embeddings):
    """Bag-of-words yang di-hash ke vektor tetap, sehingga kata yang sama menghasilkan kemiripan."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            # Typesense rejects all-zero vectors for cosine search, so unseen words map to one fixed axis.
            vector[0], norm = 1.0, 1.0
        return [v / norm for v in vector]
