"""Pengganti dependensi eksternal untuk test: embedder dan chat model tanpa jaringan."""

import math
import re
import zlib
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from hr_assistant.rag.store import SearchHit


class FakeToolChatModel(BaseChatModel):
    """Chat model skrip: mengembalikan AIMessage berurutan dan mencatat pesan serta tool yang diterima."""

    _responses: list[AIMessage] = PrivateAttr(default_factory=list)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _tool_names: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[AIMessage], **kwargs):
        super().__init__(**kwargs)
        self._responses = list(responses)

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def tool_names(self) -> list[str]:
        return self._tool_names

    @property
    def _llm_type(self) -> str:
        return "fake-tool-chat"

    def bind_tools(self, tools, **kwargs):
        self._tool_names = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._calls.append(list(messages))
        assert self._responses, "FakeToolChatModel kehabisan balasan"
        return ChatResult(generations=[ChatGeneration(message=self._responses.pop(0))])


def call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{len(args)}"}])


def final(text: str) -> AIMessage:
    return AIMessage(content=text)


class FakeRetriever:
    """Mengembalikan bagian Cuti Tahunan untuk query yang menyebut cuti, selain itu kosong."""

    def __init__(self):
        self.queries: list[str] = []

    def relevant(self, query: str) -> list[SearchHit]:
        self.queries.append(query)
        if "cuti" not in query.lower():
            return []
        doc = Document(
            page_content="Kebijakan Cuti > Cuti Tahunan\nSetiap karyawan mendapat 12 hari kerja per tahun.",
            metadata={"source": "kebijakan_cuti.md", "section": "Kebijakan Cuti > Cuti Tahunan", "chunk_id": "kebijakan_cuti.md::0"},
        )
        return [SearchHit(doc, similarity=0.8, text_match=1)]


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
