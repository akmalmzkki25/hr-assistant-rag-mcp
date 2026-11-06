"""Pengganti dependensi eksternal untuk test: embedder dan LLM tanpa jaringan."""

import math
import re
import zlib
from collections.abc import Sequence

from hr_assistant.llm import LLMReply, Message, ToolCall, ToolSpec


class ScriptedLLM:
    """Mengembalikan balasan yang sudah ditentukan dan mencatat setiap panggilan."""

    def __init__(self, replies: list[LLMReply]):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def generate(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMReply:
        self.calls.append({"system": system, "messages": list(messages), "tools": list(tools)})
        assert self._replies, "ScriptedLLM kehabisan balasan"
        return self._replies.pop(0)


def call(name: str, **args) -> LLMReply:
    return LLMReply(text="", tool_calls=(ToolCall(id=f"call_{name}", name=name, args=args),))


def final(text: str) -> LLMReply:
    return LLMReply(text=text)


class HashEmbedder:
    """Bag-of-words yang di-hash ke vektor tetap, sehingga kata yang sama menghasilkan kemiripan."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
