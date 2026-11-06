"""Guardrail deterministik: jawaban final harus bersumber dari hasil tool pada giliran ini."""

import re
from dataclasses import dataclass, field
from typing import Literal

from hr_assistant.agent.prompts import UNKNOWN_ANSWER

_CITATION = re.compile(r"\[sumber:\s*([^\]]+?)\s*\]", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

Route = Literal["rag", "tool", "mixed", "none"]


@dataclass(frozen=True)
class Verdict:
    kind: Literal["grounded", "unknown", "clarify", "ungrounded"]
    answer: str
    route: Route = "none"
    sources: list[str] = field(default_factory=list)


def check(answer: str, retrieved_ids: set[str], mcp_outputs: list[str], used_tools: bool) -> Verdict:
    used_mcp = bool(mcp_outputs)
    # With MCP data in hand, "tidak tahu" may cover only part of the answer, so grounding decides.
    if not used_mcp and "tidak tahu" in answer.lower():
        return Verdict("unknown", UNKNOWN_ANSWER)
    if not used_tools and _asks_for_employee_id(answer):
        return Verdict("clarify", answer)
    if not answer.strip():
        return Verdict("ungrounded", answer)

    known = {_normalize(source_id): source_id for source_id in retrieved_ids}
    cited = [_normalize(c) for c in _CITATION.findall(answer)]
    if any(c not in known for c in cited):
        return Verdict("ungrounded", answer)
    if cited:
        sources = list(dict.fromkeys(known[c] for c in cited))
        return Verdict("grounded", answer, "mixed" if used_mcp else "rag", sources)
    if retrieved_ids:
        return Verdict("ungrounded", answer)
    if used_mcp and _numbers_in(answer, mcp_outputs):
        return Verdict("grounded", answer, "tool")
    return Verdict("ungrounded", answer)


def _asks_for_employee_id(answer: str) -> bool:
    """Satu kalimat tanya yang meminta ID karyawan, sebelum tool apa pun dipakai, tidak memuat fakta."""
    text = answer.strip()
    single_question = text.endswith("?") and len(re.findall(r"[.!?]", text)) == 1 and len(text) <= 120
    return single_question and "id karyawan" in text.lower() and not _NUMBER.search(text)


def _numbers_in(answer: str, outputs: list[str]) -> bool:
    """Angka di jawaban tanpa sitasi harus muncul di hasil tool MCP, supaya aturan karangan tidak lolos."""
    haystack = "\n".join(outputs)
    return all(number in haystack for number in _NUMBER.findall(answer))


def _normalize(source_id: str) -> str:
    return " ".join(source_id.casefold().split())
