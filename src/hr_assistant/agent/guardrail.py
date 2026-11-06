"""Guardrail deterministik: jawaban final harus bersumber dari hasil tool pada giliran ini."""

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

from hr_assistant.agent.prompts import ASK_EMPLOYEE_ID, UNKNOWN_ANSWER

_CITATION = re.compile(r"\[sumber:\s*([^\]]+?)\s*\]", re.IGNORECASE)
# Leading and trailing \d keep "Rp500.000" whole and stop "17" from matching "7".
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
_EMPLOYEE_ID = re.compile(r"\b[A-Z]\d+\b")
_TOOL_ERROR_PREFIX = re.compile(r"^Error executing tool \w+:\s*")
_REQUEST_WORDS = ("mohon", "tolong", "silakan", "sebutkan", "berikan", "masukkan", "informasikan")

Route = Literal["rag", "tool", "mixed", "none"]


@dataclass(frozen=True)
class Verdict:
    kind: Literal["grounded", "unknown", "clarify", "ungrounded"]
    answer: str
    route: Route = "none"
    sources: list[str] = field(default_factory=list)


def check(answer: str, retrieved: Mapping[str, str], mcp_outputs: list[str]) -> Verdict:
    """retrieved memetakan source_id ke teks chunk yang diambil pada giliran ini."""
    used_mcp = bool(mcp_outputs)
    if _normalize(answer) == _normalize(UNKNOWN_ANSWER):
        return Verdict("unknown", UNKNOWN_ANSWER)
    # With MCP data in hand, "tidak tahu" may cover only part of the answer, so grounding decides.
    if not used_mcp and "tidak tahu" in answer.lower():
        return Verdict("unknown", UNKNOWN_ANSWER)
    # A fact-free question for the ID is safe after a document search, but not once employee data was read.
    if not used_mcp and _asks_for_employee_id(answer):
        # The fixed sentence replaces the model text, so facts slipped into a clarifying reply never reach the user.
        return Verdict("clarify", ASK_EMPLOYEE_ID)
    if not answer.strip():
        return Verdict("ungrounded", answer)

    known = {_normalize(source_id): source_id for source_id in retrieved}
    cited = [_normalize(c) for c in _CITATION.findall(answer)]
    if any(c not in known for c in cited):
        return Verdict("ungrounded", answer)
    if not _numbers_supported(answer, [*retrieved.values(), *mcp_outputs]):
        return Verdict("ungrounded", answer)
    if cited:
        sources = list(dict.fromkeys(known[c] for c in cited))
        return Verdict("grounded", answer, "mixed" if used_mcp else "rag", sources)
    if retrieved:
        return Verdict("ungrounded", answer)
    if used_mcp and _mentions_tool_fact(answer, mcp_outputs):
        return Verdict("grounded", answer, "tool")
    return Verdict("ungrounded", answer)


def _asks_for_employee_id(answer: str) -> bool:
    """Balasan singkat tanpa angka yang meminta ID karyawan, berupa pertanyaan atau kalimat permintaan."""
    text = answer.strip()
    lowered = text.lower()
    is_request = text.endswith("?") or any(word in lowered for word in _REQUEST_WORDS)
    return len(text) <= 300 and is_request and "id karyawan" in lowered and not _NUMBER.search(text)


def _numbers_supported(answer: str, evidence: list[str]) -> bool:
    """Setiap angka di jawaban harus ada sebagai angka utuh di chunk atau hasil MCP, supaya angka karangan tidak lolos."""
    body = _LIST_MARKER.sub("", _CITATION.sub("", answer))
    known = {token for text in evidence for token in _number_forms(text)}
    return all(not known.isdisjoint(_number_forms(token)) for token in _NUMBER.findall(body))


def _number_forms(text: str) -> set[str]:
    # "Rp500.000" in a chunk and "500000" in JSON are the same amount.
    tokens = _NUMBER.findall(text)
    return {*tokens, *(re.sub(r"[.,]", "", token) for token in tokens)}


def _mentions_tool_fact(answer: str, outputs: list[str]) -> bool:
    """Jawaban tanpa sitasi harus menyebut minimal satu nilai dari hasil MCP, bukan hanya bebas dari angka."""
    body = _LIST_MARKER.sub("", answer)
    tool_numbers = {token for output in outputs for token in _number_forms(output)}
    if any(not tool_numbers.isdisjoint(_number_forms(token)) for token in _NUMBER.findall(body)):
        return True
    text = _normalize(body)
    return any(fact in text for output in outputs for fact in _tool_facts(output))


def _tool_facts(output: str) -> Iterator[str]:
    try:
        data = json.loads(output)
    except ValueError:
        # Tool errors arrive as plain text; relaying the message or the employee ID it names counts.
        message = _TOOL_ERROR_PREFIX.sub("", output.strip())
        yield from (_normalize(employee_id) for employee_id in _EMPLOYEE_ID.findall(message))
        first_sentence = message.split(".")[0]
        if first_sentence.strip():
            yield _normalize(first_sentence)
        return
    # Numbers are matched as whole tokens by the caller; short strings would match by accident.
    for value in _leaves(data):
        if isinstance(value, str) and len(value.strip()) >= 3:
            yield _normalize(value)


def _leaves(data) -> Iterator:
    if isinstance(data, dict):
        for value in data.values():
            yield from _leaves(value)
    elif isinstance(data, list):
        for value in data:
            yield from _leaves(value)
    else:
        yield data


def _normalize(source_id: str) -> str:
    return " ".join(source_id.casefold().split())
