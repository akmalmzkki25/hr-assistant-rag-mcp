"""Agent HR: loop tool calling terbatas dengan guardrail grounding."""

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from mcp import Client

from hr_assistant.llm import LLM, Message, ToolCall, ToolSpec
from hr_assistant.prompts import CITATION_CORRECTION, SYSTEM_PROMPT, UNKNOWN_ANSWER
from hr_assistant.rag import Retriever

__all__ = ["UNKNOWN_ANSWER", "SEARCH_TOOL", "Step", "AgentResult", "HRAgent"]

SEARCH_TOOL = ToolSpec(
    name="search_hr_documents",
    description=(
        "Cari potongan dokumen kebijakan HR (cuti, kerja remote, reimbursement, absensi, lembur, FAQ). "
        "Mengembalikan found dan daftar chunk berisi source_id, text, dan score."
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Kata kunci atau pertanyaan dalam Bahasa Indonesia."}},
        "required": ["query"],
    },
)

_CITATION = re.compile(r"\[sumber:\s*([^\]]+?)\s*\]", re.IGNORECASE)


@dataclass
class Step:
    kind: Literal["llm", "tool", "guardrail"]
    name: str
    detail: dict


@dataclass
class AgentResult:
    answer: str
    route: Literal["rag", "tool", "mixed", "none"]
    stop_reason: Literal["final", "max_steps", "guardrail"]
    sources: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Run:
    """State satu pertanyaan: riwayat pesan, chunk yang pernah diambil, dan apakah MCP dipakai."""

    messages: list[Message]
    steps: list[Step] = field(default_factory=list)
    retrieved_ids: set[str] = field(default_factory=set)
    used_mcp: bool = False
    mcp_outputs: list[str] = field(default_factory=list)


class HRAgent:
    def __init__(self, llm: LLM, mcp: Client, retriever: Retriever, max_steps: int = 6, max_answer_retries: int = 1):
        self.llm = llm
        self.mcp = mcp
        self.retriever = retriever
        self.max_steps = max_steps
        self.max_answer_retries = max_answer_retries

    async def ask(self, question: str, employee_id: str | None = None) -> AgentResult:
        text = f"{question}\n\nID karyawan yang bertanya: {employee_id}" if employee_id else question
        run = _Run(messages=[Message(role="user", text=text)])
        tools = [SEARCH_TOOL] + await self._mcp_tools()
        retries_left = self.max_answer_retries

        for _ in range(self.max_steps):
            reply = await asyncio.to_thread(self.llm.generate, SYSTEM_PROMPT, list(run.messages), tools)
            run.steps.append(Step("llm", "generate", {"text": reply.text, "tool_calls": [asdict(c) for c in reply.tool_calls]}))
            run.messages.append(Message(role="assistant", text=reply.text, tool_calls=reply.tool_calls, raw=reply.raw))

            if reply.tool_calls:
                for tool_call in reply.tool_calls:
                    result = await self._execute(tool_call, run)
                    run.messages.append(
                        Message(role="tool", tool_name=tool_call.name, tool_call_id=tool_call.id, tool_result=result)
                    )
                continue

            # With MCP data in hand, "tidak tahu" may cover only part of the answer, so grounding decides.
            if not run.used_mcp and "tidak tahu" in reply.text.lower():
                return AgentResult(UNKNOWN_ANSWER, "none", "final", steps=run.steps)
            if _asks_for_employee_id(reply.text, run):
                return AgentResult(reply.text, "none", "final", steps=run.steps)

            verdict = _check_grounding(reply.text, run)
            if verdict is not None:
                route, sources = verdict
                return AgentResult(reply.text, route, "final", sources, run.steps)

            run.steps.append(Step("guardrail", "ungrounded_answer", {"retries_left": retries_left}))
            if retries_left == 0:
                return AgentResult(UNKNOWN_ANSWER, "none", "guardrail", steps=run.steps)
            retries_left -= 1
            run.messages.append(Message(role="user", text=CITATION_CORRECTION))

        run.steps.append(Step("guardrail", "max_steps", {"max_steps": self.max_steps}))
        return AgentResult(UNKNOWN_ANSWER, "none", "max_steps", steps=run.steps)

    async def _mcp_tools(self) -> list[ToolSpec]:
        listed = await self.mcp.list_tools()
        return [ToolSpec(t.name, t.description or "", t.input_schema) for t in listed.tools]

    async def _execute(self, tool_call: ToolCall, run: _Run) -> dict:
        if tool_call.name == SEARCH_TOOL.name:
            result = await asyncio.to_thread(self._search, str(tool_call.args.get("query", "")), run)
        else:
            result = await self._call_mcp(tool_call, run)
        run.steps.append(Step("tool", tool_call.name, {"args": tool_call.args, "result": result}))
        return result

    def _search(self, query: str, run: _Run) -> dict:
        hits = self.retriever.relevant(query)
        chunks = [
            {"source_id": f"{h.chunk.source}#{h.chunk.section}", "text": h.chunk.text, "score": round(h.score, 3)}
            for h in hits
        ]
        run.retrieved_ids.update(c["source_id"] for c in chunks)
        return {"found": bool(chunks), "chunks": chunks}

    async def _call_mcp(self, tool_call: ToolCall, run: _Run) -> dict:
        available = {spec.name for spec in await self._mcp_tools()}
        if tool_call.name not in available:
            return {"error": f"Tool {tool_call.name} tidak tersedia."}
        result = await self.mcp.call_tool(tool_call.name, tool_call.args)
        run.used_mcp = True
        text = "\n".join(getattr(block, "text", "") for block in result.content)
        run.mcp_outputs.append(text)
        if result.is_error:
            return {"error": text}
        if result.structured_content:
            return dict(result.structured_content)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"result": text}
        return parsed if isinstance(parsed, dict) else {"result": parsed}


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _asks_for_employee_id(answer: str, run: _Run) -> bool:
    """Satu kalimat tanya yang meminta ID karyawan, sebelum tool apa pun dipakai, tidak memuat fakta."""
    text = answer.strip()
    no_tools_used = not any(step.kind == "tool" for step in run.steps)
    single_question = text.endswith("?") and len(re.findall(r"[.!?]", text)) == 1 and len(text) <= 120
    return no_tools_used and single_question and "id karyawan" in text.lower() and not _NUMBER.search(text)


def _check_grounding(answer: str, run: _Run) -> tuple[str, list[str]] | None:
    """Kembalikan (route, sources) kalau jawaban bersumber dari tool, None kalau tidak."""
    if not answer.strip():
        return None
    cited = list(dict.fromkeys(_CITATION.findall(answer)))
    if any(source not in run.retrieved_ids for source in cited):
        return None
    if cited:
        return ("mixed" if run.used_mcp else "rag"), cited
    if run.retrieved_ids:
        return None
    if run.used_mcp and _numbers_in(answer, run.mcp_outputs):
        return "tool", []
    return None


def _numbers_in(answer: str, outputs: list[str]) -> bool:
    """Angka di jawaban tanpa sitasi harus muncul di hasil tool MCP, supaya aturan karangan tidak lolos."""
    haystack = "\n".join(outputs)
    return all(number in haystack for number in _NUMBER.findall(answer))
