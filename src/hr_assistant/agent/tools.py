"""Tool RAG untuk agent: pencarian dokumen HR yang mengembalikan source_id sebagai artifact."""

import json
from typing import Protocol

from langchain_core.tools import BaseTool, tool

from hr_assistant.rag.store import SearchHit

SEARCH_TOOL_NAME = "search_hr_documents"


class Retriever(Protocol):
    def relevant(self, query: str) -> list[SearchHit]: ...


def make_search_tool(retriever: Retriever) -> BaseTool:
    @tool(SEARCH_TOOL_NAME, response_format="content_and_artifact")
    def search_hr_documents(query: str) -> tuple[str, list[str]]:
        """Cari potongan dokumen kebijakan HR: cuti, kerja remote, reimbursement, absensi, lembur, perjalanan dinas, FAQ.

        Mengembalikan found dan daftar chunk berisi source_id, text, dan score. Pakai source_id persis untuk sitasi.
        query adalah kata kunci atau pertanyaan dalam Bahasa Indonesia.
        """
        chunks = [
            {
                "source_id": f"{hit.document.metadata['source']}#{hit.document.metadata['section']}",
                "text": hit.document.page_content,
                "score": round(hit.similarity, 3),
            }
            for hit in retriever.relevant(query)
        ]
        # The artifact stays out of the model's context; the guardrail uses it to verify citations.
        return json.dumps({"found": bool(chunks), "chunks": chunks}, ensure_ascii=False), [c["source_id"] for c in chunks]

    return search_hr_documents
