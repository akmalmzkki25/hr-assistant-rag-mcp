"""Memecah dokumen per heading Markdown, lalu per ukuran, dengan LangChain text splitter."""

import re
from pathlib import PurePosixPath

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADER_LEVELS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
_HEADING_LINE = re.compile(r"^#{1,3}\s+\S", re.MULTILINE)


def split_documents(docs: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Document]:
    sizer = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE if chunk_size is None else chunk_size,
        chunk_overlap=CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
    )
    chunks: list[Document] = []
    for doc in docs:
        source = doc.metadata["source"]
        pieces = 0
        for section, body in _sections(doc.page_content, source):
            for text in sizer.split_text(body):
                chunks.append(
                    Document(
                        page_content=f"{section}\n{text}",
                        metadata={"source": source, "section": section, "chunk_id": f"{source}::{pieces}"},
                    )
                )
                pieces += 1
    return chunks


def _sections(text: str, source: str) -> list[tuple[str, str]]:
    if not _HEADING_LINE.search(text):
        # PDF and plain-text extraction has no Markdown headings, so the file name labels the section.
        body = text.strip()
        return [(PurePosixPath(source).stem, body)] if body else []
    sections = []
    for part in MarkdownHeaderTextSplitter(HEADER_LEVELS).split_text(text):
        titles = [part.metadata[key] for _, key in HEADER_LEVELS if key in part.metadata]
        body = part.page_content.strip()
        # Text before the first heading (a document disclaimer) has no titles and is not HR content.
        if titles and body:
            # The citation format [sumber: ...] ends at the first "]", so brackets in a title would break it.
            title = " > ".join(titles).translate(str.maketrans("[]", "()"))
            sections.append((title, body))
    return sections
