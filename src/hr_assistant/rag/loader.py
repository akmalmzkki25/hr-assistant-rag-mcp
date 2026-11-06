"""Membaca dokumen HR (Markdown, teks, PDF) dari folder menjadi LangChain Document."""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_documents(docs_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            source = path.relative_to(docs_dir).as_posix()
            documents.append(Document(page_content=_read_text(path), metadata={"source": source}))
    return documents


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")
