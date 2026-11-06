"""Membaca dokumen HR (Markdown, teks, PDF) dari folder menjadi LangChain Document."""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from hr_assistant.errors import SetupError

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


class DocumentLoadError(SetupError):
    """Satu berkas dokumen tidak bisa dibaca; pesannya menyebut berkas mana."""


def load_documents(docs_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            source = path.relative_to(docs_dir).as_posix()
            try:
                text = _read_text(path)
            except (PyPdfError, UnicodeDecodeError, OSError) as error:
                message = f"Gagal membaca {source}: {error}. Perbaiki atau keluarkan berkas ini dari folder dokumen."
                raise DocumentLoadError(message) from error
            documents.append(Document(page_content=text, metadata={"source": source}))
    return documents


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")
