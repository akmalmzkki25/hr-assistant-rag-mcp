"""Memecah dokumen HR menjadi chunk per heading untuk di-embed."""

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    section: str


def chunk_markdown(text: str, source: str, max_chars: int = 700, overlap_chars: int = 100) -> list[Chunk]:
    if any(_HEADING.match(line) for line in text.splitlines()):
        sections = _sections(text)
    else:
        # PDF and plain text extraction has no Markdown headings, so the file name labels the section.
        body = "\n".join(line for line in text.splitlines() if line.strip()).strip()
        sections = [(PurePosixPath(source).stem, body)] if body else []
    chunks: list[Chunk] = []
    for section, body in sections:
        for piece in _split_body(body, max_chars, overlap_chars):
            chunks.append(Chunk(id=f"{source}::{len(chunks)}", text=f"{section}\n{piece}", source=source, section=section))
    return chunks


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            name = path.relative_to(docs_dir).as_posix()
            documents.append((name, _read_text(path)))
    return documents


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def _sections(text: str) -> list[tuple[str, str]]:
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(line for line in body if line.strip()).strip()
        if stack and content:
            sections.append((" > ".join(title for _, title in stack), content))
        body.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if not match:
            body.append(line)
            continue
        flush()
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2)))
    flush()
    return sections


def _split_body(body: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    units = [unit for sentence in _SENTENCE_END.split(body) for unit in _fit(sentence, max_chars - overlap_chars)]
    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        pieces.append(current)
        current = f"{_tail(current, overlap_chars)} {unit}".strip()
    if current:
        pieces.append(current)
    return pieces


def _fit(sentence: str, limit: int) -> list[str]:
    """Pecah kalimat yang lebih panjang dari limit di batas kata."""
    if len(sentence) <= limit:
        return [sentence]
    parts, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > limit:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return parts + [current] if current else parts


def _tail(text: str, overlap_chars: int) -> str:
    tail = text[-overlap_chars:]
    return tail.split(" ", 1)[1] if " " in tail else tail
