from pathlib import Path

from langchain_core.documents import Document

from hr_assistant.rag.chunking import split_documents
from hr_assistant.rag.loader import load_documents

DATA_DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


def doc(text: str, source: str = "kebijakan_cuti.md") -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_headings_become_section_path():
    md = "# Kebijakan Cuti\n\n## Cuti Tahunan\n12 hari kerja per tahun.\n\n## Cuti Sakit\nWajib surat dokter."
    chunks = split_documents([doc(md)])
    assert [c.metadata["section"] for c in chunks] == ["Kebijakan Cuti > Cuti Tahunan", "Kebijakan Cuti > Cuti Sakit"]
    assert [c.metadata["chunk_id"] for c in chunks] == ["kebijakan_cuti.md::0", "kebijakan_cuti.md::1"]
    assert chunks[0].page_content.startswith("Kebijakan Cuti > Cuti Tahunan\n")
    assert chunks[0].metadata["source"] == "kebijakan_cuti.md"


def test_long_section_split_with_overlap():
    body = " ".join(f"Kalimat nomor {i} tentang aturan cuti." for i in range(120))
    chunks = split_documents([doc(f"## Panjang\n{body}")], chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 400 + len("Panjang\n") for c in chunks)
    tail_word = chunks[0].page_content.split()[-1]
    assert tail_word in chunks[1].page_content


def test_text_before_first_heading_is_dropped():
    chunks = split_documents([doc("> Dokumen contoh fiktif.\n\n## Isi\nada isi")])
    assert [c.metadata["section"] for c in chunks] == ["Isi"]


def test_headingless_text_uses_file_stem_as_section():
    chunks = split_documents([doc("Kebijakan cuti.\nSetiap karyawan dapat 12 hari.", source="sop/cuti_2026.pdf")])
    assert [c.metadata["section"] for c in chunks] == ["cuti_2026"]
    assert "12 hari" in chunks[0].page_content


def test_empty_sections_skipped():
    assert [c.metadata["section"] for c in split_documents([doc("## Kosong\n\n## Isi\nada")])] == ["Isi"]


def test_loader_reads_markdown_and_skips_other_files(tmp_path):
    (tmp_path / "x.md").write_text("## A\nisi", encoding="utf-8")
    (tmp_path / "skip.png").write_bytes(b"\x89PNG")
    docs = load_documents(tmp_path)
    assert [(d.metadata["source"], d.page_content) for d in docs] == [("x.md", "## A\nisi")]


def test_sample_pdf_is_loaded_and_chunked():
    pdf_docs = [d for d in load_documents(DATA_DOCS) if d.metadata["source"].endswith(".pdf")]
    assert len(pdf_docs) == 1
    chunks = split_documents(pdf_docs)
    assert chunks and chunks[0].metadata["section"] == "pedoman_perjalanan_dinas"
    assert "uang harian" in " ".join(c.page_content for c in chunks).lower()
