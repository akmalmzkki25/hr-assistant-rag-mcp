from hr_assistant.chunking import chunk_markdown, load_documents


def test_chunks_follow_headings():
    md = "# Kebijakan Cuti\n\n## Cuti Tahunan\nKaryawan mendapat 12 hari kerja per tahun.\n\n## Cuti Sakit\nWajib surat dokter."
    chunks = chunk_markdown(md, source="kebijakan_cuti.md")
    assert [c.section for c in chunks] == ["Kebijakan Cuti > Cuti Tahunan", "Kebijakan Cuti > Cuti Sakit"]
    assert chunks[0].id == "kebijakan_cuti.md::0"
    assert chunks[0].text.startswith("Kebijakan Cuti > Cuti Tahunan\n")


def test_long_section_split_with_overlap():
    body = " ".join(f"kalimat{i}." for i in range(400))
    chunks = chunk_markdown(f"## Panjang\n{body}", source="a.md", max_chars=700, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 + len("Panjang\n") for c in chunks)
    assert chunks[0].text[-50:].split()[-1] in chunks[1].text


def test_empty_sections_skipped():
    assert chunk_markdown("## Kosong\n\n## Isi\nada", source="b.md")[0].section == "Isi"


def test_text_before_first_heading_is_ignored():
    chunks = chunk_markdown("> Dokumen contoh fiktif.\n\n## Isi\nada", source="c.md")
    assert [c.section for c in chunks] == ["Isi"]


def test_load_documents_reads_markdown(tmp_path):
    (tmp_path / "x.md").write_text("## A\nisi", encoding="utf-8")
    (tmp_path / "skip.png").write_bytes(b"\x89PNG")
    assert load_documents(tmp_path) == [("x.md", "## A\nisi")]
