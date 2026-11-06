import pytest

from fakes import HashEmbedder
from hr_assistant.rag import Retriever, build_index
from hr_assistant.vector_store import ChromaStore

CUTI_DOC = """# Kebijakan Cuti

## Cuti Tahunan
Setiap karyawan tetap mendapat cuti tahunan sebanyak 12 hari kerja per tahun.

## Cuti Sakit
Sakit lebih dari 2 hari wajib surat keterangan dokter.
"""

REMOTE_DOC = """# Kebijakan Kerja Remote

## Ketentuan Umum
Karyawan boleh kerja remote maksimal 2 hari per minggu dengan persetujuan atasan.
"""


@pytest.fixture
def sample_docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "kebijakan_cuti.md").write_text(CUTI_DOC, encoding="utf-8")
    (docs / "kerja_remote.md").write_text(REMOTE_DOC, encoding="utf-8")
    return docs


@pytest.fixture
def retriever(tmp_path, sample_docs):
    store = ChromaStore(tmp_path / "idx", "hr_docs")
    embedder = HashEmbedder()
    build_index(sample_docs, store, embedder)
    return Retriever(store, embedder, top_k=4, min_score=0.45)
