import pytest

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
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def test_database_url():
    import psycopg

    from hr_assistant.config import get_settings
    from hr_assistant.db.database import init_database, with_database

    url = with_database(get_settings().database_url, "hr_assistant_test")
    try:
        init_database(url)
    except psycopg.OperationalError:
        pytest.skip("Postgres tidak bisa dihubungi")
    return url


@pytest.fixture
def sample_docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "kebijakan_cuti.md").write_text(CUTI_DOC, encoding="utf-8")
    (docs / "kerja_remote.md").write_text(REMOTE_DOC, encoding="utf-8")
    return docs
