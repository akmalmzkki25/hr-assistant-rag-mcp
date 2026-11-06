import asyncio
import time
from contextlib import asynccontextmanager

import anyio
import psycopg
import pytest
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from hr_assistant import cli
from hr_assistant.agent.graph import AgentResult
from hr_assistant.config import get_settings
from hr_assistant.rag.embeddings import EmbeddingUnavailableError


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def ask(self, question, employee_id=None, thread_id=None):
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def ready(monkeypatch):
    """Lewati pengecekan servis, sehingga test fokus pada perilaku CLI."""
    monkeypatch.setattr(cli, "_ready_settings", lambda: get_settings())


def use_agent(monkeypatch, agent, in_task_group=False):
    @asynccontextmanager
    async def fake_open_agent(settings):
        if in_task_group:
            # The real MCP client runs inside an anyio task group, which wraps errors in an ExceptionGroup.
            async with anyio.create_task_group():
                yield agent
        else:
            yield agent

    monkeypatch.setattr(cli, "open_agent", fake_open_agent)


def stderr_lines(capsys) -> list[str]:
    return capsys.readouterr().err.strip().splitlines()


def test_blank_question_rejected(capsys):
    assert cli.main(["ask", "   "]) == 2
    assert stderr_lines(capsys) == ["Pertanyaan tidak boleh kosong."]


def test_missing_api_key_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert cli.main(["ask", "Berapa cuti?"]) == 2
    assert stderr_lines(capsys) == [cli.NO_API_KEY]


def test_typesense_unreachable_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TYPESENSE_URL", "http://localhost:9")
    started = time.monotonic()
    assert cli.main(["ask", "Berapa cuti?"]) == 2
    assert time.monotonic() - started < 15
    assert "Typesense tidak bisa dihubungi di http://localhost:9" in stderr_lines(capsys)[0]


@pytest.mark.integration
def test_missing_index_exits_2(monkeypatch, capsys):
    from hr_assistant.rag.store import TypesenseStore, make_typesense_client

    if not TypesenseStore(make_typesense_client(get_settings()), "x", 1).is_reachable():
        pytest.skip("Typesense tidak bisa dihubungi")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TYPESENSE_COLLECTION", "hr_docs_missing_test")
    assert cli.main(["ask", "Berapa cuti?"]) == 2
    assert stderr_lines(capsys) == [cli.NO_INDEX]


def test_llm_error_inside_task_group_prints_one_line(ready, monkeypatch, capsys):
    error = ChatGoogleGenerativeAIError("Error calling model 'gemini-3.8-flash' (INVALID_ARGUMENT): API key not valid.")
    use_agent(monkeypatch, FakeAgent(error=error), in_task_group=True)
    assert cli.main(["ask", "Berapa cuti?"]) == 1
    assert stderr_lines(capsys) == [
        "LLM tidak tersedia: Error calling model 'gemini-3.8-flash' (INVALID_ARGUMENT): API key not valid."
    ]


def test_embedding_error_prints_one_line(ready, monkeypatch, capsys):
    use_agent(monkeypatch, FakeAgent(error=EmbeddingUnavailableError("429 RESOURCE_EXHAUSTED: kuota habis")))
    assert cli.main(["ask", "Berapa cuti?"]) == 1
    assert stderr_lines(capsys) == ["Embedding Gemini tidak tersedia: 429 RESOURCE_EXHAUSTED: kuota habis"]


def test_db_init_with_postgres_down_prints_one_line(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:x@localhost:9/hr_assistant")
    started = time.monotonic()
    assert cli.main(["db-init"]) == 1
    assert time.monotonic() - started < 30
    lines = stderr_lines(capsys)
    assert len(lines) == 1 and lines[0].startswith("Postgres tidak bisa dihubungi:")


def test_ingest_with_no_chunks_exits_2(tmp_path, monkeypatch, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "kosong.md").write_text("# Judul saja\n\n## Bagian kosong\n", encoding="utf-8")
    monkeypatch.setenv("HR_DOCS_DIR", str(docs))
    assert cli.main(["ingest"]) == 2
    assert "Tidak ada teks yang bisa di-index" in stderr_lines(capsys)[0]


def test_ask_prints_answer_and_sources(ready, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:x@localhost:9/hr_assistant")
    result = AgentResult("12 hari kerja.", "rag", "final", ["kebijakan_cuti.md#Cuti Tahunan"])
    use_agent(monkeypatch, FakeAgent(result=result))
    started = time.monotonic()
    assert cli.main(["ask", "Berapa cuti?"]) == 0
    assert time.monotonic() - started < 30
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["12 hari kerja.", "Sumber: kebijakan_cuti.md#Cuti Tahunan"]
    assert "Log run tidak tersimpan" in captured.err


@pytest.mark.integration
def test_ask_logs_run_to_postgres(ready, monkeypatch, test_database_url):
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    use_agent(monkeypatch, FakeAgent(result=AgentResult("Sisa cuti 7 hari.", "tool", "final")))
    with psycopg.connect(test_database_url) as conn:
        before = conn.execute("select count(*) from agent_runs").fetchone()[0]
    assert cli.main(["ask", "Sisa cuti saya?", "--employee-id", "E001"]) == 0
    with psycopg.connect(test_database_url) as conn:
        row = conn.execute("select question, employee_id, route from agent_runs order by id desc limit 1").fetchone()
        after = conn.execute("select count(*) from agent_runs").fetchone()[0]
    assert after == before + 1 and row == ("Sisa cuti saya?", "E001", "tool")


def test_ctrl_c_returns_130(monkeypatch, capsys):
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_cmd_ask", interrupted)
    assert cli.main(["ask", "Berapa cuti?"]) == 130
    assert stderr_lines(capsys) == ["Dibatalkan."]


def test_read_line_returns_typed_text(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "halo")
    assert asyncio.run(cli._read_line("Kamu: ")) == "halo"
