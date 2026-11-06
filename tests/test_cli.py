from contextlib import asynccontextmanager

import pytest

from fakes import HashEmbedder
from hr_assistant import cli
from hr_assistant.llm import LLMUnavailableError
from hr_assistant.rag import build_index
from hr_assistant.vector_store import ChromaStore


@pytest.fixture
def built_index(tmp_path, monkeypatch, sample_docs):
    index_dir = tmp_path / "index"
    build_index(sample_docs, ChromaStore(index_dir, "hr_docs"), HashEmbedder())
    monkeypatch.setenv("HR_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return index_dir


class RaisingAgent:
    def __init__(self, error):
        self.error = error

    async def ask(self, question, employee_id=None):
        raise self.error


def patch_agent_to_raise(monkeypatch, error):
    @asynccontextmanager
    async def fake_open_agent(settings):
        yield RaisingAgent(error)

    monkeypatch.setattr(cli, "open_agent", fake_open_agent)


def test_llm_error_inside_task_group_prints_one_line(built_index, monkeypatch, capsys):
    import anyio

    @asynccontextmanager
    async def task_group_agent(settings):
        # The real MCP client runs inside an anyio task group, which wraps errors in an ExceptionGroup.
        async with anyio.create_task_group():
            yield RaisingAgent(LLMUnavailableError("400 INVALID_ARGUMENT: API key not valid."))

    monkeypatch.setattr(cli, "open_agent", task_group_agent)
    assert cli.main(["ask", "Berapa cuti?"]) == 1
    assert capsys.readouterr().err.strip().splitlines() == ["LLM tidak tersedia: 400 INVALID_ARGUMENT: API key not valid."]


def test_ask_without_index_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HR_INDEX_DIR", str(tmp_path / "missing"))
    assert cli.main(["ask", "Berapa cuti?"]) == 2
    assert "Index belum dibuat. Jalankan: python -m hr_assistant ingest" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()


def test_blank_question_rejected(capsys):
    assert cli.main(["ask", "   "]) == 2
    assert "Pertanyaan tidak boleh kosong." in capsys.readouterr().err


def test_missing_api_key_exits_2(built_index, monkeypatch, capsys):
    # An env var that exists, even empty, is not overridden by .env.
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert cli.main(["ask", "Berapa cuti?"]) == 2
    assert "GEMINI_API_KEY belum diisi" in capsys.readouterr().err


def test_llm_unavailable_prints_one_line(built_index, monkeypatch, capsys):
    patch_agent_to_raise(monkeypatch, LLMUnavailableError("kuota habis"))
    assert cli.main(["ask", "Berapa cuti?"]) == 1
    assert capsys.readouterr().err.strip().splitlines() == ["LLM tidak tersedia: kuota habis"]
