"""CLI: siapkan database, ingest dokumen, tanya jawab, chat, dan eval."""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from hr_assistant.agent.graph import AgentResult, HRAgent
from hr_assistant.config import PROJECT_ROOT, Settings, get_settings
from hr_assistant.errors import SetupError

NO_INDEX = "Index belum dibuat. Jalankan: python -m hr_assistant ingest"
NO_API_KEY = "GEMINI_API_KEY belum diisi. Salin .env.example ke .env lalu isi key."
EMPTY_QUESTION = "Pertanyaan tidak boleh kosong."


EXIT_WORDS = {"keluar", "exit", "quit"}

UsageError = SetupError


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except UsageError as error:
        print(error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Dibatalkan.", file=sys.stderr)
        return 130
    except BaseException as error:
        message = _describe_service_error(error)
        if message is None:
            raise
        print(message, file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hr_assistant", description="Chatbot HR internal (RAG + MCP).")
    commands = parser.add_subparsers(required=True)

    commands.add_parser("db-init", help="Buat database dan jalankan migrasi Alembic.").set_defaults(handler=_cmd_db_init)
    commands.add_parser("ingest", help="Baca dokumen HR dan buat index di Typesense.").set_defaults(handler=_cmd_ingest)

    ask = commands.add_parser("ask", help="Ajukan satu pertanyaan.")
    ask.add_argument("question")
    ask.add_argument("--employee-id")
    ask.add_argument("--trace", action="store_true", help="Cetak setiap langkah agent sebagai JSON.")
    ask.set_defaults(handler=_cmd_ask)

    chat = commands.add_parser("chat", help="Tanya jawab interaktif dengan memori percakapan.")
    chat.add_argument("--employee-id")
    chat.set_defaults(handler=_cmd_chat)

    evaluate = commands.add_parser("eval", help="Jalankan eval harness.")
    evaluate.add_argument("--cases", type=Path, default=PROJECT_ROOT / "evals" / "cases.json")
    evaluate.add_argument("--out", type=Path, default=PROJECT_ROOT / "logs" / "eval_report.json")
    evaluate.add_argument("--delay", type=float, default=4.0, help="Jeda antar kasus (detik), untuk kuota gratis.")
    evaluate.set_defaults(handler=_cmd_eval)
    return parser


def _cmd_db_init(args: argparse.Namespace) -> int:
    from hr_assistant.db.database import init_database, migration_head

    init_database(get_settings().database_url)
    print(f"Database siap, migrasi sampai revision {migration_head()}.")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from hr_assistant.rag.chunking import split_documents
    from hr_assistant.rag.ingest import build_index
    from hr_assistant.rag.loader import load_documents

    settings = get_settings()
    documents = load_documents(settings.docs_dir)
    if not documents:
        raise UsageError(f"Tidak ada dokumen .md, .txt, atau .pdf di {settings.docs_dir}")
    if not split_documents(documents):
        raise UsageError(f"Tidak ada teks yang bisa di-index di {settings.docs_dir}. Untuk PDF hasil scan, jalankan OCR dulu.")
    _require_api_key(settings)
    store = _store(settings)
    _require_typesense(store, settings)
    count = build_index(settings.docs_dir, store, _embeddings(settings))
    print(f"Index dibuat: {count} chunk dari {len(documents)} dokumen.")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    question = args.question.strip()
    if not question:
        raise UsageError(EMPTY_QUESTION)
    settings = _ready_settings()

    async def run() -> tuple[AgentResult, int]:
        async with open_agent(settings) as agent:
            started = time.monotonic()
            result = await agent.ask(question, employee_id=args.employee_id)
            return result, int((time.monotonic() - started) * 1000)

    result, latency_ms = asyncio.run(run())
    if args.trace:
        for step in result.to_dict()["steps"]:
            print(json.dumps(step, ensure_ascii=False))
    _print_answer(result)
    _log_run(settings, question, args.employee_id, result, latency_ms)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    settings = _ready_settings()
    thread_id = uuid4().hex

    async def run() -> None:
        async with open_agent(settings) as agent:
            print("Ketik pertanyaan. Ketik keluar, atau kosongkan lalu Enter, untuk selesai.")
            while True:
                try:
                    question = (await _read_line("Kamu: ")).strip()
                except EOFError:
                    return
                if not question or question.casefold() in EXIT_WORDS:
                    return
                started = time.monotonic()
                result = await agent.ask(question, employee_id=args.employee_id, thread_id=thread_id)
                _print_answer(result)
                _log_run(settings, question, args.employee_id, result, int((time.monotonic() - started) * 1000))

    asyncio.run(run())
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from hr_assistant.evals import load_cases, run_eval

    settings = _ready_settings()
    cases = load_cases(args.cases)

    async def run():
        async with open_agent(settings) as agent:
            return await run_eval(agent, cases, delay_seconds=args.delay)

    report = asyncio.run(run())
    for score in report.scores:
        failed = [name for name, ok in score.checks.items() if not ok]
        print(f"{'LULUS' if score.passed else 'GAGAL'}  {score.case_id:<28} route={score.route:<6} {' '.join(failed)}")
    print(f"Lulus: {report.passed}/{report.total}")
    print("Per cek: " + ", ".join(f"{name} {rate:.0%}" for name, rate in report.check_rates.items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Laporan: {args.out}")
    return 0


@asynccontextmanager
async def open_agent(settings: Settings) -> AsyncIterator[HRAgent]:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    from hr_assistant.agent.graph import build_graph
    from hr_assistant.agent.tools import make_search_tool
    from hr_assistant.rag.retriever import HRRetriever

    retriever = HRRetriever(_store(settings), _embeddings(settings), settings.top_k, settings.min_score)
    llm = ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.gemini_api_key, max_retries=3)
    # The MCP server runs as a separate process; tools are reached only through the protocol.
    servers = {
        "hr": {
            "command": sys.executable,
            "args": ["-m", "hr_assistant.mcp_server"],
            "transport": "stdio",
            "env": dict(os.environ),
            "cwd": str(PROJECT_ROOT),
        }
    }
    async with MultiServerMCPClient(servers).session("hr") as session:
        tools = [make_search_tool(retriever), *await load_mcp_tools(session)]
        yield HRAgent(build_graph(llm, tools, max_steps=settings.max_steps), max_steps=settings.max_steps)


def _ready_settings() -> Settings:
    settings = get_settings()
    _require_api_key(settings)
    store = _store(settings)
    _require_typesense(store, settings)
    if store.count() == 0:
        raise UsageError(NO_INDEX)
    return settings


def _require_api_key(settings: Settings) -> None:
    if not settings.gemini_api_key:
        raise UsageError(NO_API_KEY)


def _require_typesense(store, settings: Settings) -> None:
    if not store.is_reachable():
        raise UsageError(f"Typesense tidak bisa dihubungi di {settings.typesense_url}. Nyalakan container: docker start typesense")


def _store(settings: Settings):
    from hr_assistant.rag.store import TypesenseStore, make_typesense_client

    return TypesenseStore(make_typesense_client(settings), settings.typesense_collection, settings.embedding_dim)


def _embeddings(settings: Settings):
    from hr_assistant.rag.embeddings import GeminiEmbeddings

    return GeminiEmbeddings(settings.embedding_model, settings.embedding_dim, api_key=settings.gemini_api_key)


def _log_run(settings: Settings, question: str, employee_id: str | None, result: AgentResult, latency_ms: int) -> None:
    """Log run untuk sampling eval di production; kegagalannya tidak boleh menggagalkan jawaban."""
    from hr_assistant.db.database import RunLogRepository, RunRecord

    record = RunRecord(
        question, employee_id, result.answer, result.route, result.stop_reason,
        result.sources, result.to_dict()["steps"], latency_ms,
    )
    try:
        RunLogRepository(settings.database_url).log(record)
    except Exception as error:
        print(f"Log run tidak tersimpan: {_first_line(error)}", file=sys.stderr)


def _describe_service_error(error: BaseException) -> str | None:
    import psycopg
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

    from hr_assistant.rag.embeddings import EmbeddingUnavailableError
    from hr_assistant.rag.store import SearchUnavailableError

    labels = (
        (ChatGoogleGenerativeAIError, "LLM tidak tersedia"),
        (EmbeddingUnavailableError, "Embedding Gemini tidak tersedia"),
        (psycopg.OperationalError, "Postgres tidak bisa dihubungi"),
        # This message already names Typesense.
        (SearchUnavailableError, None),
    )
    for leaf in _leaves(error):
        for error_type, label in labels:
            if isinstance(leaf, error_type):
                return f"{label}: {_first_line(leaf)}" if label else _first_line(leaf)
    return None


def _leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for inner in error.exceptions for leaf in _leaves(inner)]
    return [error, *([_leaves(error.__cause__)[0]] if error.__cause__ else [])]


def _first_line(error: BaseException) -> str:
    return (str(error).strip().splitlines() or [type(error).__name__])[0]


async def _read_line(prompt: str) -> str:
    """Baca input di daemon thread: asyncio.to_thread akan menahan Ctrl+C sampai Enter ditekan."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def read() -> None:
        try:
            line = input(prompt)
        except BaseException as error:
            loop.call_soon_threadsafe(future.set_exception, error)
        else:
            loop.call_soon_threadsafe(future.set_result, line)

    threading.Thread(target=read, daemon=True).start()
    return await future


def _print_answer(result: AgentResult) -> None:
    print(result.answer)
    if result.sources:
        print("Sumber: " + "; ".join(result.sources))
