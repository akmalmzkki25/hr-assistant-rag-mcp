"""CLI: ingest dokumen, tanya jawab, chat, dan eval."""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import Client, StdioServerParameters

from hr_assistant.agent import AgentResult, HRAgent
from hr_assistant.config import PROJECT_ROOT, Settings, load_settings
from hr_assistant.evals import load_cases, run_eval
from hr_assistant.llm import GeminiLLM, LLMUnavailableError

NO_INDEX = "Index belum dibuat. Jalankan: python -m hr_assistant ingest"
NO_API_KEY = "GEMINI_API_KEY belum diisi. Salin .env.example ke .env lalu isi key."
EMPTY_QUESTION = "Pertanyaan tidak boleh kosong."


class UsageError(Exception):
    """Kesalahan input atau setup yang bisa diperbaiki pengguna (exit code 2)."""


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except UsageError as error:
        print(error, file=sys.stderr)
        return 2
    except LLMUnavailableError as error:
        print(f"LLM tidak tersedia: {error}", file=sys.stderr)
        return 1
    except BaseExceptionGroup as group:
        llm_errors = group.subgroup(LLMUnavailableError)
        if llm_errors is None:
            raise
        print(f"LLM tidak tersedia: {_first_leaf(llm_errors)}", file=sys.stderr)
        return 1


def _first_leaf(group: BaseExceptionGroup) -> BaseException:
    error: BaseException = group
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hr_assistant", description="Chatbot HR internal (RAG + MCP).")
    commands = parser.add_subparsers(required=True)

    ingest = commands.add_parser("ingest", help="Baca dokumen HR dan buat index vektor.")
    ingest.set_defaults(handler=_cmd_ingest)

    ask = commands.add_parser("ask", help="Ajukan satu pertanyaan.")
    ask.add_argument("question")
    ask.add_argument("--employee-id")
    ask.add_argument("--trace", action="store_true", help="Cetak setiap langkah agent sebagai JSON.")
    ask.set_defaults(handler=_cmd_ask)

    chat = commands.add_parser("chat", help="Tanya jawab interaktif.")
    chat.add_argument("--employee-id")
    chat.set_defaults(handler=_cmd_chat)

    evaluate = commands.add_parser("eval", help="Jalankan eval harness.")
    evaluate.add_argument("--cases", type=Path, default=PROJECT_ROOT / "evals" / "cases.json")
    evaluate.add_argument("--out", type=Path, default=PROJECT_ROOT / "logs" / "eval_report.json")
    evaluate.add_argument("--delay", type=float, default=4.0, help="Jeda antar kasus (detik), untuk kuota gratis.")
    evaluate.set_defaults(handler=_cmd_eval)
    return parser


def _cmd_ingest(args: argparse.Namespace) -> int:
    from hr_assistant.embeddings import FastEmbedEmbedder
    from hr_assistant.rag import build_index
    from hr_assistant.chunking import load_documents
    from hr_assistant.vector_store import ChromaStore

    settings = load_settings()
    documents = load_documents(settings.docs_dir)
    if not documents:
        raise UsageError(f"Tidak ada dokumen .md, .txt, atau .pdf di {settings.docs_dir}")
    store = ChromaStore(settings.index_dir, settings.collection)
    count = build_index(settings.docs_dir, store, FastEmbedEmbedder(settings.embedding_model))
    print(f"Index dibuat: {count} chunk dari {len(documents)} dokumen.")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    question = _require_question(args.question)
    settings = _ready_settings()

    async def run() -> AgentResult:
        async with open_agent(settings) as agent:
            return await agent.ask(question, employee_id=args.employee_id)

    result = asyncio.run(run())
    if args.trace:
        for step in result.to_dict()["steps"]:
            print(json.dumps(step, ensure_ascii=False))
    _print_answer(result)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    settings = _ready_settings()

    async def run() -> None:
        async with open_agent(settings) as agent:
            print("Ketik pertanyaan. Kosongkan lalu Enter untuk keluar.")
            while True:
                try:
                    question = (await asyncio.to_thread(input, "Kamu: ")).strip()
                except EOFError:
                    return
                if not question:
                    return
                _print_answer(await agent.ask(question, employee_id=args.employee_id))

    asyncio.run(run())
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    settings = _ready_settings()
    cases = load_cases(args.cases)

    async def run():
        async with open_agent(settings) as agent:
            return await run_eval(agent, cases, delay_seconds=args.delay)

    report = asyncio.run(run())
    for score in report.scores:
        failed = [name for name, ok in score.checks.items() if not ok]
        print(f"{'LULUS' if score.passed else 'GAGAL'}  {score.case_id:<26} route={score.route:<6} {' '.join(failed)}")
    print(f"Lulus: {report.passed}/{report.total}")
    print("Per cek: " + ", ".join(f"{name} {rate:.0%}" for name, rate in report.check_rates.items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Laporan: {args.out}")
    return 0


@asynccontextmanager
async def open_agent(settings: Settings) -> AsyncIterator[HRAgent]:
    from hr_assistant.embeddings import FastEmbedEmbedder
    from hr_assistant.rag import Retriever
    from hr_assistant.vector_store import ChromaStore

    store = ChromaStore(settings.index_dir, settings.collection)
    retriever = Retriever(store, FastEmbedEmbedder(settings.embedding_model), settings.top_k, settings.min_score)
    llm = GeminiLLM(settings.gemini_model, api_key=settings.gemini_api_key)
    # The MCP server runs as a separate process; tools are reached only through the protocol.
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "hr_assistant.mcp_server"], env=dict(os.environ), cwd=str(PROJECT_ROOT)
    )
    async with Client(params) as mcp:
        yield HRAgent(llm, mcp, retriever, max_steps=settings.max_steps)


def _require_question(question: str) -> str:
    if not question.strip():
        raise UsageError(EMPTY_QUESTION)
    return question.strip()


def _ready_settings() -> Settings:
    settings = load_settings()
    if not _index_ready(settings):
        raise UsageError(NO_INDEX)
    if not settings.gemini_api_key:
        raise UsageError(NO_API_KEY)
    return settings


def _index_ready(settings: Settings) -> bool:
    if not settings.index_dir.exists():
        return False
    from hr_assistant.vector_store import ChromaStore

    return ChromaStore(settings.index_dir, settings.collection).count() > 0


def _print_answer(result: AgentResult) -> None:
    print(result.answer)
    if result.sources:
        print("Sumber: " + "; ".join(result.sources))
