# HR Assistant v2 (LangGraph, Typesense, Postgres, Gemini Embedding) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Membangun ulang HR Assistant di atas LangGraph dan LangChain, dengan Typesense sebagai vector store (hybrid search), Postgres sebagai sumber data karyawan dan log run, serta `gemini-embedding-2` untuk embedding. Perilaku yang sudah terbukti (guardrail, eval 12/12, CLI) harus tetap sama.

**Architecture:**
- Agent berupa `StateGraph` LangGraph dengan node `agent` (Gemini via `langchain-google-genai`), `tools` (`ToolNode`), dan `guardrail`.
- Tool berasal dari dua sumber: `search_hr_documents` (LangChain `@tool` di atas retriever Typesense) dan tool MCP yang dimuat lewat `langchain-mcp-adapters` dari MCP server stdio (FastMCP).
- MCP server membaca Postgres. Setiap run agent dicatat ke tabel `agent_runs`.
- Guardrail tetap deterministik dan dipisah ke modul murni supaya bisa dites tanpa LLM.

**Tech Stack:** Python 3.12, langgraph 1.2, langchain-core 1.7, langchain-google-genai 4.4, langchain-mcp-adapters 0.3.2, mcp 1.30 (FastMCP), langchain-text-splitters 1.1, typesense 2.1 (server 0.25.2), psycopg 3.3, pydantic-settings 2.16, google-genai (embedding).

Plan v1: `docs/superpowers/plans/2026-11-06-hr-assistant.md`.

## Global Constraints

- Semua constraint v1 tetap berlaku: nama tool MCP persis `get_leave_balance(employee_id)` dan `get_subordinat_list(employee_id)`, tool hanya lewat protokol MCP, `UNKNOWN_ANSWER` sama persis, tanpa UI dan tanpa deploy cloud, serta teks Bahasa Indonesia tanpa em dash.
- `mcp` turun ke 1.30 karena `langchain-mcp-adapters` 0.3.2 mensyaratkan `mcp<2`. Server memakai `mcp.server.fastmcp.FastMCP`.
- Embedding memakai `gemini-embedding-2` dengan `output_dimensionality=768`.
  - Format teks: dokumen `title: <section> | text: <isi>`, query `task: search result | query: <pertanyaan>`.
  - Batch dikirim sebagai list `types.Content`, karena list string digabung menjadi satu embedding.
- Typesense: `http://localhost:8108`, key dari `TYPESENSE_API_KEY`, collection `hr_docs`. Query memakai hybrid search (`q` + `vector_query`).
- Postgres:
  - `DATABASE_URL` default `postgresql://postgres:<password>@localhost:5432/hr_assistant`.
  - Database `hr_assistant` dibuat oleh `python -m hr_assistant db-init`, dan database lain di server tidak boleh disentuh.
  - Test memakai database `hr_assistant_test` dan collection `hr_docs_test`.
- Rahasia hanya di `.env`. `.env.example` memuat nama variabel tanpa nilai rahasia.
- Test integrasi (Postgres, Typesense, MCP stdio) diberi penanda `integration` dan dilewati otomatis kalau servis tidak bisa dihubungi. Test unit tidak memakai jaringan.
- Pustaka dipakai menggantikan kode tulisan sendiri: text splitter LangChain, `ToolNode`, adapter MCP, pydantic-settings.

## Review Focus

1. Pertanyaan lanjutan di `chat` ("kalau carry over?") memakai riwayat percakapan, tetapi guardrail tetap menilai per giliran, sehingga sitasi dari giliran sebelumnya tidak dihitung. Test ada di Task 5.
2. Typesense mati atau Postgres mati: CLI mencetak satu baris pesan yang menyebut servisnya, bukan traceback. Test ada di Task 6.
3. Kuota embedding Gemini habis saat `ingest`: retry dengan backoff, lalu pesan satu baris. Test ada di Task 2.
4. `db-init` dijalankan dua kali: idempoten dan tidak menggandakan data seed. Test ada di Task 3.
5. Dokumen PDF ikut ter-index, dan ada sampel PDF di `data/docs/`. Test ada di Task 1.

---

### Task 1: Config, loader, dan chunking dengan LangChain splitter

**Files:** Modify `pyproject.toml`, `.env.example`, `src/hr_assistant/config.py`; Create `src/hr_assistant/rag/__init__.py`, `rag/loader.py`, `rag/chunking.py`, `data/docs/pedoman_perjalanan_dinas.pdf`; Delete `src/hr_assistant/chunking.py`; Test `tests/test_chunking.py`.

**Interfaces:**
- `Settings(BaseSettings)` memuat `gemini_api_key`, `gemini_model`, `embedding_model="gemini-embedding-2"`, `embedding_dim=768`, `typesense_url`, `typesense_api_key`, `typesense_collection="hr_docs"`, `database_url`, `docs_dir`, `top_k=4`, `min_score`, `max_steps=6`. Dibaca lewat `get_settings()`.
- `load_documents(docs_dir: Path) -> list[Document]`, dengan `metadata["source"]` berupa path relatif.
- `split_documents(docs: list[Document], chunk_size: int = 1200, chunk_overlap: int = 150) -> list[Document]`.
  - Metadata yang dihasilkan: `source`, `section` (path heading digabung `" > "`, atau stem nama berkas kalau tidak ada heading), dan `chunk_id` (`"{source}::{n}"`).
  - `page_content` diawali `section`.

- [ ] Test: heading menjadi `section`, potongan panjang memakai overlap, teks sebelum heading pertama dibuang, teks tanpa heading memakai stem, dan PDF terbaca. Jalankan test sampai FAIL, implementasikan, lalu PASS. Commit.

### Task 2: Gemini embeddings, Typesense store, dan retriever hybrid

**Files:** Create `rag/embeddings.py`, `rag/store.py`, `rag/retriever.py`, `rag/ingest.py`; Delete `embeddings.py`, `vector_store.py`, `rag.py`; Test `tests/test_embeddings.py`, `tests/test_store.py` (integration).

**Interfaces:**
- `GeminiEmbeddings(Embeddings)`, dibuat dengan `model`, `dimensions`, `api_key`, `client=None`, `max_retries=3`, `sleep=time.sleep`, `batch_size=32`.
  - `embed_documents` mengirim list `Content` berformat `title:`, dan `embed_query` memakai format `task: search result`.
  - 429 atau 5xx dicoba ulang, lalu memunculkan `EmbeddingUnavailableError`.
- `SearchHit(document: Document, similarity: float, text_match: int)`.
- `TypesenseStore(client, collection, dim)`, dengan:
  - `recreate()`
  - `upsert(docs, vectors)`
  - `search(query: str, vector: list[float], k: int) -> list[SearchHit]` (hybrid, `similarity = 1 - vector_distance`)
  - `count() -> int`
  - `make_typesense_client(settings)`
- `HRRetriever(store, embeddings, top_k, min_score)`, dengan `search(query, k=None)` dan `relevant(query)`.
- `build_index(docs_dir, store, embeddings) -> int`.

- [ ] Test unit embeddings dengan client palsu: format prefix, satu embedding per dokumen, retry 429, dan 400 tidak dicoba ulang.
- [ ] Test integrasi store dengan embedder hash 768 dimensi: upsert dan count, ingest idempoten, top-1 benar untuk "cuti tahunan", serta hybrid menemukan kata kunci persis.
- [ ] Kalibrasi `min_score` dengan embedding sungguhan pada pertanyaan HR vs di luar dokumen, lalu catat sebagai ruling.
- [ ] Commit.

### Task 3: Postgres: schema, seed, repository, dan db-init

**Files:** Create `src/hr_assistant/db/__init__.py`, `db/schema.sql`, `db/seed.sql`, `db/database.py`; Delete `data/employees.json`; Test `tests/test_db.py` (integration).

**Interfaces:**
- Tabel `employees(employee_id PK, name, position, manager_id FK nullable)`.
- Tabel `leave_balances(employee_id FK, year, annual_quota, annual_used, sick_days_used, PK(employee_id, year))`.
- Tabel `agent_runs(id bigserial, created_at, question, employee_id, answer, route, stop_reason, sources jsonb, steps jsonb, latency_ms)`.
- Seed berisi data dummy v1 (E001, E002, E003, tahun 2026), ditulis memakai `ON CONFLICT DO NOTHING`.
- `init_database(database_url: str) -> None`: membuat database kalau belum ada, lalu menerapkan schema dan seed.
- `EmployeeRepository(database_url)`, dengan:
  - `leave_balance(employee_id) -> dict | None`
  - `employee(employee_id) -> dict | None`
  - `subordinates(employee_id) -> list[dict]`
- `RunLogRepository(database_url)` dengan `log(result, question, employee_id, latency_ms) -> None`.

- [ ] Test: init idempoten (dua kali, jumlah baris tetap), E001 sisa 7, bawahan E002 dan E003 terurut, ID tak dikenal `None`, dan log run tersimpan. Commit.

### Task 4: MCP server FastMCP di atas Postgres

**Files:** Modify `src/hr_assistant/mcp_server.py`; Test `tests/test_mcp_server.py`.

**Interfaces:**
- `server = FastMCP("hr-tools", log_level="WARNING")`.
- Tool `get_leave_balance` dan `get_subordinat_list` dengan bentuk keluaran seperti v1, tetapi data diambil dari `EmployeeRepository(get_settings().database_url)`.
- ID tidak dikenal memunculkan `ToolError("Karyawan dengan ID {id} tidak ditemukan.")`.

- [ ] Test integrasi lewat sesi in-memory dan lewat stdio: nama tool persis, E001 sisa 7, error E999, serta stderr kosong untuk E999. Commit.

### Task 5: Agent LangGraph dan guardrail

**Files:** Create `src/hr_assistant/agent/__init__.py`, `agent/prompts.py`, `agent/guardrail.py`, `agent/tools.py`, `agent/graph.py`; Delete `agent.py`, `prompts.py`, `llm.py`; Test `tests/test_guardrail.py`, `tests/test_graph.py`.

**Interfaces:**
- `guardrail.check(answer: str, retrieved_ids: set[str], mcp_outputs: list[str], used_tools: bool) -> Verdict(kind: Literal["grounded", "unknown", "clarify", "ungrounded"], route, sources)`. Aturannya sama dengan v1 setelah perbaikan review.
- `make_search_tool(retriever) -> BaseTool` dengan `response_format="content_and_artifact"`. Artifact-nya adalah list `source_id`.
- `AgentState(MessagesState)` menambah `retrieved_ids`, `mcp_outputs`, `used_tools`, `llm_calls`, `retries`, `verdict`.
- `build_graph(llm: BaseChatModel, tools: list[BaseTool], max_steps: int, max_retries: int, checkpointer=None)`.
- `HRAgent(graph)` dengan `async ask(question, employee_id=None, thread_id=None) -> AgentResult`. `AgentResult` sama dengan v1.
- `UNKNOWN_ANSWER` diekspor dari `hr_assistant.agent`.

- [ ] Test guardrail murni mencakup semua kasus test v1 yang relevan.
- [ ] Test graph dengan `FakeToolChatModel` (turunan `GenericFakeChatModel` dengan `bind_tools`), search tool berbasis retriever palsu, dan tool MCP dari server test in-memory. Kasus yang dites: RAG dengan sitasi, MCP, retry lalu unknown, max_steps, memori thread untuk pertanyaan lanjutan, dan sitasi giliran sebelumnya yang tidak dihitung.
- [ ] Commit.

### Task 6: CLI, eval, dan log run

**Files:** Modify `src/hr_assistant/cli.py`, `evals.py`; Test `tests/test_cli.py`, `tests/test_evals.py`.

**Interfaces:**
- Perintah `db-init`, `ingest`, `ask`, `chat` (dengan memori thread), dan `eval`.
- `open_agent(settings)` merakit Gemini, retriever Typesense, `MultiServerMCPClient` stdio, dan `HRAgent`.
- Setiap `ask` dicatat ke `agent_runs`, dan kegagalan log tidak menggagalkan jawaban.
- Pesan satu baris untuk: Typesense tidak bisa dihubungi, Postgres tidak bisa dihubungi, index kosong, key kosong, LLM atau embedding tidak tersedia, dan Ctrl+C.

- [ ] Test semua jalur error di atas. Commit.

### Task 7: Dokumentasi, uji live, dan review

- [ ] Perbarui `docs/architecture.md` dan `README.md` (komponen baru beserta justifikasinya, setup Docker, alur, trade-off, eval, dan status).
- [ ] Jalankan `db-init`, `ingest`, empat demo `ask`, `chat` dua giliran, dan `eval` dengan Gemini asli.
- [ ] Lakukan review akhir oleh reviewer baru, lalu satu pass perbaikan dengan TDD.
