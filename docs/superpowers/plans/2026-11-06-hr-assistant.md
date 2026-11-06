# HR Assistant (RAG + MCP Agent) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chatbot HR internal berbasis CLI yang menjawab pertanyaan dari dokumen HR (RAG) dan mengambil data karyawan lewat MCP server, dengan guardrail "tidak tahu" dan eval harness.

**Architecture:** Dokumen Markdown/PDF dipecah per heading, di-embed secara lokal dengan fastembed, lalu disimpan di Chroma. Agent memakai Gemini function calling dalam loop bertahap yang dibatasi jumlah langkahnya. Ada dua sumber tool: `search_hr_documents` (RAG lokal) dan tool MCP yang ditemukan lewat `list_tools` dari MCP server stdio. Setelah loop, guardrail deterministik memeriksa grounding dan sitasi. Kalau gagal, agent diberi satu kesempatan revisi, lalu jawabannya diganti dengan jawaban baku "tidak tahu".

**Tech Stack:** Python 3.12, `mcp` 2.3 (`MCPServer`, `Client`), `google-genai` 3.0, `chromadb` 1.5, `fastembed` 0.9 (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`), `pytest` 9.

**Kebutuhan:** chatbot HR internal dengan RAG atas dokumen HR, MCP server untuk data karyawan, agent dengan guardrail "tidak tahu", dan dokumentasi arsitektur.

## Global Constraints

- Bahasa: Python 3.12, layout `src/hr_assistant/`, dijalankan dengan `.venv/Scripts/python.exe`.
- Dependensi: `mcp>=2.3,<3`, `google-genai>=3.0`, `chromadb>=1.5`, `fastembed>=0.9`, `pypdf>=5`, `python-dotenv>=1`; dev: `pytest>=9`.
- Nama tool MCP persis dari spec: `get_leave_balance(employee_id)` dan `get_subordinat_list(employee_id)`.
- Agent wajib memanggil tool lewat protokol MCP (subprocess stdio di CLI, `Client(server)` in-process hanya di test). Agent tidak boleh mengimpor fungsi tool secara langsung.
- Guardrail: kalau informasi tidak ada di dokumen, jawabannya persis `UNKNOWN_ANSWER = "Maaf, saya tidak tahu. Informasi itu tidak ada di dokumen HR yang saya punya."`
- LLM: Gemini dengan `GEMINI_API_KEY` dari environment atau `.env`, dan model dari `GEMINI_MODEL` (default `gemini-3.8-flash`).
- Tidak ada UI dan tidak ada deploy cloud. Demo lokal lewat `python -m hr_assistant ...`.
- Teks untuk manusia (README, docs, prompt) dalam Bahasa Indonesia, tanpa em dash. Komentar kode hanya untuk hal yang tidak terlihat dari kode (aturan antislop-code).
- Test tidak boleh memanggil jaringan: embedder palsu, LLM skrip, dan MCP in-process. Satu-satunya test yang menyentuh subprocess adalah test stdio MCP.

## Review Focus

1. ID karyawan yang tidak dikenal (`E999`): tool MCP mengembalikan error yang terbaca, dan agent menyampaikan bahwa ID tidak ditemukan tanpa mengarang angka cuti. Test ada di Task 3 dan Task 5.
2. Pertanyaan di luar dokumen ("berapa gaji direktur?"): hasilnya `UNKNOWN_ANSWER`, meskipun LLM sempat menulis jawaban tanpa sitasi. Test ada di Task 5.
3. Index belum dibuat (`ask` dijalankan sebelum `ingest`): CLI menampilkan pesan "Index belum dibuat. Jalankan: python -m hr_assistant ingest" dengan exit code 2, bukan traceback. Test ada di Task 6.
4. Kuota gratis Gemini habis (HTTP 429): adapter mencoba ulang 3 kali dengan backoff, lalu memunculkan `LLMUnavailableError` yang ditampilkan CLI sebagai satu baris pesan. Test ada di Task 4.
5. Pertanyaan kosong atau hanya spasi: CLI menolaknya dengan pesan "Pertanyaan tidak boleh kosong." dan exit code 2. Test ada di Task 6.

---

### Task 1: Scaffold, dokumen contoh, dan chunking

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `src/hr_assistant/__init__.py`, `src/hr_assistant/config.py`, `src/hr_assistant/chunking.py`
- Create: `data/docs/kebijakan_cuti.md`, `data/docs/kerja_remote.md`, `data/docs/reimbursement.md`, `data/docs/absensi_lembur.md`, `data/docs/faq_karyawan.md`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Produces: `Chunk(id: str, text: str, source: str, section: str)` (frozen dataclass); `chunk_markdown(text: str, source: str, max_chars: int = 700, overlap_chars: int = 100) -> list[Chunk]`; `load_documents(docs_dir: Path) -> list[tuple[str, str]]` (nama berkas, isi) untuk `.md`, `.txt`, `.pdf`; `Settings` (frozen) dengan `docs_dir`, `index_dir`, `collection="hr_docs"`, `embedding_model`, `gemini_model`, `top_k=4`, `min_score=0.45`, `max_steps=6`, `employee_data`; `load_settings() -> Settings` (membaca env dan `.env`, path relatif terhadap root proyek).

- [ ] **Step 1:** `git init`, buat `.gitignore` (`.venv/`, `.index/`, `.env`, `__pycache__/`, `.pytest_cache/`, `logs/`), lalu `pyproject.toml` dengan setuptools, src layout, dan dependensi dari Global Constraints. Jalankan `uv pip install --python .venv/Scripts/python.exe -e ".[dev]"`. Hasil yang diharapkan: install sukses.
- [ ] **Step 2:** Tulis lima dokumen contoh fiktif berbahasa Indonesia, masing-masing 3 sampai 6 bagian `##`, dan tandai sebagai contoh di baris pertama. Fakta yang dipakai test di task lain:
  - `kebijakan_cuti.md` bagian `Cuti Tahunan`: "12 hari kerja per tahun"
  - `kebijakan_cuti.md` bagian `Cuti Melahirkan`: "3 bulan"
  - `kerja_remote.md`: "maksimal 2 hari per minggu"
  - `reimbursement.md`: "paling lambat 30 hari"
- [ ] **Step 3: Write the failing tests** di `tests/test_chunking.py`:

```python
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

def test_load_documents_reads_markdown(tmp_path):
    (tmp_path / "x.md").write_text("## A\nisi", encoding="utf-8")
    (tmp_path / "skip.png").write_bytes(b"\x89PNG")
    assert load_documents(tmp_path) == [("x.md", "## A\nisi")]
```

- [ ] **Step 4:** Jalankan `.venv/Scripts/python.exe -m pytest tests/test_chunking.py -v`. Hasil yang diharapkan: FAIL (ImportError).
- [ ] **Step 5:** Implementasikan `chunk_markdown` dan `load_documents`.
  - Section dipecah di heading `#`/`##`/`###`, dan path heading digabung dengan `" > "`.
  - Bagian yang panjang dipecah per kalimat sampai `max_chars`, dengan ekor `overlap_chars` dari potongan sebelumnya.
  - `text` diawali baris path heading, supaya konteks ikut ter-embed.
  - PDF dibaca lewat `pypdf.PdfReader`, dengan teks per halaman digabung.
  - Lalu implementasikan `Settings` dan `load_settings`.
- [ ] **Step 6:** Jalankan test yang sama. Hasil yang diharapkan: PASS.
- [ ] **Step 7:** Commit `feat: scaffold project, sample HR docs, markdown chunker`.

### Task 2: Embedding, vector store, retrieval top-k, dan ingest

**Files:**
- Create: `src/hr_assistant/embeddings.py`, `src/hr_assistant/vector_store.py`, `src/hr_assistant/rag.py`, `tests/fakes.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `Chunk`, `chunk_markdown`, `load_documents` (Task 1).
- Produces:
  - `Embedder` (Protocol): `embed_documents(texts: Sequence[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`
  - `FastEmbedEmbedder(model_name: str)`
  - `RetrievedChunk(chunk: Chunk, score: float)` (frozen)
  - `ChromaStore(path: Path, collection: str)`, dengan `upsert(chunks, vectors) -> None`, `query(vector, k: int) -> list[RetrievedChunk]`, `count() -> int`, `reset() -> None`
  - `build_index(docs_dir: Path, store: ChromaStore, embedder: Embedder) -> int`
  - `Retriever(store, embedder, top_k: int = 4, min_score: float = 0.45)`, dengan `search(query: str, k: int | None = None) -> list[RetrievedChunk]` (top-k mentah, skor menurun) dan `relevant(query: str) -> list[RetrievedChunk]` (hanya yang `score >= min_score`)
  - `tests/fakes.py::HashEmbedder(dim: int = 256)`: bag-of-words yang di-hash dan dinormalisasi L2, deterministik tanpa jaringan.

- [ ] **Step 1: Write the failing tests** di `tests/test_rag.py`. Pakai `HashEmbedder` dan `tmp_path` untuk index serta dokumen.

```python
def test_build_index_counts_chunks(tmp_path, sample_docs):
    store = ChromaStore(tmp_path / "idx", "hr_docs")
    n = build_index(sample_docs, store, HashEmbedder())
    assert n == store.count() > 0

def test_build_index_is_idempotent(tmp_path, sample_docs):
    store = ChromaStore(tmp_path / "idx", "hr_docs")
    first = build_index(sample_docs, store, HashEmbedder())
    assert build_index(sample_docs, store, HashEmbedder()) == first == store.count()

def test_search_returns_top_k_sorted(retriever):
    hits = retriever.search("berapa hari cuti tahunan", k=3)
    assert len(hits) == 3
    assert hits[0].chunk.section.endswith("Cuti Tahunan")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

def test_relevant_filters_by_min_score(retriever):
    assert retriever.relevant("zzqx kuantum nebula") == []
```

  `sample_docs` dan `retriever` adalah fixture di `tests/conftest.py` yang menulis dua dokumen kecil (berisi `Cuti Tahunan` dan `Kerja Remote`) ke `tmp_path`.
- [ ] **Step 2:** Jalankan `.venv/Scripts/python.exe -m pytest tests/test_rag.py -v`. Hasil yang diharapkan: FAIL.
- [ ] **Step 3:** Implementasikan modul-modulnya.
  - Koleksi Chroma dibuat dengan `configuration={"hnsw": {"space": "cosine"}}`, dan `score = 1 - distance`.
  - Metadata yang disimpan: `source` dan `section`.
  - `build_index` memanggil `reset()` lalu upsert dalam batch 64.
  - `FastEmbedEmbedder` memakai `TextEmbedding(model_name).embed(...)`, lalu hasilnya diubah ke list float.
- [ ] **Step 4:** Jalankan test yang sama. Hasil yang diharapkan: PASS.
- [ ] **Step 5:** Commit `feat: chroma vector store, top-k retriever, ingest`.

### Task 3: MCP server dengan dua tool

**Files:**
- Create: `src/hr_assistant/mcp_server.py`, `data/employees.json`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Produces:
  - `server: MCPServer` bernama `"hr-tools"`.
  - Tool `get_leave_balance(employee_id: str) -> dict`, yang mengembalikan `employee_id`, `name`, `year`, `annual_quota`, `annual_used`, `annual_remaining`, dan `sick_days_used`.
  - Tool `get_subordinat_list(employee_id: str) -> dict`, yang mengembalikan `employee_id`, `name`, dan `subordinates: [{employee_id, name, position}]`.
  - Entry point `python -m hr_assistant.mcp_server` (stdio).
  - Path data diambil dari env `HR_EMPLOYEE_DATA`, dengan default `data/employees.json`.
  - ID dinormalisasi dengan `strip().upper()`. ID yang tidak dikenal memunculkan `mcp.server.mcpserver.exceptions.ToolError("Karyawan dengan ID {id} tidak ditemukan.")`.
- Data dummy: `E001` Rina Wijaya (Engineering Manager, kuota 12, terpakai 5, sisa 7, sakit 1, bawahan `E002` dan `E003`), `E002` Budi Santoso (Backend Engineer, kuota 12, terpakai 10, sisa 2, tanpa bawahan), `E003` Sari Lestari (Data Analyst, kuota 12, terpakai 0, sisa 12), dan `year` 2026.

- [ ] **Step 1: Write the failing tests.** Pakai `async with Client(server) as c` dan `pytest.mark.anyio` (atau `asyncio.run` di dalam test sync).

```python
async def test_leave_balance_known_employee():
    res = await call("get_leave_balance", {"employee_id": " e001 "})
    assert not res.is_error and json.loads(res.content[0].text)["annual_remaining"] == 7

async def test_subordinates():
    data = json.loads((await call("get_subordinat_list", {"employee_id": "E001"})).content[0].text)
    assert [s["employee_id"] for s in data["subordinates"]] == ["E002", "E003"]

async def test_unknown_employee_is_tool_error():
    res = await call("get_leave_balance", {"employee_id": "E999"})
    assert res.is_error and "E999" in res.content[0].text

async def test_tools_listed_with_exact_names():
    async with Client(server) as c:
        assert {t.name for t in (await c.list_tools()).tools} == {"get_leave_balance", "get_subordinat_list"}

async def test_stdio_transport_roundtrip():
    params = StdioServerParameters(command=sys.executable, args=["-m", "hr_assistant.mcp_server"])
    async with Client(params) as c:
        res = await c.call_tool("get_leave_balance", {"employee_id": "E002"})
    assert json.loads(res.content[0].text)["annual_remaining"] == 2
```

- [ ] **Step 2:** Jalankan `.venv/Scripts/python.exe -m pytest tests/test_mcp_server.py -v`. Hasil yang diharapkan: FAIL.
- [ ] **Step 3:** Implementasikan server dengan `@server.tool()` dan docstring Bahasa Indonesia sebagai deskripsi tool. LLM membaca deskripsi ini untuk memilih tool. `main()` memanggil `server.run("stdio")`.
- [ ] **Step 4:** Jalankan test yang sama. Hasil yang diharapkan: PASS, termasuk test stdio. Kalau `ToolError` ternyata tidak meneruskan pesan, sesuaikan dengan pola di `mcp/server/mcpserver/exceptions.py`, dan pertahankan assert `"E999" in text`.
- [ ] **Step 5:** Commit `feat: MCP server with leave balance and subordinate tools`.

### Task 4: Adapter LLM (Gemini) dan LLM skrip untuk test

**Files:**
- Create: `src/hr_assistant/llm.py`
- Modify: `tests/fakes.py` (tambah `ScriptedLLM`)
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `ToolSpec(name: str, description: str, parameters: dict)`
  - `ToolCall(id: str, name: str, args: dict)`
  - `LLMReply(text: str, tool_calls: tuple[ToolCall, ...], raw: Any = None)`
  - `Message(role: Literal["user", "assistant", "tool"], text: str = "", tool_calls: tuple[ToolCall, ...] = (), tool_name: str = "", tool_result: dict | None = None, raw: Any = None)`
  - `LLM` (Protocol): `generate(system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMReply`
  - `LLMUnavailableError(Exception)`
  - `GeminiLLM(model: str, api_key: str | None = None, max_retries: int = 3, sleep: Callable[[float], None] = time.sleep)`
  - `to_gemini_tools(tools) -> list[types.Tool]` dan `to_gemini_contents(messages) -> list[types.Content]`
  - `tests/fakes.py::ScriptedLLM(replies: list[LLMReply])`, yang mencatat setiap panggilan di `.calls` dan memunculkan `AssertionError` kalau balasan habis.
- Gemini 3 menolak riwayat function call yang kehilangan thought signature. Jadi pesan `assistant` membawa `raw` (objek `types.Content` asli dari respons), dan `to_gemini_contents` memakai `raw` itu apa adanya kalau ada.

- [ ] **Step 1: Write the failing tests:**

```python
def test_tools_use_json_schema():
    spec = ToolSpec("get_leave_balance", "Sisa cuti", {"type": "object", "properties": {"employee_id": {"type": "string"}}, "required": ["employee_id"]})
    decl = to_gemini_tools([spec])[0].function_declarations[0]
    assert decl.name == "get_leave_balance" and decl.parameters_json_schema["required"] == ["employee_id"]

def test_tool_message_becomes_function_response():
    content = to_gemini_contents([Message(role="tool", tool_name="get_leave_balance", tool_result={"annual_remaining": 7})])[0]
    assert content.parts[0].function_response.response == {"annual_remaining": 7}

def test_assistant_raw_content_passed_through():
    raw = types.Content(role="model", parts=[types.Part.from_text(text="hai")])
    assert to_gemini_contents([Message(role="assistant", raw=raw)])[0] is raw

def test_rate_limit_retries_then_raises(monkeypatch):
    llm = GeminiLLM("m", api_key="x", max_retries=3, sleep=lambda s: None)
    calls = fake_generate_raising(monkeypatch, llm, code=429)
    with pytest.raises(LLMUnavailableError):
        llm.generate("sys", [Message(role="user", text="hai")], [])
    assert calls.count == 4
```

  `fake_generate_raising` mengganti `llm._client.models.generate_content` dengan fungsi yang memunculkan `genai.errors.ClientError(429, {...})`.
- [ ] **Step 2:** Jalankan `.venv/Scripts/python.exe -m pytest tests/test_llm.py -v`. Hasil yang diharapkan: FAIL.
- [ ] **Step 3:** Implementasikan.
  - `GenerateContentConfig(system_instruction=..., tools=..., automatic_function_calling=AutomaticFunctionCallingConfig(disable=True), temperature=0.2)`.
  - Respons dibaca dari `response.candidates[0].content`: part `function_call` menjadi `ToolCall` (dengan `id` dari `function_call.id` atau `f"call_{i}"`), dan part teks digabung.
  - Retry hanya untuk `ClientError` 429 dan `ServerError`, dengan backoff `2 ** attempt` detik.
  - Error lain langsung diteruskan sebagai `LLMUnavailableError` dengan pesan asli.
- [ ] **Step 4:** Jalankan test yang sama. Hasil yang diharapkan: PASS.
- [ ] **Step 5:** Commit `feat: Gemini adapter with retry and scripted test LLM`.

### Task 5: Agent loop, routing, dan guardrail

**Files:**
- Create: `src/hr_assistant/prompts.py`, `src/hr_assistant/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `Retriever`, `RetrievedChunk` (Task 2), `server` (Task 3, hanya di test lewat `Client(server)`), dan `LLM`, `Message`, `ToolSpec`, `ToolCall`, `LLMReply` (Task 4).
- Produces:
  - `UNKNOWN_ANSWER` (lihat Global Constraints).
  - `SEARCH_TOOL = ToolSpec("search_hr_documents", ..., {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})`.
  - `Step(kind: Literal["llm", "tool", "guardrail"], name: str, detail: dict)`.
  - `AgentResult(answer: str, route: Literal["rag", "tool", "mixed", "none"], stop_reason: Literal["final", "max_steps", "guardrail"], sources: list[str], steps: list[Step])` dengan `to_dict()`.
  - `HRAgent(llm: LLM, mcp: Client, retriever: Retriever, max_steps: int = 6, max_answer_retries: int = 1)` dengan `async ask(question: str, employee_id: str | None = None) -> AgentResult`.
- Aturan loop:
  - Satu langkah adalah satu panggilan LLM.
  - Tool call di dalam balasan dieksekusi berurutan: `search_hr_documents` ke `retriever.relevant`, dan nama lain ke `mcp.call_tool`.
  - Hasil `search_hr_documents` berbentuk `{"found": bool, "chunks": [{"source_id": f"{source}#{section}", "text", "score"}]}`.
  - Hasil MCP yang `is_error` dikirim balik ke LLM sebagai `{"error": text}`.
  - Loop berhenti ketika balasan tidak berisi tool call, atau setelah `max_steps`. Kalau berhenti karena `max_steps`, `answer = UNKNOWN_ANSWER` dengan `stop_reason="max_steps"`.
- Aturan guardrail setelah jawaban final:
  - Grounded berarti ada hasil MCP (sukses maupun error, karena jawaban "ID tidak ditemukan" berasal dari tool), atau jawabannya memuat minimal satu sitasi `[sumber: <source_id>]` yang `source_id`-nya ada di chunk hasil pencarian pada pertanyaan ini.
  - Sitasi yang tidak dikenal dianggap tidak grounded.
  - Kalau tidak grounded, tambahkan satu pesan user koreksi yang meminta mengutip sumber atau menjawab persis `UNKNOWN_ANSWER`, lalu ulangi loop maksimal `max_answer_retries` kali.
  - Kalau masih gagal, `answer = UNKNOWN_ANSWER` dengan `stop_reason="guardrail"`.
  - Jawaban yang memuat frasa "tidak tahu" dinormalisasi persis menjadi `UNKNOWN_ANSWER`.
  - `route` ditentukan dari jenis tool yang sukses.
- Isi `SYSTEM_PROMPT` di `prompts.py`:
  - Data pribadi (sisa cuti, bawahan) dicari lewat tool MCP.
  - Kebijakan atau prosedur dicari lewat `search_hr_documents`.
  - Jawab hanya dari hasil tool, dengan format sitasi `[sumber: ...]`.
  - Kalau tidak ada, jawab persis `UNKNOWN_ANSWER`.
  - Jangan menebak ID karyawan.
  - `ask` menyisipkan `ID karyawan yang bertanya: {employee_id}` ke pesan user kalau ID diberikan.

- [ ] **Step 1: Write the failing tests.** Fixture memakai `ScriptedLLM`, `retriever` dari Task 2, dan `Client(server)`.

```python
async def test_policy_question_uses_rag_with_citation(agent_factory):
    agent = agent_factory([
        call("search_hr_documents", query="cuti tahunan"),
        final("Cuti tahunan 12 hari kerja per tahun. [sumber: kebijakan_cuti.md#Kebijakan Cuti > Cuti Tahunan]"),
    ])
    res = await agent.ask("Berapa jatah cuti tahunan?")
    assert res.route == "rag" and res.stop_reason == "final"
    assert "12 hari" in res.answer and res.sources == ["kebijakan_cuti.md#Kebijakan Cuti > Cuti Tahunan"]

async def test_personal_question_goes_through_mcp(agent_factory):
    agent = agent_factory([call("get_leave_balance", employee_id="E001"), final("Sisa cuti kamu 7 hari.")])
    res = await agent.ask("Sisa cuti saya berapa?", employee_id="E001")
    assert res.route == "tool" and "7 hari" in res.answer
    assert [s.name for s in res.steps if s.kind == "tool"] == ["get_leave_balance"]

async def test_uncited_answer_retried_then_unknown(agent_factory):
    agent = agent_factory([final("Gaji direktur 100 juta."), final("Gaji direktur 100 juta.")])
    res = await agent.ask("Berapa gaji direktur?")
    assert res.answer == UNKNOWN_ANSWER and res.stop_reason == "guardrail"

async def test_fabricated_citation_rejected(agent_factory):
    agent = agent_factory([call("search_hr_documents", query="bonus"), final("Bonus 5x gaji. [sumber: bonus.md#Bonus]"), final(UNKNOWN_ANSWER)])
    assert (await agent.ask("Berapa bonus tahunan?")).answer == UNKNOWN_ANSWER

async def test_unknown_employee_error_reaches_llm(agent_factory):
    llm_replies = [call("get_leave_balance", employee_id="E999"), final("ID E999 tidak ditemukan di sistem HR.")]
    agent = agent_factory(llm_replies)
    res = await agent.ask("Sisa cuti E999?")
    tool_msg = agent.llm.calls[1]["messages"][-1]
    assert "E999" in tool_msg.tool_result["error"]
    assert res.route == "tool" and "tidak ditemukan" in res.answer

async def test_max_steps_stops_loop(agent_factory):
    agent = agent_factory([call("search_hr_documents", query="x")] * 10, max_steps=3)
    res = await agent.ask("loop")
    assert res.stop_reason == "max_steps" and len([s for s in res.steps if s.kind == "llm"]) == 3
```

  Test ID tidak dikenal memastikan pesan error MCP sampai ke LLM, lalu jawaban "tidak ditemukan" lolos guardrail karena berasal dari hasil tool.
- [ ] **Step 2:** Jalankan `.venv/Scripts/python.exe -m pytest tests/test_agent.py -v`. Hasil yang diharapkan: FAIL.
- [ ] **Step 3:** Implementasikan `prompts.py` dan `agent.py` sesuai aturan di atas.
  - Daftar tool untuk LLM adalah `[SEARCH_TOOL] + ToolSpec` hasil konversi `mcp.list_tools()`, dengan `input_schema` menjadi `parameters`, diambil sekali per `ask`.
  - Panggilan LLM yang sinkron dijalankan dengan `await asyncio.to_thread(...)`.
- [ ] **Step 4:** Jalankan test yang sama. Hasil yang diharapkan: PASS.
- [ ] **Step 5:** Commit `feat: bounded agent loop with MCP tools, RAG tool, grounding guardrail`.

### Task 6: CLI dan eval harness

**Files:**
- Create: `src/hr_assistant/cli.py`, `src/hr_assistant/__main__.py`, `src/hr_assistant/evals.py`, `evals/cases.json`
- Test: `tests/test_cli.py`, `tests/test_evals.py`

**Interfaces:**
- Consumes: semua interface Task 1 sampai Task 5.
- Produces:
  - `main(argv: list[str] | None = None) -> int` dengan subcommand:
    - `ingest`
    - `ask QUESTION [--employee-id ID] [--trace]`
    - `chat [--employee-id ID]`
    - `eval [--cases PATH] [--out PATH]`
  - CLI menghubungkan agent ke MCP lewat `Client(StdioServerParameters(command=sys.executable, args=["-m", "hr_assistant.mcp_server"]))`.
  - `EvalCase(id: str, question: str, employee_id: str | None, expected_route: Literal["rag", "tool", "unknown"], must_contain: list[str], expected_source: str | None)`.
  - `load_cases(path: Path) -> list[EvalCase]`.
  - `score_case(case: EvalCase, result: AgentResult) -> CaseScore(case_id: str, passed: bool, checks: dict[str, bool])`, dengan check `route`, `contains`, `source`, dan `refusal`.
  - `async run_eval(agent: HRAgent, cases: Sequence[EvalCase]) -> EvalReport(total: int, passed: int, check_rates: dict[str, float], failures: list[CaseScore])`.
- `evals/cases.json` berisi 12 kasus:
  - 6 kasus RAG, masing-masing dengan `expected_source`
  - 3 kasus tool MCP (E001 sisa cuti, E001 bawahan, E999)
  - 3 kasus di luar dokumen dengan `expected_route: "unknown"`
- Arti `refusal` per rute:
  - Untuk kasus `unknown`, `refusal` berarti `answer == UNKNOWN_ANSWER`.
  - Kasus E999 memakai `expected_route: "tool"` dengan `must_contain: ["E999"]`.
  - Untuk kasus lain, `refusal` berarti jawabannya bukan `UNKNOWN_ANSWER`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_ask_without_index_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HR_INDEX_DIR", str(tmp_path / "missing"))
    assert main(["ask", "Berapa cuti?"]) == 2
    assert "Index belum dibuat. Jalankan: python -m hr_assistant ingest" in capsys.readouterr().err

def test_blank_question_rejected(capsys):
    assert main(["ask", "   "]) == 2
    assert "Pertanyaan tidak boleh kosong." in capsys.readouterr().err

def test_llm_unavailable_prints_one_line(monkeypatch, capsys, built_index):
    patch_agent_to_raise(monkeypatch, LLMUnavailableError("kuota habis"))
    assert main(["ask", "Berapa cuti?"]) == 1
    assert capsys.readouterr().err.strip().splitlines() == ["LLM tidak tersedia: kuota habis"]

def test_score_unknown_case():
    case = EvalCase("u1", "gaji direktur?", None, "unknown", [], None)
    assert score_case(case, result(answer=UNKNOWN_ANSWER, route="none")).passed

def test_score_rag_case_checks_source():
    case = EvalCase("r1", "cuti tahunan?", None, "rag", ["12 hari"], "kebijakan_cuti.md#Kebijakan Cuti > Cuti Tahunan")
    s = score_case(case, result(answer="12 hari [sumber: x]", route="rag", sources=["x"]))
    assert s.checks == {"route": True, "contains": True, "source": False, "refusal": True} and not s.passed

def test_cases_file_is_valid():
    cases = load_cases(Path("evals/cases.json"))
    assert len(cases) == 12 and {c.expected_route for c in cases} == {"rag", "tool", "unknown"}
```

- [ ] **Step 2:** Jalankan `.venv/Scripts/python.exe -m pytest tests/test_cli.py tests/test_evals.py -v`. Hasil yang diharapkan: FAIL.
- [ ] **Step 3:** Implementasikan CLI.
  - `ingest` mencetak `Index dibuat: {n} chunk dari {k} dokumen.`.
  - `ask` mencetak jawaban, lalu `Sumber: ...` kalau ada. `--trace` mencetak setiap `Step` sebagai JSON satu baris.
  - `eval` mencetak tabel per kasus, lalu `Lulus: {passed}/{total}`, dan menulis laporan JSON ke `--out` (default `logs/eval_report.json`).
  - `GEMINI_API_KEY` yang kosong menghasilkan pesan `GEMINI_API_KEY belum diisi. Salin .env.example ke .env lalu isi key.` dengan exit code 2.
  - Lalu implementasikan `evals.py` dan `evals/cases.json`.
- [ ] **Step 4:** Jalankan semua test dengan `.venv/Scripts/python.exe -m pytest -v`. Hasil yang diharapkan: semua PASS.
- [ ] **Step 5:** Commit `feat: CLI and eval harness`.

### Task 7: Dokumentasi dan verifikasi end-to-end

**Files:**
- Create: `README.md`, `docs/architecture.md`

- [ ] **Step 1:** Tulis `docs/architecture.md`:
  - diagram Mermaid end-to-end (CLI, agent loop, Gemini, retriever, Chroma, fastembed, MCP client stdio, MCP server, data karyawan, eval harness)
  - diagram urutan keputusan RAG vs MCP
  - 2 sampai 3 kalimat justifikasi untuk tiap komponen: Gemini, fastembed multilingual, Chroma, MCP stdio, guardrail deterministik, loop terbatas
- [ ] **Step 2:** Tulis `README.md`:
  - cara setup (`uv venv`, install, `.env`)
  - perintah demo
  - ringkasan arsitektur dengan tautan ke `docs/architecture.md`
  - trade-off: chunk 700 karakter karena jendela input embedding, Chroma vs pgvector/FAISS, embedding lokal vs API, satu kali revisi sebelum "tidak tahu"
  - use case harness dan loop engineering: loop terbatas `max_steps`, verifikasi sitasi dengan revisi, trace per langkah, dan eval harness
  - rencana eval production: metrik, LLM judge untuk faithfulness, sampling log, dan regression gate
  - keterbatasan
- [ ] **Step 3:** Jalankan `.venv/Scripts/python.exe -m hr_assistant ingest`. Hasil yang diharapkan: `Index dibuat: N chunk dari 5 dokumen.` dengan N > 0, dan model fastembed terunduh pada run pertama.
- [ ] **Step 4:** Jalankan `.venv/Scripts/python.exe -m pytest -v`. Hasil yang diharapkan: semua PASS.
- [ ] **Step 5:** Kalau `GEMINI_API_KEY` tersedia, jalankan tiga demo `ask`, yaitu kebijakan, `--employee-id E001` sisa cuti, dan di luar dokumen, lalu `eval`. Catat hasilnya di README. Kalau key tidak tersedia, tulis di README bahwa demo live belum dijalankan.
- [ ] **Step 6:** Commit `docs: README, architecture, trade-offs, eval plan`.
