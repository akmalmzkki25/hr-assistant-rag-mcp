# HR Assistant: RAG + MCP Agent

Chatbot HR internal berbasis CLI. Chatbot ini menjawab pertanyaan kebijakan dari dokumen HR (RAG), mengambil data cuti dan bawahan karyawan lewat MCP server, dan menjawab "tidak tahu" kalau jawabannya tidak ada di dokumen.

Dokumen HR di `data/docs/` dan data karyawan di migrasi seed adalah contoh fiktif untuk demo.

## Isi repo

| Kebutuhan | Lokasi |
|---|---|
| 1. Diagram arsitektur dan justifikasi komponen | [docs/architecture.md](docs/architecture.md) |
| 2. Pipeline RAG: chunking, embedding, vector store, retrieval top-k | `src/hr_assistant/rag/` (`loader.py`, `chunking.py`, `embeddings.py`, `store.py`, `retriever.py`, `ingest.py`) |
| 3. MCP server dengan `get_leave_balance` dan `get_subordinat_list` | `src/hr_assistant/mcp_server.py`, data di `src/hr_assistant/db/` |
| 4. Agent dengan tool calling lewat MCP, guardrail, harness dan loop | `src/hr_assistant/agent/` (`graph.py`, `guardrail.py`, `tools.py`, `prompts.py`), `evals.py` |
| 5. Dokumentasi | README ini |
| Test otomatis | `tests/` (109 test: 70 unit tanpa servis, 39 integrasi dengan Postgres, Typesense, dan MCP stdio) |

## Stack

| Komponen | Pilihan |
|---|---|
| LLM | Gemini `gemini-3.1-flash-lite` lewat `langchain-google-genai` |
| Harness agent | LangGraph `StateGraph` + `ToolNode`, memori chat dengan `InMemorySaver` |
| Embedding | `gemini-embedding-2`, 768 dimensi |
| Vector store | Typesense 0.25.2, hybrid search (kata kunci + vektor) |
| Data karyawan dan log run | Postgres, migrasi dengan Alembic |
| MCP | FastMCP (`mcp` 1.30) lewat stdio, client `langchain-mcp-adapters` |

Alasan setiap pilihan ada di [docs/architecture.md](docs/architecture.md#alasan-pilihan-komponen).

## Setup

Butuh Python 3.12, [uv](https://docs.astral.sh/uv/), dan Docker.

Jalankan Typesense dan Postgres:

```bash
docker run -d --name typesense --restart unless-stopped -p 8108:8108 -v typesense_data:/data typesense/typesense:0.25.2 --data-dir /data --api-key=ganti-password-typesense --listen-port 8108 --enable-cors
```

```bash
docker run -d --name postgres --restart unless-stopped -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=ganti-password-postgres -p 5432:5432 postgres:16
```

Pasang dependensi:

```bash
uv venv .venv --python 3.12
```

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

Salin `.env.example` menjadi `.env`, lalu isi:

- `GEMINI_API_KEY`: key gratis dari [Google AI Studio](https://aistudio.google.com/apikey).
- `TYPESENSE_API_KEY`: sama dengan `--api-key` di perintah Docker.
- `DATABASE_URL`: misalnya `postgresql://postgres:ganti-password-postgres@localhost:5432/hr_assistant`.

Model bisa diganti lewat `GEMINI_MODEL` (default `gemini-3.1-flash-lite`) dan `EMBEDDING_MODEL` (default `gemini-embedding-2`). Di macOS atau Linux, ganti `.venv/Scripts/python.exe` dengan `.venv/bin/python` di semua perintah.

## Menjalankan

Buat database `hr_assistant` dan jalankan migrasi Alembic (skema dan data dummy). Aman dijalankan berulang.

```bash
.venv/Scripts/python.exe -m hr_assistant db-init
```

```text
Database siap, migrasi sampai revision 0002_seed_dummy_employees.
```

Buat index dokumen di Typesense:

```bash
.venv/Scripts/python.exe -m hr_assistant ingest
```

```text
Index dibuat: 23 chunk dari 6 dokumen.
```

Pertanyaan kebijakan (lewat RAG):

```bash
.venv/Scripts/python.exe -m hr_assistant ask "Berapa jatah cuti tahunan karyawan?"
```

Pertanyaan dari dokumen PDF:

```bash
.venv/Scripts/python.exe -m hr_assistant ask "Berapa uang harian perjalanan dinas untuk manajer?"
```

Pertanyaan data pribadi (lewat MCP). ID karyawan dummy: `E001` (manager dari `E002` dan `E003`), `E002`, `E003`. Penanya hanya bisa melihat datanya sendiri dan data bawahan langsungnya, jadi `E003` yang menanyakan sisa cuti `E002` mendapat "Akses ditolak". Tanpa `--employee-id`, tool data karyawan tidak ditawarkan ke model dan agent meminta ID karyawan.

```bash
.venv/Scripts/python.exe -m hr_assistant ask "Siapa saja bawahan saya?" --employee-id E001
```

Pertanyaan di luar dokumen, yang harus dijawab "tidak tahu":

```bash
.venv/Scripts/python.exe -m hr_assistant ask "Berapa gaji CEO Majoo?"
```

Tambahkan `--trace` untuk melihat setiap langkah agent (panggilan LLM dan tool) sebagai JSON.

Mode interaktif dengan memori percakapan. Pertanyaan lanjutan seperti "kalau sisanya tidak terpakai?" memakai konteks giliran sebelumnya. Ketik `keluar`, atau kosongkan baris lalu Enter, untuk selesai.

```bash
.venv/Scripts/python.exe -m hr_assistant chat --employee-id E001
```

Setiap run `ask` dan `chat` dicatat di tabel `agent_runs`. Kalau Postgres mati, jawaban tetap dicetak dan CLI hanya memberi peringatan bahwa log tidak tersimpan.

Menjalankan eval harness (16 kasus, jeda 4 detik antar kasus untuk kuota gratis):

```bash
.venv/Scripts/python.exe -m hr_assistant eval
```

MCP server juga bisa dijalankan sendiri, misalnya untuk diuji dengan MCP Inspector:

```bash
.venv/Scripts/python.exe -m hr_assistant.mcp_server
```

Menjalankan test. Test integrasi dilewati otomatis kalau Typesense atau Postgres tidak bisa dihubungi. Test memakai database `hr_assistant_test` dan collection `hr_docs_test`, jadi data demo tidak tersentuh.

```bash
.venv/Scripts/python.exe -m pytest -q
```

### Kalau ada error

CLI mencetak satu baris yang menyebut servis yang bermasalah, bukan traceback:

| Pesan | Penyebab dan solusi |
|---|---|
| `Typesense tidak bisa dihubungi ...` | Container mati, sebelum atau di tengah pertanyaan. Jalankan `docker start typesense`. |
| `Postgres tidak bisa dihubungi: ...` | Saat `db-init`: container mati atau `DATABASE_URL` salah. Jalankan `docker start postgres`. |
| Jawaban "Database karyawan tidak bisa dihubungi" | Postgres mati saat MCP server mengambil data. Error ini lewat protokol MCP, jadi agent menyampaikannya sebagai jawaban. |
| `DATABASE_URL harus menyebut nama database ...` | Tambahkan nama database di akhir URL, misalnya `/hr_assistant`. |
| `Gagal membaca <berkas> ...` | Berkas di `data/docs/` rusak atau bukan UTF-8. Perbaiki atau keluarkan berkas itu. |
| `Index belum dibuat ...` | Jalankan `ingest` dulu. |
| `LLM tidak tersedia: ... 404 NOT_FOUND` | Model belum tersedia untuk key kamu. Isi `GEMINI_MODEL` dengan model yang ada di Google AI Studio, misalnya `gemini-2.5-flash`. |
| `... 429 RESOURCE_EXHAUSTED` | Kuota gratis per menit habis. Tunggu sebentar lalu ulangi. Embedding dan LLM sudah mencoba ulang 3 kali dengan backoff. Kalau kuota habis di tengah `ingest`, index lama tidak tersentuh, karena semua chunk di-embed dulu sebelum index dibuat ulang. |

## Arsitektur singkat

```mermaid
flowchart LR
    CLI --> Graph[LangGraph<br/>agent, tools, guardrail, finish]
    Graph <--> Gemini
    Graph -->|search_hr_documents| Retriever --> Typesense[(Typesense)]
    Retriever --> Embed[gemini-embedding-2]
    Graph -->|MCP stdio| MCPServer[FastMCP server] --> Postgres[(Postgres)]
```

Agent mengirim pertanyaan ke Gemini bersama dua jenis tool: `search_hr_documents` (RAG) dan tool yang ditemukan dari MCP server lewat `list_tools`. Gemini memilih tool dan node `tools` mengeksekusinya. Tool MCP dipanggil lewat `call_tool` ke proses MCP server terpisah, bukan dengan memanggil fungsi Python langsung. Penjelasan lengkap, diagram urutan, dan alasan tiap komponen ada di [docs/architecture.md](docs/architecture.md).

## Trade-off

**Chunk per heading, maksimal 1200 karakter dengan overlap 150.** Dokumen HR sudah terstruktur per bagian, jadi satu bagian biasanya berisi satu aturan utuh. `MarkdownHeaderTextSplitter` memotong di heading, lalu `RecursiveCharacterTextSplitter` memotong bagian yang terlalu panjang. Path heading (misalnya `Kebijakan Cuti > Cuti Tahunan`) ikut di-embed supaya konteksnya tidak hilang. Batas 1200 karakter jauh di bawah jendela 8192 token `gemini-embedding-2`, tetapi chunk yang lebih besar mencampur beberapa aturan dalam satu vektor dan membuat sitasi kurang tepat. Dokumen tanpa heading, seperti PDF contoh, memakai nama berkas sebagai bagian.

**Typesense, bukan pgvector.** Postgres sudah ada di stack, jadi pgvector juga masuk akal. Typesense dipilih karena hybrid search kata kunci dan vektor tersedia dalam satu query tanpa menulis SQL ranking sendiri, dan typo tolerance-nya membantu istilah yang salah ketik. Harganya satu servis tambahan untuk dijalankan.

**Embedding lewat API, bukan model lokal.** Versi pertama memakai model lokal sekitar 1 GB. `gemini-embedding-2` tidak perlu unduhan, mendukung input panjang, dan menaruh bagian yang benar di urutan pertama lebih sering. Harganya: ingest memakai kuota API, isi dokumen dikirim ke Google, dan jarak skor antara pertanyaan di dalam dan di luar dokumen lebih sempit (lihat ambang di bawah), sehingga penolakan lebih bergantung pada guardrail. Untuk dokumen HR sungguhan, perlu dicek dulu apakah itu diizinkan.

**Rerank dengan kemiripan vektor.** Fusion bawaan Typesense memberi peringkat tinggi ke chunk yang hanya cocok di kata umum seperti "berapa". Retriever mengambil 5 kali `top_k` kandidat, lalu mengurutkan dengan kemiripan vektor. Hasilnya, bagian Cuti Tahunan sekarang di urutan pertama untuk "Berapa jatah cuti tahunan karyawan?" (skor 0,806). Di versi pertama bagian itu di urutan ke-4.

**Ambang kemiripan 0,70.** Ambang ini dipilih dari uji kalibrasi dengan `gemini-embedding-2`. Pertanyaan yang jawabannya ada di dokumen mendapat skor 0,72 sampai 0,86, sedangkan pertanyaan di luar dokumen 0,54 sampai 0,68. Satu pengecualian: "bonus tahunan" mendapat 0,73 karena mirip bagian cuti tahunan. Kasus itu ditolak oleh prompt dan guardrail, bukan oleh ambang, dan ada di eval (`unknown-bonus-tahunan`).

**Satu kali revisi sebelum "tidak tahu".** Kalau jawaban pertama tidak punya sitasi valid, agent diberi satu pesan koreksi. Ini memulihkan kasus model lupa menulis sitasi, dengan biaya satu panggilan LLM tambahan.

**Guardrail berbasis sitasi dan angka.** Jawaban dari dokumen harus menunjuk chunk yang benar-benar diambil pada giliran itu. Setiap angka di jawaban, dengan atau tanpa sitasi, harus muncul sebagai angka utuh di chunk atau hasil MCP giliran itu, sehingga "carry over maksimal 6 hari" atau "7 hari" padahal datanya 17 ditolak. Jawaban dari tool MCP tanpa sitasi juga harus menyebut minimal satu nilai dari hasil tool (angka, nama, ID, atau pesan error), supaya kalimat bebas angka seperti "sisa cuti kamu masih banyak" tidak lolos. Guardrail ini belum memeriksa apakah isi kalimat benar-benar didukung chunk, dan angka yang ditulis dengan huruf ("dua kali") tidak ikut diperiksa. Pemeriksaan isi butuh LLM judge atau model NLI.

**`mcp` 1.30, bukan 2.x.** `langchain-mcp-adapters` 0.3.2 belum mendukung `mcp` 2, jadi server memakai `FastMCP` dari `mcp` 1.30. Nama tool dan protokolnya sama.

**Memori chat di proses, bukan di database.** `InMemorySaver` cukup untuk satu sesi CLI. Untuk banyak pengguna, checkpointer bisa diganti `PostgresSaver` dari `langgraph-checkpoint-postgres` tanpa mengubah graph.

## Harness dan loop engineering

Harness di sini adalah kerangka yang menjalankan, membatasi, dan memeriksa loop agent. LangGraph menjadi harness-nya, dengan empat bagian:

1. **Loop agent terbatas.** Graph berputar antara node `agent` dan `tools` sampai model memberi jawaban final atau batas `max_steps` (6 panggilan LLM) tercapai. Kalau batas tercapai, node `finish` menjawab "tidak tahu" dengan `stop_reason=max_steps`, bukan menggantung. `recursion_limit` LangGraph dipasang di atas batas itu sebagai pengaman kedua.
2. **Verifikasi lalu revisi.** Jawaban final masuk node `guardrail`. Kalau gagal, edge kembali ke `agent` dengan pesan koreksi, lalu diperiksa lagi. Revisi dibatasi satu kali.
3. **Error tool dipilah.** Error yang memang jawaban, seperti ID `E999` tidak ditemukan atau akses ditolak, dikembalikan ke model sebagai `ToolMessage` berstatus error, lalu model menyampaikannya. Error infrastruktur, seperti kuota embedding habis atau Typesense mati, tidak dikembalikan ke model. Error itu naik ke CLI dan dicetak sebagai satu baris, supaya tidak berubah menjadi jawaban "tidak tahu" yang menyesatkan dan tidak terhitung sebagai penolakan yang benar di eval.
4. **Eval harness.** `python -m hr_assistant eval` menjalankan 16 kasus di `evals/cases.json` lewat graph yang sama, lalu menilai empat cek per kasus:
   - `route`: tool yang dipakai sesuai harapan (`rag`, `tool`, `mixed`, atau `none`)
   - `contains`: jawaban memuat fakta kunci
   - `source`: sitasi menunjuk bagian dokumen yang benar
   - `refusal`: "tidak tahu" hanya muncul di kasus yang memang di luar dokumen

   Laporan JSON ditulis ke `logs/eval_report.json`. Kegagalan per cek menunjukkan bagian mana yang perlu diperbaiki: retrieval, pemilihan tool, atau guardrail.

## Evaluasi untuk production

Eval harness di repo ini adalah titik awal. Untuk production, eval berikut perlu ditambahkan:

- **Golden set dari pertanyaan sungguhan.** Ambil 100 sampai 200 pertanyaan dari tabel `agent_runs` dan tiket HR, lalu minta tim HR menulis jawaban acuan dan bagian dokumen sumbernya. Perbarui setiap kali kebijakan berubah.
- **Metrik retrieval.** Recall@k dan MRR terhadap bagian sumber yang benar, untuk mendeteksi masalah chunking atau embedding secara terpisah dari LLM.
- **Faithfulness dengan LLM judge.** Model penilai memeriksa apakah setiap klaim di jawaban didukung chunk yang dikutip. Penilai dikalibrasi dulu terhadap penilaian manusia pada sampel kecil.
- **Akurasi penolakan.** Dua angka terpisah: seberapa sering agent menolak pertanyaan yang bisa dijawab, dan seberapa sering agent menjawab pertanyaan yang seharusnya ditolak. Angka kedua yang paling berbahaya untuk chatbot HR.
- **Akurasi pemilihan tool.** Persentase pertanyaan data pribadi yang benar-benar memakai tool MCP dengan argumen ID yang benar.
- **Regression gate di CI.** Eval dijalankan pada setiap perubahan prompt, model, chunking, atau ambang. Perubahan ditolak kalau faithfulness atau akurasi penolakan turun melewati batas yang disepakati.
- **Monitoring online.** Latensi dan rute per run dari `agent_runs`, tombol jempol atas/bawah, sampling percakapan untuk ditinjau manusia setiap minggu, dan rasio jawaban "tidak tahu" sebagai sinyal dokumen yang kurang lengkap.

## Status verifikasi

Diuji pada 6 November 2026 dengan `gemini-3.1-flash-lite`, `gemini-embedding-2`, Typesense 0.25.2, dan Postgres 16.

- 109 test otomatis lulus (`pytest -q`, sekitar 50 detik). Test unit memakai embedder palsu dan LLM skrip. Test integrasi memakai Postgres dan Typesense sungguhan, serta MCP server lewat stdio.
- `db-init` dijalankan dua kali tanpa menggandakan data. `ingest`: 23 chunk dari 6 dokumen, termasuk satu PDF.
- Eval harness dengan Gemini asli: **16/16 lulus**, dengan cek rute, isi, sumber, dan penolakan masing-masing 100%.
- Contoh jawaban dari uji live:
  - "Berapa jatah cuti tahunan karyawan?" dijawab "12 hari kerja per tahun" dengan sitasi `kebijakan_cuti.md#Kebijakan Cuti > Cuti Tahunan`.
  - "Berapa uang harian perjalanan dinas untuk manajer?" dijawab "Rp500.000 per hari" dengan sitasi ke PDF `pedoman_perjalanan_dinas.pdf`.
  - "Siapa saja bawahan saya?" dengan `--employee-id E001` dijawab Budi Santoso (E002) dan Sari Lestari (E003) lewat `get_subordinat_list`.
  - "Berapa gaji CEO Majoo?" dijawab "Maaf, saya tidak tahu. Informasi itu tidak ada di dokumen HR yang saya punya."
  - "Berapa sisa cuti karyawan E002?" dengan `--employee-id E003` dijawab "Akses ditolak: data karyawan E002 hanya bisa dilihat oleh E002 sendiri atau atasan langsungnya." Dengan `--employee-id E001` (atasan Budi), jawabannya 2 hari.
  - "Sisa cuti saya berapa?" tanpa `--employee-id` dijawab dengan permintaan ID karyawan (4 dari 4 percobaan).
  - Di `chat`, "Berapa sisa cuti tahunan saya?" dijawab 7 hari, lalu pertanyaan lanjutan "Kalau sisanya tidak terpakai, boleh dibawa ke tahun depan berapa hari?" dijawab maksimal 5 hari dengan sitasi bagian Carry Over Cuti.

## Keterbatasan

- Korpus contoh hanya 6 dokumen, jauh lebih kecil dari dokumen HR sungguhan yang bisa mencapai ratusan halaman. Pipeline-nya sama untuk korpus besar, tetapi ambang 0,70 dan `top_k=4` perlu dikalibrasi ulang dengan dokumen sungguhan.
- Data karyawan adalah dummy tanpa autentikasi. Aturan akses (data sendiri dan bawahan langsung) sudah dijalankan di node `tools`, tetapi ID penanya masih diambil dari argumen `--employee-id` yang bisa diisi siapa saja. Di production, ID harus berasal dari sesi login, dan pemeriksaan akses sebaiknya juga ada di MCP server, supaya client lain tidak bisa melewatinya.
- Guardrail memeriksa sitasi dan angka, belum memeriksa kebenaran isi kalimat (lihat Trade-off).
- Jawaban yang sebagian benar tetapi juga memuat "tidak tahu" (tanpa data MCP) diganti seluruhnya dengan kalimat "tidak tahu" baku. Ini aman, tetapi bisa membuang bagian jawaban yang sebenarnya benar.
- Index dibangun ulang penuh setiap `ingest`. Untuk dokumen yang sering berubah, perlu ingest inkremental berdasarkan hash berkas.
- Memori chat hilang saat proses CLI ditutup (lihat Trade-off).
