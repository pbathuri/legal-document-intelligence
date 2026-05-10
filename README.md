# Legal / Property Document Intelligence Agent

Agentic diligence on **AMD MI300X-class** setups: **LangGraph**, **Qdrant** RAG, **sentence-transformers** embeddings, and an **OpenAI-compatible** LLM (**vLLM** + Llama-class models).

Two modes:

| Domain (`DILIGENCE_DOMAIN`) | Use case |
|-----------------------------|----------|
| `india_re` (default) | **India property diligence**: instruments, **TitleGraph** (heuristic chain + breaks), encumbrance/dispute signals, records-context memo with **presumptive-titling** disclaimer. |
| `mna` | **M&A contracts**: obligations, risks, cross-document, compliance memo. |

## Quick start (mock mode, in-memory)

```bash
cd AMD_Hackathon
pip install -e ".[dev]"
LEGAL_INTEL_MOCK_LLM=1 QDRANT_URL=:memory: pytest -q
LEGAL_INTEL_MOCK_LLM=1 QDRANT_URL=:memory: streamlit run streamlit_app.py
```

### Scrape + build training data

```bash
python -m legal_intel.scraper.cli igrs --max-results 20 --training --output data/raw/igrs.json
python -m legal_intel.training.prepare --data-dir data/raw --output-dir data/training
```

Equivalent entry points: `legal-scrape …`, `legal-train-prep …`. Optional extras: `pip install -e ".[dev,scraping]"` if you use **scrapling**-backed paths; demo JSON under `data/raw/` works offline.

### On MI300X: fine-tune and benchmark

```bash
python scripts/finetune_rocm.py --train-file data/training/train.jsonl --val-file data/training/val.jsonl
python scripts/benchmark_inference.py --base-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-70B-Instruct
```

See `Makefile` targets: `make test`, `make ui`, `make train` (runs scrapers then prep).

## Quick start (CPU / demo)

```bash
cd /path/to/AMD_Hackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# optional: docker compose up -d
export LEGAL_INTEL_MOCK_LLM=1
streamlit run streamlit_app.py
```

In the UI, choose **India property** or **M&A** in the sidebar. For India, read the **red disclaimer**: registration and digitized records are **evidence**, not a guaranteed title; outputs are **not legal advice**.

## HTTP API (FastAPI) + hybrid frontend (Vercel)

The interactive dashboard at repo root (`index.html`) is designed for **static hosting** (e.g. Vercel). LLMs, embeddings, and Qdrant run on a **separate Python host** — not inside Vercel serverless.

1. Install and run the API:

```bash
pip install -e ".[dev]"
export LEGAL_INTEL_MOCK_LLM=1          # or 0 + configure LLM below
export QDRANT_URL=:memory:             # or http://localhost:6333 with docker compose up -d
# CPU / CI-friendly embeddings (matches tests/conftest.py). Omit these only if you use Ollama embeddings.
export EMBEDDING_PROVIDER=sentence_transformers
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
legal-api
# → http://127.0.0.1:8080  (override with LEGAL_INTEL_API_PORT)
```

2. Endpoints (selection):
   - `GET /health` — mock/live LLM, embedding provider, Qdrant, resolved multimodel names; when `LLM_PROVIDER=ollama`, lists models from local **Ollama** `/api/tags`
   - `GET /health/live` — liveness only (process accepting requests; no dependency checks)
   - `GET /health/ready` — readiness from **`gather_preflight()`** (Qdrant, Ollama when used, embed probe)
   - `GET /v1/preflight` — single JSON for ops: Qdrant ping, Ollama `/api/tags` + optional `/api/embed` probe, disk, `device` profile; add **`?deep=1`** for embedded **`/api/version`** + **`/api/ps`**
   - `GET /v1/build` — package version, API version, Python, optional **`LEGAL_INTEL_GIT_SHA`**
   - `POST /v1/llm/probe` — one minimal chat completion via routed stack (skipped when mock LLM)
   - `GET /v1/metrics` — in-process HTTP request totals + path buckets (UUID run ids normalized); resets on restart
   - `GET /v1/metrics/prometheus` — Prometheus text exposition for scraping
   - `GET /v1/ollama/host` — native daemon introspection: **`/api/version`** + **`/api/ps`** (running models on the machine)
   - `GET /v1/ollama/agent-stack` — **single JSON** for local agents: Ollama **`/api/version`** + **`/api/tags`** + **`/api/ps`**, embed probe, per-task **`model_routing`**, and configuration **warnings** (partial failures are surfaced per-field, not 503)
   - `GET /v1/ollama/version` — lightweight **`GET /api/version`** from the Ollama daemon (single-purpose; also folded into preflight deep)
   - `GET /v1/ollama/ps` — native **`/api/ps`** JSON (models currently loaded in the daemon — complements **`GET /v1/ollama/host`**)
   - `GET /v1/ollama/tags` — native **`GET /api/tags`** — **full** daemon JSON (digests, sizes; complements name-only lists in **`/health`**)
   - `POST /v1/ollama/embed-proxy` — raw **`POST /api/embed`** passthrough (full daemon JSON; defaults **`model`** to **`OLLAMA_EMBEDDING_MODEL`**)
   - `GET /v1/embeddings/info` — resolved embedding provider/model env + vector **dimension** + probe encode timing
   - `POST /v1/embeddings/warmup` — forces embedding backend load (sentence-transformers or Ollama `/api/embed`)
   - `POST /v1/embeddings/embed-texts` — batch vectors for up to **48** strings (same embedding backend as RAG; large payloads — use for pipelines/agents on-device)
   - `POST /v1/embeddings/embed-local-text-files` — read UTF-8 files from absolute paths under **`LEGAL_INTEL_ALLOW_LOCAL_PATHS`** (≤16 paths, ≤256KiB each) → per-path vectors + errors (same embedding backend as RAG)
   - `POST /v1/embeddings/similarity` — JSON `{ "text_a", "text_b" }` → cosine similarity + embedding dimension (same backend as RAG)
   - `POST /v1/embeddings/pairwise-matrix` — JSON `{ "texts": ["...", ...] }` (2–24 strings) → full **N×N** cosine matrix + short text previews (same embedding backend as RAG)
   - `POST /v1/embeddings/centroid-similarities` — JSON `{ "texts": ["...", ...] }` (2–48 strings) → **mean embedding vector** + each row's cosine to that centroid (topic coherence / bundle QA)
   - `POST /v1/embeddings/nearest-to-query` — JSON `{ "query", "candidates": ["...", ...] }` (≤64 candidates) → **ranked** list by cosine similarity to the query embedding (same backend as RAG — useful for local rerank / shortlist)
   - `POST /v1/embeddings/farthest-pair` — JSON `{ "texts": ["...", ...] }` (3–40 strings) → indices + cosine for the pair with **minimum** similarity (**most divergent** excerpts — divergence QA / contrast mining)
   - `POST /v1/embeddings/document-centroid-similarity` — JSON `{ "doc_id_a", "doc_id_b", "max_chunks_per_document" }` → cosine between **mean embeddings** of chunk texts per document (bounded Qdrant scroll — topical similarity / clustering signal without LLM)
   - `POST /v1/embeddings/document-chunk-stats` — JSON `{ "doc_id", "max_chunks_scanned" }` → **chunk length statistics** from Qdrant (non-empty vs empty chunks, mean/max/min chars; **`truncated_scan`** if more chunks exist — OCR / segmentation QA; **no LLM**)
   - `POST /v1/embeddings/document-lexical-jaccard` — JSON `{ "doc_id_a", "doc_id_b", "max_chunks_per_document" }` → **token Jaccard** overlap on chunked text (stopwords trimmed — cheap lexical complement to semantic centroid similarity; **no LLM**)
   - `POST /v1/embeddings/document-token-difference` — JSON `{ "doc_id_a", "doc_id_b", "max_chunks_per_document", "max_tokens_per_side" }` → **sorted token vocabulary deltas** (tokens only in A vs only in B; truncation flags — **no LLM**)
   - `GET /v1/runtime` — Python version, cwd, resolved upload DB paths (device-local); **`device`** may include **`nvidia_gpus`** when `nvidia-smi` is available; responses include **`X-Request-ID`** (echo or generated) and **`X-Process-Time`**
   - `GET /v1/runtime/storage` — bounded filesystem inventory: bytes + file counts under **`UPLOAD_STORAGE_DIR`**, **`manifest.jsonl`** size/line estimate, **`RUNS_DB_PATH`** file size (device-local maintenance)
   - `GET /v1/runtime/git` — best-effort **Git** snapshot for the API **`cwd`** (`commit`, `branch`, `dirty` when **`git`** is on **PATH**)
   - `GET /v1/runtime/host-metrics` — extended **psutil** snapshot: CPU sample, RAM/swap %, disk usage of **`cwd`**, boot time, bounded disk partition list (optional dependency)
   - `GET /v1/runtime/network` — hostname/FQDN + per-interface IPv4/IPv6 addresses (**psutil**); aggregate **`net_io_counters`** when available
   - `GET /v1/runtime/local-path-allowlist` — inspect **`LEGAL_INTEL_ALLOW_LOCAL_PATHS`** prefixes (exists, resolved path, **`disk_usage`** free/total per prefix root)
   - `GET /v1/runtime/rlimits` — Unix **`resource.getrlimit`** snapshot (**NOFILE**, stack, AS, …); Windows returns a short **note** instead of limits
   - `GET /v1/runtime/sys-path` — bounded **`sys.path`** prefix (`limit` query param, default 64 entries) for interpreter / packaging debugging on the API host
   - `GET /v1/runtime/path-entries` — non-empty segments from the process **`PATH`** env var (`limit` query param, default 80; max 200) — toolchain / shell debugging on the API host
   - `GET /v1/runtime/platform-detail` — stdlib **`platform`** snapshot (**uname**, libc version when available, Python build tags) for diagnosing native wheels / ROCm / toolchain on the API host
   - `GET /v1/runtime/optional-imports` — best-effort **`__version__`** probes for **numpy**, **torch**, **sentence-transformers**, **LangChain/LangGraph**, **Qdrant**, **PyMuPDF (`fitz`)**, **sklearn**, **psutil**, **openai** on the active interpreter (offline — agents size workloads / debug imports)
   - `GET /v1/runtime/process-memory` — this API process **RSS/VMS**, thread count, optional **`resource.getrusage` maxrss**, **`psutil`** when installed (local sizing / leak sniffing)
   - `GET /v1/runtime/agent-bootstrap` — **single JSON** for on-device agents: resolved **Ollama / OpenAI-compatible** model routing (**extraction** / **synthesis** / **specialist**), **`gather_preflight()`**, **`platform_detail`**, **`device`**, and route hints (no secrets — aligns local agents with this API before RAG calls)
   - `GET /v1/agents` — LangGraph node lists + model routing map (all agents use your Ollama/vLLM routing)
   - `GET /v1/runs/stats` — SQLite file size, total rows, counts **by domain**, min/max `created_at` (uses configured `RUNS_DB_PATH` even if `PERSIST_RUNS=0`)
   - `GET /v1/runs` / `GET /v1/runs/search` / `GET /v1/runs/export/json` / `GET /v1/runs/export` / `GET /v1/runs/{id}/memo.md` / `GET /v1/runs/{id}` / `DELETE /v1/runs/{id}` — SQLite history; substring search; JSON array or NDJSON export; Markdown memo (`final_report`)
   - `GET /v1/disk` — free space on volume holding upload storage
   - `GET /v1/settings/effective` — full resolved config with **secrets redacted**
   - `GET /v1/qdrant/info` — collection existence + point count (uses same client as RAG, including in-process `:memory:`)
   - `GET /v1/documents` — distinct indexed `doc_id` values (bounded payload scan)
   - `GET /v1/documents/{doc_id}/chunks` — paginated chunk payloads for RAG debugging (`cursor` + `limit`)
   - `DELETE /v1/documents/{doc_id}` — remove all vectors for that document from Qdrant
   - `POST /v1/documents/purge` — JSON `{ "doc_ids": ["...", "..."] }` batch-delete vectors (≤200 ids)
   - `POST /v1/rag/near-duplicate-chunks` — intra-**doc_id** chunk pairs above a cosine threshold (bounded chunk count; uses same embedding backend as RAG — OCR overlap / duplicate page QA)
   - `POST /v1/rag/document-summary` — retrieval scoped to one **`doc_id`** + **synthesis**-task memo (grounded summary via **`SUMMARIZE_SYSTEM`**; **`retrieval_query`** drives chunk selection)
   - `POST /v1/rag/document-summary/stream` — **SSE** (`text/event-stream`): initial **`sources`** event + synthesis token stream + **`done`** (same retrieval path as non-stream)
   - `POST /v1/rag/compare-documents` — retrieval from **two** **`doc_id`** values + specialist comparison (**`COMPARE_DOCUMENTS_SYSTEM`**); optional **`limit_per_document`**
   - `POST /v1/rag/compare-documents/stream` — **SSE**: **`sources`** for both sides + specialist comparison tokens + **`done`**
   - `POST /v1/rag/cross-document-summary` — retrieval from **2–12** distinct **`doc_id`** values (shared **`retrieval_query`**) + one **synthesis** memo across labeled excerpts (**`CROSS_DOCUMENT_SUMMARIZE_SYSTEM`**); returns **`sources_by_doc_id`**
   - `POST /v1/rag/cross-document-summary/stream` — **SSE**: **`sources_by_doc_id`** event + synthesis token stream + **`done`**
   - `POST /v1/rag/cross-document-contradictions` — same multi-doc retrieval as cross-summary (defaults tuned for tension mining) + JSON **`tensions`** / **`aligned_points`** (**`CONTRADICTIONS_JSON_SYSTEM`** / **`contradictions_scan_v1`**)
   - `POST /v1/rag/cross-document-contradictions/stream` — **SSE**: **`sources_by_doc_id`** + streaming contradictions JSON (**extraction** routing)
   - `POST /v1/rag/structured-extract` — retrieval + **`json_object`** extraction for caller-defined **`categories`** keys + **`evidence_refs`** (**`STRUCTURED_EXTRACT_SYSTEM`**; extraction-task routing)
   - `POST /v1/rag/timeline-extract` — retrieval + JSON **timeline** (`events` with **`date_text`**, **`evidence_refs`** ↔ **`[n]`**; **`TIMELINE_JSON_SYSTEM`** / **`timeline_extract_v1`**)
   - `POST /v1/rag/timeline-extract/stream` — **SSE**: **`sources`** event + token stream of JSON timeline text (**extraction** routing)
   - `POST /v1/rag/risk-scan` — retrieval + JSON **risk register** (`risks[]` with **`severity`**, **`evidence_refs`**; **`RISK_SCAN_JSON_SYSTEM`** / **`risk_scan_v1`**)
   - `POST /v1/rag/risk-scan/stream` — **SSE**: **`sources`** + streaming JSON risk text (**extraction** routing)
   - `POST /v1/rag/glossary-extract` — retrieval + JSON **glossary** (`terms[]` + **`evidence_refs`**; **`GLOSSARY_JSON_SYSTEM`** / **`glossary_extract_v1`**)
   - `POST /v1/rag/glossary-extract/stream` — **SSE**: **`sources`** + streaming glossary JSON (**extraction** routing)
   - `POST /v1/rag/document-outline` — retrieval + JSON **outline** (`sections[]` + **`evidence_refs`**; **`DOCUMENT_OUTLINE_JSON_SYSTEM`** / **`document_outline_v1`**)
   - `POST /v1/rag/document-outline/stream` — **SSE**: **`sources`** + streaming outline JSON (**extraction** routing)
   - `POST /v1/rag/diligence-checklist` — retrieval + JSON **diligence checklist** (`items[]` with **`priority`** + **`evidence_refs`**; **`DILIGENCE_CHECKLIST_JSON_SYSTEM`** / **`diligence_checklist_v1`**)
   - `POST /v1/rag/diligence-checklist/stream` — **SSE**: **`sources`** + streaming checklist JSON (**extraction** routing)
   - `POST /v1/rag/issue-spotter` — retrieval + JSON **issues** register (**`ISSUE_SPOTTER_JSON_SYSTEM`** / **`issue_spotter_v1`**)
   - `POST /v1/rag/issue-spotter/stream` — **SSE**: **`sources`** + streaming issue JSON (**extraction** routing)
   - `POST /v1/rag/suggested-questions` — retrieval + JSON **follow-up diligence questions** (**`SUGGESTED_QUESTIONS_JSON_SYSTEM`** / **`suggested_questions_v1`**)
   - `POST /v1/rag/suggested-questions/stream` — **SSE**: **`sources`** + streaming suggested-questions JSON (**extraction** routing)
   - `POST /v1/rag/deal-thesis` — retrieval + JSON **bull/bear deal thesis** + evidence refs (**`DEAL_THESIS_JSON_SYSTEM`** / **`deal_thesis_v1`**)
   - `POST /v1/rag/deal-thesis/stream` — **SSE**: **`sources`** + streaming deal-thesis JSON (**extraction** routing)
   - `POST /v1/rag/bibliography-export` — retrieval + **markdown bibliography / excerpt digest** for counsel review (**`BIBLIOGRAPHY_EXPORT_SYSTEM`**; **`citation_style`** `neutral` \| `deal_memo` \| `compact`; **synthesis** routing)
   - `POST /v1/rag/bibliography-export/stream` — **SSE**: **`sources`** + bibliography markdown tokens (**synthesis** routing)
   - `POST /v1/rag/covenant-matrix` — retrieval + JSON **obligation / covenant matrix** (affirmative vs negative, topics, triggers; **`covenant_matrix_v1`**)
   - `POST /v1/rag/covenant-matrix/stream` — **SSE**: **`sources`** + streaming covenant-matrix JSON (**extraction** routing)
   - `POST /v1/rag/financial-terms-ledger` — retrieval + JSON **quantitative term ledger** (amounts, baskets, caps — excerpts only; **`financial_terms_ledger_v1`**)
   - `POST /v1/rag/financial-terms-ledger/stream` — **SSE**: **`sources`** + streaming ledger JSON (**extraction** routing)
   - `POST /v1/rag/remedies-playbook` — retrieval + JSON **remedies / forum / fee-shifting map** (**`remedies_playbook_v1`**)
   - `POST /v1/rag/remedies-playbook/stream` — **SSE**: **`sources`** + streaming remedies JSON (**extraction** routing)
   - `POST /v1/rag/conditions-precedent` — retrieval + JSON **conditions precedent / closing deliverables** (**`conditions_precedent_v1`**)
   - `POST /v1/rag/conditions-precedent/stream` — **SSE**: **`sources`** + streaming CP JSON (**extraction** routing)
   - `POST /v1/rag/execution-formalities` — retrieval + JSON **counterparts / e-sign / execution mechanics** (**`execution_formalities_v1`**)
   - `POST /v1/rag/execution-formalities/stream` — **SSE**: **`sources`** + streaming execution JSON (**extraction** routing)
   - `POST /v1/rag/retrieval-expand-plan` — retrieval + **`agent_goal`** + JSON **follow-up vector queries** for agents (**`retrieval_expand_plan_v1`**; **specialist** routing — uses **`LLM_MODEL_SPECIALIST`** / Ollama specialist slot)
   - `POST /v1/rag/retrieval-expand-plan/stream` — **SSE**: **`sources`** + streaming expand-plan JSON (**specialist** routing)
   - `POST /v1/rag/survival-schedule` — retrieval + JSON **survival-of-obligations schedule** (duration text per topic; **`survival_schedule_v1`**)
   - `POST /v1/rag/survival-schedule/stream` — **SSE**: **`sources`** + streaming survival JSON (**extraction** routing)
   - `POST /v1/rag/assignment-coc` — retrieval + JSON **assignment / change-of-control** map (**`assignment_coc_v1`**)
   - `POST /v1/rag/assignment-coc/stream` — **SSE**: **`sources`** + streaming assignment JSON (**extraction** routing)
   - `POST /v1/rag/ip-assets-sweep` — retrieval + JSON **IP / software / OSS** sweep (**`ip_assets_sweep_v1`**)
   - `POST /v1/rag/ip-assets-sweep/stream` — **SSE**: **`sources`** + streaming IP sweep JSON (**extraction** routing)
   - `POST /v1/rag/post-closing-covenants` — retrieval + JSON **post-closing / transition / TSA-style obligations** (**`post_closing_covenants_v1`**)
   - `POST /v1/rag/post-closing-covenants/stream` — **SSE**: **`sources`** + streaming JSON (**extraction** routing)
   - `POST /v1/rag/earn-out-mechanics` — retrieval + JSON **earn-out / milestone / contingent consideration mechanics** (**`earn_out_mechanics_v1`**)
   - `POST /v1/rag/earn-out-mechanics/stream` — **SSE**: **`sources`** + streaming JSON (**extraction** routing)
   - `POST /v1/rag/representations-buckets` — retrieval + JSON **representations & warranties thematic buckets + qualifiers** (**`reps_buckets_v1`**)
   - `POST /v1/rag/representations-buckets/stream` — **SSE**: **`sources`** + streaming JSON (**extraction** routing)
   - `POST /v1/rag/tax-withholding` — retrieval + JSON **tax / withholding / FIRPTA / gross-up** hooks (**`tax_withholding_v1`**)
   - `POST /v1/rag/tax-withholding/stream` — **SSE**: **`sources`** + streaming JSON (**extraction** routing)
   - `POST /v1/rag/insurance-requirements` — retrieval + JSON **insurance / R&W / D&O tail** signals (**`insurance_requirements_v1`**)
   - `POST /v1/rag/insurance-requirements/stream` — **SSE**: **`sources`** + streaming JSON (**extraction** routing)
   - `POST /v1/rag/sanctions-export-compliance` — retrieval + JSON **sanctions / export / anti-corruption** hooks (**`sanctions_export_compliance_v1`**)
   - `POST /v1/rag/sanctions-export-compliance/stream` — **SSE**: **`sources`** + streaming JSON (**extraction** routing)
   - `GET /v1/uploads/manifest` — tail of `manifest.jsonl` beside persisted uploads (requires `PERSIST_UPLOADS`)
   - `GET /v1/uploads/files` — bounded listing of files under **`UPLOAD_STORAGE_DIR`** (mtime-descending; requires **`PERSIST_UPLOADS`**)
   - `POST /v1/ingest` — multipart PDF → includes **page/char stats** and optional `persisted_path`
   - `POST /v1/ingest/local` — JSON body `{ "path": "/absolute/file.pdf", "use_ocr": false }` — only if `LEGAL_INTEL_ALLOW_LOCAL_PATHS` lists allowed absolute directory prefixes (comma-separated)
   - `POST /v1/ingest/batch` — many PDFs in one request → `{ items[], errors[] }`
   - `POST /v1/analyze` — full graph run; returns optional `run_id` when persistence enabled
   - `POST /v1/analyze/stream` — **SSE** (`text/event-stream`) LangGraph step updates + final merged state
   - `POST /v1/query` / `POST /v1/query/stream` — grounded Q&A (stream returns tokens + sources); optional JSON **`limit`** (1–128) overrides **`RETRIEVAL_TOP_K`** for that call
   - `POST /v1/query/hyde` — **HyDE**-style RAG: hypothetical excerpt (**specialist** routing) + retrieval using **question + excerpt** concatenated query → grounded answer (same **`QUERY_SYSTEM`** stack as **`/v1/query`**; **`hyde_temperature`** controls the fiction step)
   - `POST /v1/query/hyde/stream` — **SSE**: **`hypothetical_document`** event → **`sources`** → **`QUERY_SYSTEM`** token stream + **`done`**
   - `POST /v1/query/citations` — same retrieval as **`/v1/query`**, but the LLM returns **JSON** with **`direct_answer`**, **`citations`** (`ref_index` ↔ chunk **`[n]`**), **`limitations`** (`json_object` / Ollama-compatible); exposes **`structured`** + flattened **`answer_markdown`**
   - `POST /v1/query/batch` — multiple grounded questions in one request (shared **`doc_id`** / **`limit`**; sequential specialist calls — fewer round trips for agent pipelines)
   - `POST /v1/query/retrieve-only/batch` — batch **`retrieve-only`** (no LLM; contexts + sources per question)
   - `POST /v1/query/retrieve-only` — same retrieval + **`formatted_context`** as `/v1/query`, but **no LLM** (sources + context only); supports **`limit`** override
   - `POST /v1/ollama/generate` — forwards non-streaming requests to Ollama’s native **`POST /api/generate`** (model, prompt, optional `system`, `options`; uses origin from `OLLAMA_BASE_URL`)
   - `POST /v1/ollama/generate/batch` — sequential native **`/api/generate`** for up to **12** prompts (same **`model`** / **`system`** / **`options`**; per-item success/error — agent batching over local Ollama)
   - `POST /v1/ollama/chat` — Ollama **`POST /api/chat`** (multi-turn **`messages`**, non-streaming; optional **`options`** merge)
   - `POST /v1/ollama/show` — Ollama **`POST /api/show`** (inspect model template, parameters, etc.)
   - `POST /v1/ollama/models/inspect-batch` — sequential **`/api/show`** for up to **8** model names (per-model success/error — agent/model introspection)
   - `GET /v1/system/snapshot` — Unix load averages + optional **psutil** top RSS processes (`top_n` query param)
   - `GET /v1/system/process` — this API worker’s **PID**, RSS, optional thread/cpu snapshot (**psutil**)
   - `POST /v1/maintenance/optimize-sqlite` — **`PRAGMA optimize`** on the runs DB when **`PERSIST_RUNS=1`** (planner stats / lightweight maintenance)
   - `POST /v1/maintenance/integrity-sqlite` — **`PRAGMA integrity_check`** on the runs DB when **`PERSIST_RUNS=1`**
   - `POST /v1/maintenance/checkpoint-sqlite` — **`PRAGMA wal_checkpoint`** on the runs DB (JSON body **`truncate_wal`** optional); typically no-op unless journal mode is WAL
   - `POST /v1/maintenance/vacuum-sqlite` — runs **`VACUUM`** on the runs SQLite DB when **`PERSIST_RUNS=1`** (returns before/after file sizes)
   - `GET /v1/maintenance/stats-sqlite` — **`PRAGMA`** page counts / journal mode / schema versions for **`RUNS_DB_PATH`** (read-only; **`exists: false`** when the file is absent)

3. Point the **API base URL** in `index.html` (saved in browser localStorage) at your running server. Set `LEGAL_INTEL_CORS_ORIGINS` on the API if you restrict origins (default allows `*`).

### Ollama (multimodel)

Ollama exposes an OpenAI-compatible HTTP API. Example:

```bash
ollama pull llama3.2
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.2
# Optional per-task models (fallback: LLM_MODEL)
export LLM_MODEL_EXTRACTION=llama3.2
export LLM_MODEL_SPECIALIST=llama3.2
export LLM_MODEL_SYNTHESIS=llama3.2
export LEGAL_INTEL_MOCK_LLM=0
legal-api
```

Environment names map to `llm_provider`, `ollama_base_url`, `llm_model_*` in [`src/legal_intel/config.py`](src/legal_intel/config.py) (`LLM_PROVIDER`, `OLLAMA_BASE_URL`, etc.).

**Embeddings**: Default `EMBEDDING_PROVIDER=ollama` calls Ollama’s native `POST /api/embed` with `OLLAMA_EMBEDDING_MODEL` (e.g. `nomic-embed-text` after `ollama pull nomic-embed-text`), keeping vectors on the same stack as your agents. Use `EMBEDDING_PROVIDER=sentence_transformers` and `EMBEDDING_MODEL=…` for offline tests or when you prefer HuggingFace weights.

**Audit trail**: Set `LEGAL_INTEL_AUDIT_JSONL` to an absolute or relative path for append-only JSON lines on mutating `/v1/*` calls (method, path, status, duration, `request_id`). Redacted in `/v1/settings/effective`.

**Optional host metrics**: `pip install -e ".[device]"` installs `psutil` for RAM/disk detail in [`GET /v1/runtime`](src/legal_intel/runtime/device_profile.py) (`device` object).

**Health warnings**: When `LLM_PROVIDER=ollama`, `/health` compares routed models to `ollama_models` from `/api/tags` and lists actionable warnings (e.g. run `ollama pull`).

## Production-shaped stack (MI300X + vLLM)

1. Start **Qdrant**: `docker compose up -d` (or `QDRANT_URL=:memory:`).
2. Serve your model with **vLLM** on ROCm (e.g. `vllm/vllm-openai-rocm`, `VLLM_ROCM_USE_AITER=1` per AMD docs).
3. Configure `.env`:

```bash
OPENAI_API_BASE=http://<host>:8000/v1
OPENAI_API_KEY=EMPTY
LLM_MODEL=<served-model-id>
LEGAL_INTEL_MOCK_LLM=0
DILIGENCE_DOMAIN=india_re
```

## India mode: architecture

- **Ingest**: PyMuPDF → chunks → embeddings → Qdrant (payload includes `doc_id`, `page_count`, etc.).
- **Graph**: `retrieve` → **structured instrument extraction** (JSON) → **TitleGraph** (Python: link + breaks) → chain/continuity → encumbrance/dispute → records context → **synthesis** (memo + disclaimer).
- **Per-doc RAG**: `LegalVectorStore.search(..., doc_id=...)` scopes retrieval for extraction.
- **Evaluation**: `scripts/eval_india.py` compares run outputs to `tests/fixtures/india_packets/expected.json` (field F1 proxy, chain-break flag, simple groundedness proxy). Methodology is intentionally simple—extend for your labeled set.

## Data governance (India / DPDP)

- The **Digital Personal Data Protection Act, 2023** applies to processing of digital personal data in India. Minimize PII in logs; do not retain client packets for demos longer than needed.
- Optional helper: `legal_intel.privacy.redact_aadhaar_like()` for best-effort masking of 12-digit-like sequences (not a compliance guarantee).

## Tests

```bash
export LEGAL_INTEL_MOCK_LLM=1
export QDRANT_URL=:memory:
export DILIGENCE_DOMAIN=mna   # optional: exercise M&A tests explicitly
pytest -q
```

## CLI

```bash
legal-diligence deed1.pdf deed2.pdf -q "Title chain and EC risk" --domain india_re --json
```

## Repository layout

- `src/legal_intel/` — core library; `legal_intel/india/` — schemas, TitleGraph, India prompts, extraction.
- `src/legal_intel/privacy.py` — optional redaction helper.
- `streamlit_app.py` — demo UI (optional).
- `index.html` — static dashboard for hybrid deploy (Vercel + FastAPI backend).
- `src/legal_intel/api/` — FastAPI application.
- `scripts/eval_india.py` — offline scoring helper.
- `docker-compose.yml` — Qdrant only.

## OCR note

Scanned deeds without a text layer may yield empty extraction. For production, add OCR (e.g. Tesseract or Unstructured) upstream of `ingest_pdf`.
