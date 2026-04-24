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
LEGAL_INTEL_MOCK_LLM=1 QDRANT_URL=:memory: pytest -q   # 21 tests pass
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
- `streamlit_app.py` — demo UI.
- `scripts/eval_india.py` — offline scoring helper.
- `docker-compose.yml` — Qdrant only.

## OCR note

Scanned deeds without a text layer may yield empty extraction. For production, add OCR (e.g. Tesseract or Unstructured) upstream of `ingest_pdf`.
