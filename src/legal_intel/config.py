from __future__ import annotations

import functools
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

DiligenceDomain = Literal["mna", "india_re"]
LlmProvider = Literal["openai_compatible", "ollama"]
LlmTaskKind = Literal["extraction", "specialist", "synthesis"]
ChunkingMode = Literal["fixed", "structural"]
OcrBackend = Literal["tesseract", "paddle"]
EmbeddingProvider = Literal["sentence_transformers", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    # LLM — default to local Ollama agents (OpenAI-compatible /v1); override for vLLM/cloud
    llm_provider: LlmProvider = "ollama"
    """Use `ollama` with `ollama_base_url` (OpenAI-compatible prefix, e.g. http://localhost:11434/v1)."""
    openai_api_base: str = "http://localhost:8000/v1"
    openai_api_key: str = "EMPTY"
    llm_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_model_extraction: str = ""
    llm_model_specialist: str = ""
    llm_model_synthesis: str = ""
    legal_intel_mock_llm: bool = False
    llm_redact_pii: bool = True
    llm_json_mode_extraction: bool = True
    # Domain
    diligence_domain: DiligenceDomain = "india_re"
    # Extraction
    extraction_max_pages: int = 10
    titlegraph_name_fuzzy_threshold: float = 0.82
    # RAG — default embeddings via Ollama (/api/embed); tests force sentence_transformers
    embedding_provider: EmbeddingProvider = "ollama"
    embedding_model: str = "BAAI/bge-m3"
    ollama_embedding_model: str = "nomic-embed-text"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_chunks"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    chunking_mode: ChunkingMode = "structural"
    retrieval_top_k: int = 12
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_rerank_multiplier: int = 4
    rerank_strict: bool = False
    # OCR
    ocr_enabled: bool = True
    ocr_backend: OcrBackend = "tesseract"
    ocr_lang: str = "eng+hin+kan+tam+tel+mar"
    ocr_dpi: int = 300
    # Scraper
    scraper_data_dir: str = "data/raw"
    scraper_rate_limit: float = 2.0
    scraper_max_pages: int = 50
    # Indian Kanoon API (https://api.indiankanoon.org) — token from api.indiankanoon.org
    indian_kanoon_api_token: str = ""
    kanoon_max_full_documents: int = 5
    # Training
    training_output_dir: str = "data/training"
    training_max_seq_len: int = 8192
    training_val_split: float = 0.1
    # Agent / graph
    agent_max_retries: int = 3
    agent_tool_timeout: int = 30
    dispute_check_timeout_seconds: float = 15.0
    # Observability (optional; requires langfuse package)
    langfuse_enabled: bool = False
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # Local HTTP API / device storage
    persist_uploads: bool = True
    upload_storage_dir: str = "data/uploads"
    persist_runs: bool = True
    runs_db_path: str = "data/local/runs.db"
    ollama_probe_timeout_seconds: float = 2.0
    # Comma-separated absolute path prefixes allowed for POST /v1/ingest/local (empty = disabled)
    legal_intel_allow_local_paths: str = ""
    # Append-only JSONL path for mutating API audit (empty = disabled)
    legal_intel_audit_jsonl: str = ""


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
