"""Test configuration — mock LLM and in-memory Qdrant by default."""

import os

# Fast local embeddings and no cross-encoder in CI (avoid large downloads).
os.environ["LEGAL_INTEL_MOCK_LLM"] = "1"
os.environ["QDRANT_URL"] = ":memory:"
os.environ["DILIGENCE_DOMAIN"] = "mna"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["EMBEDDING_PROVIDER"] = "sentence_transformers"
os.environ["RERANK_ENABLED"] = "false"
