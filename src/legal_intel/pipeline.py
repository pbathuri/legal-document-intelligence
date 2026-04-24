from __future__ import annotations

import hashlib
from pathlib import Path

from legal_intel.config import get_settings
from legal_intel.ingest.pdf_loader import (
    chunk_pages_structural,
    chunk_text,
    chunk_text_structural,
    load_pdf_text_by_page,
)
from legal_intel.rag.store import LegalVectorStore


def _doc_id_from_path(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:16]
    return f"{p.stem}_{h}"


def ingest_pdf(path: str, *, doc_label: str | None = None, use_ocr: bool = False) -> tuple[str, int]:
    """Load PDF, chunk, embed, and upsert into Qdrant. Returns (doc_id, num_chunks)."""
    s = get_settings()
    label = doc_label or Path(path).name
    doc_id = _doc_id_from_path(path)

    if use_ocr and s.ocr_enabled:
        from legal_intel.ocr.engine import ocr_pdf

        pages = ocr_pdf(path)
    else:
        pages = load_pdf_text_by_page(path)

    text = "\n\n".join(t for _, t in pages)
    page_count = max(1, len(pages))

    if s.chunking_mode == "structural":
        chunks = chunk_pages_structural(
            pages, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap
        )
        if not chunks and text.strip():
            chunks = chunk_text_structural(
                text,
                chunk_size=s.chunk_size,
                chunk_overlap=s.chunk_overlap,
                page_count=page_count,
            )
    else:
        chunks = chunk_text(
            text,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            page_count=page_count,
        )

    store = LegalVectorStore()
    tuples: list[tuple[str, dict]] = []
    for c in chunks:
        extra: dict = {
            "page_start": c.page_start,
            "page_end": c.page_end,
            "page_count": page_count,
            "chunk_index": c.chunk_index,
        }
        if c.section_label:
            extra["section_label"] = c.section_label
        tuples.append((c.text, extra))
    n = store.upsert_document_chunks(
        doc_id=doc_id, doc_label=label, chunks=tuples)
    return doc_id, n
