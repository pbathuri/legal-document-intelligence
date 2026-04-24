from __future__ import annotations

import re
from dataclasses import dataclass

import fitz  # PyMuPDF

# Split before these legal section markers (case-insensitive).
_LEGAL_SECTION_SPLIT = re.compile(
    r"(?=\n(?:"
    r"SCHEDULE(?:\s+OF\s+PROPERTY)?\b"
    r"|RECITALS\b"
    r"|NOW\s+THIS\s+DEED\b"
    r"|THIS\s+INDENTURE\b"
    r"|WITNESSETH\b"
    r"|ARTICLE\s+[IVXLCDM\d]+\b"
    r"|CHAPTER\s+[IVXLCDM\d]+\b"
    r"|COVENANTS?\b"
    r"|ENDORSEMENT\b"
    r"|DESCRIPTION\s+OF\s+PROPERTY\b"
    r"))",
    re.IGNORECASE,
)

# Numbered clause starts: newline + digit + "." + space + capital word
_NUMBERED_CLAUSE = re.compile(r"\n(?=\d{1,3}\.\s+[A-Z(])")

_DEFAULT_SPLITTERS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class TextChunk:
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    section_label: str | None = None


def load_pdf_text(path: str) -> tuple[str, int]:
    """Extract plain text and page count from a PDF."""
    doc = fitz.open(path)
    try:
        parts: list[str] = []
        for i in range(len(doc)):
            parts.append(doc.load_page(i).get_text("text"))
        return "\n\n".join(parts), len(doc)
    finally:
        doc.close()


def load_pdf_text_by_page(path: str) -> list[tuple[int, str]]:
    """Return list of (page_num_1based, text) per page."""
    doc = fitz.open(path)
    try:
        pages = []
        for i in range(len(doc)):
            pages.append((i + 1, doc.load_page(i).get_text("text")))
        return pages
    finally:
        doc.close()


def _normalize_ws(s: str) -> str:
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _first_line_label(text: str, max_len: int = 80) -> str | None:
    line = text.strip().split("\n", 1)[0].strip()
    if not line or len(line) > max_len:
        return None
    return line[:max_len]


def _split_into_legal_blocks(text: str) -> list[tuple[str | None, str]]:
    """Split text into (section_label_or_none, block) using legal-ish boundaries."""
    text = _normalize_ws(text)
    if not text:
        return []
    # Secondary split on numbered clauses within large blobs
    parts = _LEGAL_SECTION_SPLIT.split(text)
    blocks: list[tuple[str | None, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        sub = _NUMBERED_CLAUSE.split(part)
        for j, seg in enumerate(sub):
            seg = seg.strip()
            if not seg:
                continue
            label = _first_line_label(seg) if j == 0 else None
            blocks.append((label, seg))
    return blocks if blocks else [(None, text)]


def _recursive_window_chunks(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    page_start: int,
    page_end: int,
    start_index: int,
    section_label: str | None,
) -> list[TextChunk]:
    """Split a single block into overlapping windows when it exceeds chunk_size."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [
            TextChunk(
                text=text,
                page_start=page_start,
                page_end=page_end,
                chunk_index=start_index,
                section_label=section_label,
            )
        ]
    chunks: list[TextChunk] = []
    idx = start_index
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Prefer splitting at nearest separator before end (within last25% of window)
        if end < n:
            window = text[start:end]
            best = -1
            search_from = max(0, len(window) * 3 // 4)
            for sep in _DEFAULT_SPLITTERS:
                if not sep:
                    continue
                pos = window.rfind(sep, search_from)
                if pos >= 0:
                    cand = pos + len(sep)
                    best = cand if best < 0 else max(best, cand)
            if best >= 0:
                end = start + best
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                TextChunk(
                    text=piece,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_index=idx,
                    section_label=section_label,
                )
            )
            idx += 1
        if end >= n:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def chunk_text(
    full_text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    page_count: int = 1,
) -> list[TextChunk]:
    """Fixed-size sliding window over full document (legacy)."""
    text = _normalize_ws(full_text)
    if not text:
        return []
    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            ps = max(1, min(page_count, 1 +
                     int((start / max(n, 1)) * (page_count - 1))))
            pe = max(ps, min(page_count, 1 +
                     int((end / max(n, 1)) * (page_count - 1))))
            chunks.append(TextChunk(text=piece, page_start=ps,
                          page_end=pe, chunk_index=idx))
            idx += 1
        if end >= n:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def chunk_text_structural(
    full_text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    page_count: int = 1,
) -> list[TextChunk]:
    """Structural boundaries first, then recursive windows; page range is approximate."""
    blocks = _split_into_legal_blocks(full_text)
    if not blocks:
        return []
    out: list[TextChunk] = []
    idx = 0
    for label, block in blocks:
        for c in _recursive_window_chunks(
            block,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_start=1,
            page_end=max(1, page_count),
            start_index=idx,
            section_label=label,
        ):
            out.append(c)
            idx = c.chunk_index + 1
    return out


def chunk_pages_structural(
    pages: list[tuple[int, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Structural chunking with accurate page_start / page_end per page text."""
    if not pages:
        return []
    out: list[TextChunk] = []
    idx = 0
    for page_num, raw in pages:
        blocks = _split_into_legal_blocks(raw)
        if not blocks:
            continue
        for label, block in blocks:
            for c in _recursive_window_chunks(
                block,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                page_start=page_num,
                page_end=page_num,
                start_index=idx,
                section_label=label,
            ):
                out.append(c)
                idx = c.chunk_index + 1
    return out
