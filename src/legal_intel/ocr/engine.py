"""OCR pipeline for scanned Indian property documents.

Tesseract (default) or optional PaddleOCR backend. Falls back to PyMuPDF text
when OCR is disabled or engines are unavailable.
"""
from __future__ import annotations

import io
import logging

import fitz  # PyMuPDF

from legal_intel.config import get_settings

logger = logging.getLogger(__name__)


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _paddle_available() -> bool:
    try:
        import paddleocr  # noqa: F401

        return True
    except Exception:
        return False


def ocr_image(image_bytes: bytes, lang: str | None = None) -> str:
    """OCR a single image (bytes) and return text."""
    s = get_settings()
    lang = lang or s.ocr_lang

    if s.ocr_backend == "paddle" and _paddle_available():
        return _ocr_image_paddle(image_bytes)
    if not _tesseract_available():
        logger.warning("Tesseract not available; returning empty string.")
        return ""
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(img, lang=lang)


_paddle_engine = None


def _get_paddle_engine():
    global _paddle_engine
    if _paddle_engine is None:
        from paddleocr import PaddleOCR

        _paddle_engine = PaddleOCR(
            use_angle_cls=True, lang="en", show_log=False)
    return _paddle_engine


def _ocr_image_paddle(image_bytes: bytes) -> str:
    ocr = _get_paddle_engine()
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    result = ocr.ocr(arr, cls=True)
    lines: list[str] = []
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2 and line[1]:
                lines.append(str(line[1][0]))
    return "\n".join(lines)


def ocr_pdf(path: str, *, force_ocr: bool = False) -> list[tuple[int, str]]:
    """OCR each page of a PDF. Returns list of (page_num_1based, text).

    Strategy:
    1. Try PyMuPDF text extraction first
    2. If a page has < 50 chars of text and OCR is enabled, rasterize + OCR
    3. If force_ocr, always rasterize
    """
    s = get_settings()
    doc = fitz.open(path)
    pages: list[tuple[int, str]] = []
    use_ocr = s.ocr_enabled and (_tesseract_available() or (
        s.ocr_backend == "paddle" and _paddle_available()))

    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text").strip()

            if (force_ocr or len(text) < 50) and use_ocr:
                mat = fitz.Matrix(s.ocr_dpi / 72, s.ocr_dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                ocr_text = ocr_image(img_bytes, lang=s.ocr_lang)
                if len(ocr_text.strip()) > len(text):
                    text = ocr_text.strip()

            pages.append((i + 1, text))
    finally:
        doc.close()

    return pages


def load_pdf_with_ocr(path: str) -> tuple[str, int]:
    """Drop-in replacement for load_pdf_text that includes OCR fallback."""
    pages = ocr_pdf(path)
    full_text = "\n\n".join(text for _, text in pages)
    return full_text, len(pages)
