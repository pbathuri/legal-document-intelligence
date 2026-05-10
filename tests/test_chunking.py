from legal_intel.ingest.pdf_loader import (
    chunk_pages_structural,
    chunk_text,
    chunk_text_structural,
)


def test_chunk_produces_multiple_chunks():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50, page_count=5)
    assert len(chunks) > 1
    for c in chunks:
        assert c.page_start >= 1
        assert c.page_end <= 5
        assert c.chunk_index >= 0
        assert c.section_label is None


def test_empty_text_produces_no_chunks():
    assert chunk_text("", chunk_size=200, chunk_overlap=50) == []


def test_short_text_single_chunk():
    chunks = chunk_text("hello world", chunk_size=200, chunk_overlap=50, page_count=1)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_structural_split_schedule():
    text = "Intro line\n\nSCHEDULE OF PROPERTY\nPlot 1 details.\n\nRECITALS\nSome recital."
    chunks = chunk_text_structural(text, chunk_size=500, chunk_overlap=50, page_count=2)
    assert len(chunks) >= 2
    labels = [c.section_label for c in chunks if c.section_label]
    assert (
        any(
            "SCHEDULE" in (lbl or "").upper() or "RECITALS" in (lbl or "").upper() for lbl in labels
        )
        or len(chunks) >= 2
    )


def test_chunk_pages_structural_per_page():
    pages = [
        (1, "Seller A sells to Buyer B.\n\nSCHEDULE\nSurvey 12/A"),
        (2, "Witness clause continuation."),
    ]
    chunks = chunk_pages_structural(pages, chunk_size=80, chunk_overlap=10)
    assert len(chunks) >= 1
    for c in chunks:
        assert 1 <= c.page_start <= 2
        assert 1 <= c.page_end <= 2
