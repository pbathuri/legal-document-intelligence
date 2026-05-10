from pathlib import Path

from legal_intel.runtime.local_paths import is_path_under_allowlist, parse_allow_prefixes


def test_parse_allow_prefixes(tmp_path: Path):
    sub = tmp_path / "allowed"
    sub.mkdir()
    raw = f"{sub},"
    prefs = parse_allow_prefixes(raw)
    assert len(prefs) == 1
    assert prefs[0] == sub.resolve()


def test_allowlist_subpath(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    pdf = root / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal")
    assert is_path_under_allowlist(pdf, [root.resolve()]) is True
    assert is_path_under_allowlist(Path("/etc/passwd"), [root.resolve()]) is False
