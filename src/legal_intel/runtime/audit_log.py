"""Append-only JSONL audit trail for mutating HTTP operations (host filesystem)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def append_audit_event(path: Path | str, event: dict[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, default=str, ensure_ascii=False) + "\n"
    with _lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(line)
