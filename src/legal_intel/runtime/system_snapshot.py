"""Host load and process snapshot using stdlib + optional psutil (device-local)."""

from __future__ import annotations

import os
from typing import Any


def gather_system_snapshot(*, top_n: int = 8) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["loadavg"] = list(os.getloadavg())
    except (AttributeError, OSError):
        out["loadavg"] = None

    try:
        import psutil

        out["psutil_available"] = True
        out["process_count"] = len(psutil.pids())
        ranked: list[tuple[int, int, str]] = []
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            try:
                rss = proc.memory_info().rss
                pid = proc.pid
                name = (proc.name() or "")[:96]
                ranked.append((rss, pid, name))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        ranked.sort(key=lambda x: x[0], reverse=True)
        n = max(1, min(top_n, 32))
        out["top_memory_processes"] = [
            {"pid": pid, "rss_bytes": rss, "name": name} for rss, pid, name in ranked[:n]
        ]
    except ImportError:
        out["psutil_available"] = False
        out["psutil_note"] = "pip install psutil for process snapshot (optional extra)"
    except Exception as e:
        out["psutil_available"] = False
        out["psutil_error"] = str(e)
    return out
