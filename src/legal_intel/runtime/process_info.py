"""Current API process metrics (PID, RSS, threads) — optional psutil."""

from __future__ import annotations

import os
from typing import Any


def gather_api_process_snapshot() -> dict[str, Any]:
    """Lightweight introspection of the running FastAPI worker process."""
    out: dict[str, Any] = {"pid": os.getpid()}
    try:
        import psutil

        p = psutil.Process()
        out["psutil_available"] = True
        out["ppid"] = p.ppid()
        out["name"] = p.name()
        mi = p.memory_info()
        out["rss_bytes"] = mi.rss
        out["vms_bytes"] = getattr(mi, "vms", None)
        try:
            out["num_threads"] = p.num_threads()
        except Exception:
            out["num_threads"] = None
        try:
            out["cpu_percent"] = round(float(p.cpu_percent(interval=0.05)), 3)
        except Exception:
            out["cpu_percent"] = None
        try:
            out["create_time_epoch"] = p.create_time()
        except Exception:
            out["create_time_epoch"] = None
        try:
            out["open_files_count"] = len(p.open_files())
        except Exception:
            out["open_files_count"] = None
    except ImportError:
        out["psutil_available"] = False
        out["psutil_note"] = "pip install psutil for RSS/thread detail"
    except Exception as e:
        out["psutil_available"] = False
        out["psutil_error"] = str(e)
    return out
