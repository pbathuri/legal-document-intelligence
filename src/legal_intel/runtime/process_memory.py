"""Current API process memory snapshot (psutil when installed — device-local)."""

from __future__ import annotations

import os
import sys
from typing import Any


def gather_process_memory_snapshot() -> dict[str, Any]:
    """RSS/VMS/thread counts for this process; ``resource.getrusage`` maxrss when available."""
    pid = os.getpid()
    out: dict[str, Any] = {
        "pid": pid,
        "python_executable": sys.executable,
        "platform_note": "ru_maxrss units vary by OS (Linux often KiB; verify host docs).",
    }
    try:
        import psutil

        p = psutil.Process(pid)
        mi = p.memory_info()
        out["psutil_available"] = True
        out["rss_bytes"] = int(mi.rss)
        out["vms_bytes"] = int(getattr(mi, "vms", 0))
        out["num_threads"] = int(p.num_threads())
        try:
            out["open_files_count"] = len(p.open_files())
        except Exception:
            out["open_files_count"] = None
    except Exception as e:
        out["psutil_available"] = False
        out["psutil_error"] = str(e)[:400]
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        out["rusage_maxrss"] = int(ru.ru_maxrss)
    except Exception as e:
        out["resource_error"] = str(e)[:200]
    return out
