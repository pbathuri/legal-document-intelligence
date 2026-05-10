"""Unix resource limits via ``resource`` stdlib (device-local FD / memory ceilings)."""

from __future__ import annotations

import platform
from typing import Any


def gather_rlimits_snapshot() -> dict[str, Any]:
    """Best-effort ``getrlimit`` for common RLIMIT_* constants (non-Windows)."""
    out: dict[str, Any] = {"platform": platform.system()}
    if platform.system() == "Windows":
        out["rlimits_available"] = False
        out["note"] = "Python resource module rlimits are not supported on Windows hosts."
        return out
    try:
        import resource

        out["rlimits_available"] = True
        names = (
            "RLIMIT_AS",
            "RLIMIT_CORE",
            "RLIMIT_CPU",
            "RLIMIT_DATA",
            "RLIMIT_FSIZE",
            "RLIMIT_NOFILE",
            "RLIMIT_STACK",
            "RLIMIT_NPROC",
        )
        for name in names:
            if hasattr(resource, name):
                const = getattr(resource, name)
                try:
                    soft, hard = resource.getrlimit(const)
                    out[name.lower()] = {"soft": soft, "hard": hard}
                except (ValueError, OSError) as e:
                    out[name.lower()] = {"error": str(e)}
    except ImportError:
        out["rlimits_available"] = False
        out["note"] = "resource module unavailable"
    except Exception as e:
        out["rlimits_available"] = False
        out["error"] = str(e)
    return out
