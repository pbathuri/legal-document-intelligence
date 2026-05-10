"""Detailed ``platform`` introspection for the API host (stdlib only)."""

from __future__ import annotations

import platform
import sys
import sysconfig
from typing import Any


def gather_platform_detail() -> dict[str, Any]:
    out: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "",
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "architecture": platform.architecture(),
        "executable_prefix": getattr(sys, "prefix", ""),
    }
    try:
        u = platform.uname()
        out["uname"] = {
            "system": u.system,
            "node": u.node,
            "release": u.release,
            "version": u.version,
            "machine": u.machine,
        }
    except Exception as e:
        out["uname_error"] = str(e)
    try:
        out["libc_ver"] = platform.libc_ver()
    except Exception as e:
        out["libc_ver_error"] = str(e)
    try:
        out["py_ldlibrary"] = sysconfig.get_config_var("LDLIBRARY") or ""
    except Exception:
        out["py_ldlibrary"] = ""
    return out
