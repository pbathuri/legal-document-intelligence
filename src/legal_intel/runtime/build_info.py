"""Process / package metadata for support bundles (no network)."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from typing import Any


def gather_build_info(*, api_version: str) -> dict[str, Any]:
    try:
        pkg = importlib.metadata.version("legal-document-intelligence")
    except importlib.metadata.PackageNotFoundError:
        pkg = "unknown"
    sha = (
        os.environ.get("LEGAL_INTEL_GIT_SHA")
        or os.environ.get("GIT_COMMIT")
        or os.environ.get("SOURCE_VERSION")
        or ""
    ).strip()
    return {
        "package_name": "legal-document-intelligence",
        "package_version": pkg,
        "api_version": api_version,
        "git_sha": sha or None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
