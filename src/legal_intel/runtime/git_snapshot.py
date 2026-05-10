"""Best-effort Git metadata from the API process working directory (device-local)."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any


def gather_git_snapshot(*, cwd: str | None = None, timeout_seconds: float = 2.5) -> dict[str, Any]:
    """
    Resolve HEAD / branch / dirty flag when ``cwd`` is inside a Git work tree.
    Never raises — failures become structured fields for dashboards.
    """
    root = cwd or os.getcwd()
    git = shutil.which("git")
    if not git:
        return {
            "git_available": False,
            "reason": "git executable not found on PATH",
            "cwd": root,
        }

    def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

    try:
        inside = _run([git, "-C", root, "rev-parse", "--is-inside-work-tree"])
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return {
                "git_available": True,
                "inside_work_tree": False,
                "cwd": root,
            }

        head = _run([git, "-C", root, "rev-parse", "HEAD"])
        branch = _run([git, "-C", root, "symbolic-ref", "-q", "--short", "HEAD"])
        porcelain = _run([git, "-C", root, "status", "--porcelain"])

        br = branch.stdout.strip() if branch.returncode == 0 else ""
        if not br:
            br = "(detached)"

        dirty = bool(porcelain.stdout.strip())

        return {
            "git_available": True,
            "inside_work_tree": True,
            "cwd": root,
            "commit": head.stdout.strip() if head.returncode == 0 else None,
            "branch": br,
            "dirty": dirty,
        }
    except Exception as e:
        return {
            "git_available": True,
            "inside_work_tree": None,
            "cwd": root,
            "error": str(e),
        }
