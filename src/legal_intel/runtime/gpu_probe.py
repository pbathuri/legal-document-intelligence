"""Best-effort NVIDIA GPU snapshot via local ``nvidia-smi`` (no drivers = no output)."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def nvidia_smi_gpus() -> list[dict[str, Any]] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not (r.stdout or "").strip():
        return None
    out: list[dict[str, Any]] = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            out.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": float(parts[2]),
                    "memory_free_mib": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return out or None
