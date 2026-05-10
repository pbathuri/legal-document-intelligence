"""Extended host metrics (CPU sample, boot time, disk partitions) via optional psutil."""

from __future__ import annotations

import os
import sys
from typing import Any


def gather_extended_host_metrics(*, cpu_interval: float = 0.07) -> dict[str, Any]:
    """
    One-shot CPU percent + memory/disk utilization + boot time + bounded partition list.
    Falls back gracefully when psutil is not installed.
    """
    out: dict[str, Any] = {
        "cpu_interval_seconds": cpu_interval,
    }
    try:
        import psutil

        out["psutil_available"] = True
        out["cpu_percent"] = float(psutil.cpu_percent(interval=cpu_interval))
        phys = psutil.cpu_count(logical=False)
        logi = psutil.cpu_count(logical=True)
        if phys is not None:
            out["cpu_count_physical"] = phys
        if logi is not None:
            out["cpu_count_logical"] = logi
        if hasattr(os, "getloadavg"):
            try:
                la = os.getloadavg()
                out["loadavg_1m"] = la[0]
                out["loadavg_5m"] = la[1]
                out["loadavg_15m"] = la[2]
            except OSError:
                pass
        vm = psutil.virtual_memory()
        out["memory_percent"] = float(vm.percent)
        out["memory_available_bytes"] = int(vm.available)
        sm = psutil.swap_memory()
        out["swap_percent"] = float(sm.percent)
        du = psutil.disk_usage(os.getcwd())
        out["disk_workdir_percent"] = float(du.percent)
        out["disk_workdir_free_bytes"] = int(du.free)
        try:
            out["boot_time_unix"] = float(psutil.boot_time())
        except Exception as e:
            out["boot_time_error"] = str(e)
        parts: list[dict[str, str]] = []
        for p in psutil.disk_partitions(all=False)[:32]:
            parts.append(
                {
                    "device": p.device,
                    "mountpoint": p.mountpoint,
                    "fstype": p.fstype,
                }
            )
        out["disk_partitions_sample"] = parts
    except ImportError:
        out["psutil_available"] = False
        out["psutil_note"] = "pip install psutil for extended metrics (optional extra)"
        out["cpu_count_logical_env"] = os.cpu_count()
        out["python_executable"] = sys.executable
    except Exception as e:
        out["psutil_available"] = False
        out["psutil_error"] = str(e)
    return out
