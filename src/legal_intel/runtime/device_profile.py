"""Best-effort host profile using stdlib; augments with psutil when installed."""

from __future__ import annotations

import os
import socket
from typing import Any


def gather_device_profile() -> dict[str, Any]:
    out: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        du = psutil.disk_usage(os.getcwd())
        out["psutil_available"] = True
        out["memory_total_bytes"] = vm.total
        out["memory_available_bytes"] = vm.available
        out["swap_total_bytes"] = sm.total
        out["disk_workdir_total_bytes"] = du.total
        out["disk_workdir_free_bytes"] = du.free
    except ImportError:
        out["psutil_available"] = False
        out["psutil_note"] = "pip install psutil for RAM/disk detail (optional extra)"
    except Exception as e:
        out["psutil_available"] = False
        out["psutil_error"] = str(e)

    try:
        from legal_intel.runtime.gpu_probe import nvidia_smi_gpus

        gpus = nvidia_smi_gpus()
        if gpus is not None:
            out["nvidia_gpus"] = gpus
    except Exception as e:
        out["nvidia_probe_error"] = str(e)
    return out
