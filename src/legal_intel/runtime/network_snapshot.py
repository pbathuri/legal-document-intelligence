"""Network interfaces and counters (``psutil`` when installed; hostname always)."""

from __future__ import annotations

import socket
from typing import Any


def gather_network_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
    }
    try:
        import psutil

        out["psutil_available"] = True
        ifaces: dict[str, list[dict[str, Any]]] = {}
        for name, addrs in psutil.net_if_addrs().items():
            parsed: list[dict[str, Any]] = []
            for a in addrs:
                fam = getattr(a.family, "name", None) or str(a.family)
                if fam not in ("AF_INET", "AF_INET6"):
                    continue
                parsed.append(
                    {
                        "family": fam,
                        "address": a.address,
                        "netmask": a.netmask or "",
                        "broadcast": getattr(a, "broadcast", "") or "",
                    }
                )
            if parsed:
                ifaces[name] = parsed
        out["interfaces"] = ifaces
        try:
            io = psutil.net_io_counters(pernic=False)
            out["net_bytes_sent"] = int(io.bytes_sent)
            out["net_bytes_recv"] = int(io.bytes_recv)
            out["net_packets_sent"] = int(io.packets_sent)
            out["net_packets_recv"] = int(io.packets_recv)
        except Exception as e:
            out["net_io_counters_error"] = str(e)
    except ImportError:
        out["psutil_available"] = False
        out["psutil_note"] = "pip install psutil for interface addresses (optional extra)"
    except Exception as e:
        out["psutil_available"] = False
        out["psutil_error"] = str(e)
    return out
