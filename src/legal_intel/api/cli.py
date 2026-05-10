"""Run the FastAPI server (hybrid backend for Vercel + local Ollama)."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("LEGAL_INTEL_API_HOST", "0.0.0.0")
    port = int(os.environ.get("LEGAL_INTEL_API_PORT", os.environ.get("PORT", "8080")))
    uvicorn.run(
        "legal_intel.api.main:app",
        host=host,
        port=port,
        reload=os.environ.get("LEGAL_INTEL_API_RELOAD", "").lower() in ("1", "true", "yes"),
    )


if __name__ == "__main__":
    main()
