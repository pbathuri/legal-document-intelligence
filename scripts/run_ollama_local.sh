#!/usr/bin/env bash
set -euo pipefail

OLLAMA_ORIGIN="${OLLAMA_ORIGIN:-http://localhost:11434}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-${OLLAMA_ORIGIN%/}/v1}"
OLLAMA_EMBEDDING_MODEL="${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"
LEGAL_INTEL_API_HOST="${LEGAL_INTEL_API_HOST:-127.0.0.1}"
LEGAL_INTEL_API_PORT="${LEGAL_INTEL_API_PORT:-8080}"

wait_for_ollama() {
  for _ in $(seq 1 40); do
    if curl -fsS "${OLLAMA_ORIGIN%/}/api/version" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

if ! wait_for_ollama; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama CLI not found. Install Ollama first: https://ollama.com/download" >&2
    exit 1
  fi
  echo "Starting Ollama daemon at ${OLLAMA_ORIGIN} ..."
  ollama serve >/tmp/legal-intel-ollama.log 2>&1 &
  if ! wait_for_ollama; then
    echo "Ollama did not become ready. See /tmp/legal-intel-ollama.log" >&2
    exit 1
  fi
fi

echo "Ollama is ready: $(curl -fsS "${OLLAMA_ORIGIN%/}/api/version")"

if ! curl -fsS "${OLLAMA_ORIGIN%/}/api/embed" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${OLLAMA_EMBEDDING_MODEL}\",\"input\":[\"ping\"]}" >/dev/null 2>&1; then
  echo "Pulling Ollama embedding model: ${OLLAMA_EMBEDDING_MODEL}"
  ollama pull "${OLLAMA_EMBEDDING_MODEL}"
fi

if ! curl -fsS "${OLLAMA_ORIGIN%/}/api/embed" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${OLLAMA_EMBEDDING_MODEL}\",\"input\":[\"ping\"]}" >/dev/null 2>&1; then
  echo "Embedding model ${OLLAMA_EMBEDDING_MODEL} is still not usable through /api/embed." >&2
  exit 1
fi

if [[ -z "${LLM_MODEL:-}" ]]; then
  LLM_MODEL="$(
    OLLAMA_ORIGIN="$OLLAMA_ORIGIN" python3 - <<'PY'
import os
import httpx

origin = os.environ.get("OLLAMA_ORIGIN", "http://localhost:11434").rstrip("/")
try:
    models = httpx.get(f"{origin}/api/tags", timeout=5).json().get("models", [])
except Exception:
    models = []

preferred = ["gemma4:26b", "llama3.2:latest", "llama3.2", "mistral:latest", "gemma2:latest"]
names = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]

def supports_generation(name: str) -> bool:
    try:
        r = httpx.post(
            f"{origin}/api/generate",
            json={"model": name, "prompt": "Reply with one word: pong", "stream": False},
            timeout=180,
        )
        return r.status_code == 200
    except Exception:
        return False

for name in preferred + names:
    if name in names and supports_generation(name):
        print(name)
        raise SystemExit

raise SystemExit("No installed Ollama model supports /api/generate. Pull one, e.g. `ollama pull gemma3`.")
PY
  )"
fi

echo "Using Ollama LLM model: ${LLM_MODEL}"
echo "Using Ollama embedding model: ${OLLAMA_EMBEDDING_MODEL}"

export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL
export LLM_MODEL
export EMBEDDING_PROVIDER=ollama
export OLLAMA_EMBEDDING_MODEL
export QDRANT_URL="${QDRANT_URL:-:memory:}"

exec python -m uvicorn legal_intel.api.main:app --host "$LEGAL_INTEL_API_HOST" --port "$LEGAL_INTEL_API_PORT"
