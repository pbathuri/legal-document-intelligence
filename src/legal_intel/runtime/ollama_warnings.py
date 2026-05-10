"""Detect configuration mismatches vs locally installed Ollama models."""

from __future__ import annotations

from legal_intel.config import Settings
from legal_intel.llm.client import resolve_model_for_task


def _normalized_tag(tag: str) -> str:
    return tag.split(":")[0].strip().lower()


def _model_resolves(requested: str, available_tags: list[str]) -> bool:
    if not requested.strip():
        return True
    req = _normalized_tag(requested)
    if not req:
        return True
    for t in available_tags:
        base = _normalized_tag(t)
        if req == base or req in base or base.startswith(req):
            return True
    return False


def build_ollama_model_warnings(settings: Settings, ollama_tags: list[str] | None) -> list[str]:
    """Return human-readable warnings when routed models may be missing from Ollama."""
    if settings.llm_provider != "ollama":
        return []
    if ollama_tags is None:
        return []
    if len(ollama_tags) == 0:
        return [
            "Ollama reports zero installed models (/api/tags empty). Run e.g. `ollama pull llama3.2`."
        ]
    warnings: list[str] = []
    checks = [
        ("default", settings.llm_model),
        ("extraction", resolve_model_for_task("extraction")),
        ("specialist", resolve_model_for_task("specialist")),
        ("synthesis", resolve_model_for_task("synthesis")),
    ]
    seen: set[str] = set()
    for role, model in checks:
        key = model.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if not _model_resolves(key, ollama_tags):
            warnings.append(
                f"Model `{key}` ({role}) not found in local Ollama tags — run `ollama pull {key.split(':')[0]}`"
            )
    return warnings
