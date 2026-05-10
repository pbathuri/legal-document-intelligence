"""Synthetic SFT JSONL using **Ollama** (multi-model rotation) + optional **GAN latents** as prompt seeds.

Outputs Alpaca-style rows compatible with :func:`legal_intel.training.prepare.prepare_dataset`
consumption (``instruction``, ``input``, ``output``) plus metadata fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from legal_intel.training.gan_latent import latent_prompt_features

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA = "http://localhost:11434"

USER_PROMPT_INDIA = """You are generating **training data only** (fictional documents).

Write ONE fictional Indian property instrument excerpt (180–320 words), plausible but not copying any real party names.

Then output a line containing exactly: ---JSON---
Then output a single JSON object with keys:
  doc_type (string),
  registration_date (YYYY-MM-DD or null),
  seller_names (array of strings),
  buyer_names (array of strings),
  parcel_ids (array of strings),
  locality (string),
  consideration_amount (string or null),
  mentions_dispute (boolean),
  mentions_encumbrance (boolean)

Scenario knobs (must reflect these in the prose and JSON):
- District/city emphasis: {district}
- Instrument flavor: {deed_type}
- Consideration magnitude hint (Rs., lakhs scale): ~{consideration_lakhs} lakhs
- Survey / plot tail digits hint: ...{survey_tail}
- Plot mentions encumbrance (charge/mortgage): {encumbrance}
- Plot mentions dispute/litigation: {dispute_mention}
- Deterministic variation id (echo verbatim in JSON field "synthetic_seed"): "{latent_signature}"
"""

USER_PROMPT_MNA = """You are generating **training data only** (fictional contract excerpts).

Write ONE fictional M&A-style contract snippet (180–320 words): representations, indemnity, or closing conditions.

Then a line exactly: ---JSON---
Then JSON with keys:
  parties (array of 2 strings),
  governing_law (string),
  closing_condition_summary (string),
  liability_cap_hint (string),
  material_adverse_change (boolean),
  synthetic_seed (string)

Scenario knobs:
- Jurisdiction tone: {district}
- Deal stage flavor: {deed_type}
- Consideration scale hint (abstract units): {consideration_lakhs}
- Reference id tail: {survey_tail}
- Mentions carve-outs or escrow tension: {encumbrance}
- Mentions litigation/arbitration clause tension: {dispute_mention}
- synthetic_seed: "{latent_signature}"
"""


def ollama_origin(host: str) -> str:
    h = host.rstrip("/")
    if h.endswith("/v1"):
        return h[: -len("/v1")]
    return h


def fetch_tag_models(host: str, timeout: float = 30.0) -> list[str]:
    origin = ollama_origin(host)
    r = httpx.get(f"{origin}/api/tags", timeout=timeout)
    r.raise_for_status()
    data = r.json()
    models = data.get("models") or []
    names: list[str] = []
    for m in models:
        if isinstance(m, dict) and m.get("name"):
            names.append(m["name"])
    return sorted(set(names))


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    timeout: float = 300.0,
) -> str:
    """Native Ollama generate API (non-streaming)."""
    origin = ollama_origin(host)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    r = httpx.post(f"{origin}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _split_json_block(text: str) -> tuple[str, str]:
    if "---JSON---" in text:
        prose, rest = text.split("---JSON---", 1)
        return prose.strip(), rest.strip()
    # fallback: last {...} block
    m = re.search(r"(\{[\s\S]*\})\s*$", text.strip())
    if m:
        return text[: m.start()].strip(), m.group(1)
    return text.strip(), ""


def _parse_json_loose(blob: str) -> dict[str, Any]:
    blob = blob.strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", blob)
        if m:
            return json.loads(m.group(0))
        raise


def build_instruction(domain: str) -> str:
    if domain == "mna":
        return (
            "You are an AI assistant for M&A contract analysis. "
            "Given the contract excerpt, structured diligence outputs are provided as JSON."
        )
    return (
        "You are an AI assistant specialized in Indian property document analysis. "
        "Given the document text, extraction targets are expressed as JSON."
    )


def response_to_example(
    raw: str,
    *,
    domain: str,
    model: str,
    latent_signature: str,
) -> dict[str, str] | None:
    prose, json_blob = _split_json_block(raw)
    if not prose or len(prose) < 80:
        logger.warning("Discarding short or empty prose generation.")
        return None
    try:
        payload = _parse_json_loose(json_blob)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("JSON parse failed: %s", e)
        return None
    out_str = json.dumps(payload, ensure_ascii=False)
    row = {
        "instruction": build_instruction(domain),
        "input": prose,
        "output": out_str,
        "model": model,
        "is_synthetic": True,
        "synthetic_backend": "ollama",
        "domain": domain,
        "latent_signature": latent_signature,
    }
    return row


def generate_dataset_rows(
    *,
    host: str,
    models: list[str],
    n_total: int,
    domain: str,
    z_vectors: Sequence[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Produce *n_total* rows, rotating models and optionally using supplied latent vectors."""
    rows: list[dict[str, Any]] = []
    if not models:
        raise ValueError("No Ollama models provided.")

    for i in range(n_total):
        model = models[i % len(models)]
        if z_vectors is not None and i < len(z_vectors):
            z = z_vectors[i]
        else:
            rng = np.random.default_rng(int(hashlib.sha256(f"{i}-{model}".encode()).hexdigest()[:8], 16))
            z = rng.standard_normal(32)
        feats = latent_prompt_features(z)
        latent_sig = feats["latent_signature"]
        prompt_template = USER_PROMPT_MNA if domain == "mna" else USER_PROMPT_INDIA
        user_prompt = prompt_template.format(**feats)
        sys_prompt = (
            "You write concise fictional legal training samples. Output plain text followed by ---JSON---."
        )
        logger.info("Generating %s/%s with model=%s", i + 1, n_total, model)
        raw = ollama_generate(host, model, user_prompt, system=sys_prompt)
        ex = response_to_example(raw, domain=domain, model=model, latent_signature=latent_sig)
        if ex:
            rows.append(ex)
    return rows


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Generate synthetic SFT JSONL via Ollama (multi-model) + latent scenario knobs.",
    )
    ap.add_argument("--ollama-host", default=DEFAULT_OLLAMA, help="Ollama base, e.g. http://localhost:11434")
    ap.add_argument(
        "--domain",
        choices=("india_re", "mna"),
        default="india_re",
        help="Prompt family",
    )
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/ollama_synth.jsonl"))
    ap.add_argument("--n", type=int, default=8, help="Total synthetic rows to attempt")
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated Ollama model names; empty = use all tags from /api/tags",
    )
    ap.add_argument(
        "--gan-checkpoint",
        type=Path,
        default=None,
        help="Optional .pt from legal-gan-embed (torch.save payload with generator_state, z_dim, embed_dim)",
    )
    args = ap.parse_args(argv)

    host = args.ollama_host.strip()
    try:
        tag_models = fetch_tag_models(host)
    except Exception as e:
        logger.error("Cannot reach Ollama at %s: %s", host, e)
        sys.exit(1)

    if args.models.strip():
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = tag_models

    if not models:
        logger.error("No models found. Pull one with: ollama pull llama3.2")
        sys.exit(1)

    z_vectors: list[np.ndarray] | None = None
    if args.gan_checkpoint and args.gan_checkpoint.exists():
        import torch

        from legal_intel.training.gan_latent import generate_latents

        ck = torch.load(args.gan_checkpoint, map_location="cpu")
        if not isinstance(ck, dict) or "generator_state" not in ck:
            logger.error("Checkpoint must be a dict with generator_state, z_dim, embed_dim")
            sys.exit(1)
        state = ck["generator_state"]
        z_dim = int(ck["z_dim"])
        embed_dim = int(ck["embed_dim"])
        z_np, _ = generate_latents(state, z_dim=z_dim, embed_dim=embed_dim, count=args.n)
        z_vectors = [z_np[i] for i in range(z_np.shape[0])]
        logger.info("Using %s GAN-latent vectors from %s", len(z_vectors), args.gan_checkpoint)

    rows = generate_dataset_rows(
        host=host,
        models=models,
        n_total=args.n,
        domain=args.domain,
        z_vectors=z_vectors,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Wrote %d rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
