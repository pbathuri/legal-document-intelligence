"""CLI: train embedding-space GAN and save checkpoint for :mod:`legal_intel.training.ollama_synthetic`.

Example::

    pip install 'legal-document-intelligence[gan]'
    legal-gan-embed --out data/synthetic/gan_generator.pt
    legal-synth-ollama --gan-checkpoint data/synthetic/gan_generator.pt --n 16 --out data/synthetic/sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _default_seed_texts() -> list[str]:
    """Short synthetic deed-labeled snippets for embedding (GAN training anchors only)."""
    districts = ["Bengaluru", "Mumbai", "Hyderabad", "Chennai", "Pune"]
    types_t = ["sale deed", "gift deed", "lease deed", "mortgage deed"]
    lines: list[str] = []
    for d in districts:
        for t in types_t:
            lines.append(
                f"Registered {t} at Sub-Registrar Office {d}. Consideration Rs. {hash(d + t) % 900 + 50} lakhs. "
                f"Parties: Party A and Party B. Survey Khata {hash(t) % 1000}/A."
            )
    return lines


def _load_seed_texts_from_jsonl(path: Path) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        inp = row.get("input") or row.get("text") or ""
        if isinstance(inp, str) and len(inp) > 40:
            texts.append(inp[:8000])
    return texts


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train embedding GAN; save PyTorch checkpoint for Ollama synth.")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/gan_generator.pt"))
    ap.add_argument("--seed-jsonl", type=Path, default=None, help="Optional JSONL with 'input' fields")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        logger.error("sentence-transformers required: pip install sentence-transformers")
        raise SystemExit(1) from e

    from legal_intel.training.gan_latent import train_embedding_gan

    if args.seed_jsonl and args.seed_jsonl.exists():
        texts = _load_seed_texts_from_jsonl(args.seed_jsonl)
        logger.info("Loaded %d seed texts from %s", len(texts), args.seed_jsonl)
    else:
        texts = _default_seed_texts()
        logger.info("Using %d built-in seed text templates", len(texts))

    if len(texts) < 8:
        logger.error("Need at least 8 seed texts; provide --seed-jsonl with enough rows.")
        raise SystemExit(1)

    logger.info("Encoding with %s …", args.embedding_model)
    model = SentenceTransformer(args.embedding_model)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if emb.ndim != 2:
        raise RuntimeError("Encoder returned unexpected shape")

    logger.info("Training GAN (z_dim=%s, epochs=%s) …", args.z_dim, args.epochs)
    result = train_embedding_gan(
        emb,
        z_dim=args.z_dim,
        epochs=args.epochs,
        batch_size=min(args.batch_size, emb.shape[0]),
    )

    payload: dict[str, Any] = {
        "generator_state": result.generator_state,
        "z_dim": result.z_dim,
        "embed_dim": result.embed_dim,
        "embedding_model": args.embedding_model,
        "final_loss_g": result.losses_g[-1] if result.losses_g else None,
        "final_loss_d": result.losses_d[-1] if result.losses_d else None,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch = __import__("torch")
    torch.save(payload, args.out)
    logger.info(
        "Saved checkpoint to %s (final G loss=%s, D loss=%s)",
        args.out,
        payload["final_loss_g"],
        payload["final_loss_d"],
    )


if __name__ == "__main__":
    main()
