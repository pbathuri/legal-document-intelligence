"""Tests for GAN latent helpers and Ollama synthetic parsing (no live Ollama required)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from legal_intel.training.gan_latent import latent_prompt_features, train_embedding_gan
from legal_intel.training.ollama_synthetic import response_to_example


def test_latent_prompt_features_deterministic():
    z = np.array([0.1, -0.5, 0.3, 0.9, 0.0, 0.2, -0.1, 0.4])
    a = latent_prompt_features(z)
    b = latent_prompt_features(z)
    assert a == b
    assert "district" in a and "consideration_lakhs" in a


def test_response_to_example_roundtrip():
    raw = """Fictional registered sale deed excerpt for training only. This document is generated
for dataset construction and does not refer to any real transaction. The executant and claimant
are placeholders; survey numbers and consideration are invented for structure extraction drills.

---JSON---
{"doc_type": "sale_deed", "seller_names": ["A"], "buyer_names": ["B"], "parcel_ids": ["12/3"], "locality": "Test", "registration_date": null, "consideration_amount": "Rs. 10 lakhs", "mentions_dispute": false, "mentions_encumbrance": false, "synthetic_seed": "sig"}
"""
    row = response_to_example(raw, domain="india_re", model="test-model", latent_signature="sig")
    assert row is not None
    assert "instruction" in row and "input" in row and "output" in row
    out = json.loads(row["output"])
    assert out["doc_type"] == "sale_deed"


try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_train_embedding_gan_runs():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((16, 32)).astype(np.float32)
    r = train_embedding_gan(emb, z_dim=8, epochs=2, batch_size=8, device="cpu")
    assert r.z_dim == 8
    assert r.embed_dim == 32
    assert len(r.losses_g) == 2
    sd = r.generator_state
    assert isinstance(sd, dict) and len(sd) > 0
    # reload generator
    from legal_intel.training.gan_latent import generate_latents

    z, e = generate_latents(sd, z_dim=8, embed_dim=32, count=4, device="cpu")
    assert z.shape == (4, 8) and e.shape == (4, 32)
