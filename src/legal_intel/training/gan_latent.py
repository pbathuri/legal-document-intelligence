"""Lightweight embedding-space GAN for synthetic **latent seeds** (not photorealistic text).

Classic text GANs are unstable; here we train a Generator that maps Gaussian noise to vectors in the
same space as sentence-transformer embeddings. Those vectors are **not** decoded to prose directly;
instead they are summarized into deterministic prompt knobs (via ``latent_prompt_features``) that
drive Ollama generation in :mod:`legal_intel.training.ollama_synthetic`.

Requires optional dependency: ``pip install 'legal-document-intelligence[gan]'`` (PyTorch).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError(
            "GAN training requires PyTorch. Install with: "
            "pip install 'legal-document-intelligence[gan]'"
        ) from e


@dataclass
class GanTrainResult:
    generator_state: dict
    z_dim: int
    embed_dim: int
    losses_g: list[float]
    losses_d: list[float]


class _MLPGenerator:
    def __init__(self, z_dim: int, out_dim: int, hidden: int = 256):
        torch = _require_torch()
        nn = torch.nn
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class _MLPDiscriminator:
    def __init__(self, embed_dim: int, hidden: int = 256):
        torch = _require_torch()
        nn = torch.nn
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def latent_prompt_features(z: np.ndarray) -> dict[str, float | int | str]:
    """Map a noise vector *z* (any length) to stable scalar knobs for templated prompts.

    Uses only deterministic numpy ops so the same *z* always yields the same "scenario".
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size == 0:
        z = np.zeros(8, dtype=np.float64)

    def _u(i: int) -> float:
        return float(math.tanh(z[i % z.size]))

    districts = [
        "Bengaluru Urban",
        "Mumbai Suburban",
        "Hyderabad",
        "Pune",
        "Chennai",
        "Ahmedabad",
        "Kolkata",
        "Jaipur",
    ]
    deed_types = ["sale_deed", "gift_deed", "mortgage_deed", "lease_deed", "partition_deed"]
    idx_d = int(abs(z.sum() * 1000)) % len(districts)
    idx_t = int(abs(z.prod() * 1000 + 13)) % len(deed_types) if z.size > 1 else 0
    consideration_lakhs = int(50 + (abs(_u(0)) * 450))
    survey_tail = int(100 + abs(_u(1) * 899))
    enc_flag = bool(abs(_u(2)) > 0.35)
    dispute_flag = bool(abs(_u(3)) > 0.55)
    return {
        "district": districts[idx_d],
        "deed_type": deed_types[idx_t],
        "consideration_lakhs": consideration_lakhs,
        "survey_tail": survey_tail,
        "encumbrance": enc_flag,
        "dispute_mention": dispute_flag,
        "latent_signature": ",".join(f"{x:.4f}" for x in z[: min(8, z.size)]),
    }


def train_embedding_gan(
    real_embeddings: np.ndarray,
    *,
    z_dim: int = 32,
    epochs: int = 200,
    batch_size: int = 64,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    device: str | None = None,
) -> GanTrainResult:
    """Train a tiny DCGAN-style MLP on fixed embeddings (numpy ``float32``, shape [N, dim])."""
    torch = _require_torch()
    nn = torch.nn

    if real_embeddings.ndim != 2:
        raise ValueError("real_embeddings must be 2-dim [N, embed_dim]")
    x_real = np.asarray(real_embeddings, dtype=np.float32)
    n, embed_dim = x_real.shape
    if n < 8:
        raise ValueError("Need at least 8 seed embeddings to train the GAN.")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x_real_t = torch.from_numpy(x_real).to(dev)

    G = _MLPGenerator(z_dim, embed_dim).net.to(dev)
    D = _MLPDiscriminator(embed_dim).net.to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    losses_g: list[float] = []
    losses_d: list[float] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=dev)
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            real_batch = x_real_t[idx]
            bs = real_batch.size(0)
            noise = torch.randn(bs, z_dim, device=dev)
            fake = G(noise)

            # Discriminator
            opt_d.zero_grad()
            out_real = D(real_batch)
            out_fake = D(fake.detach())
            y_real = torch.ones_like(out_real)
            y_fake = torch.zeros_like(out_fake)
            loss_d = bce(out_real, y_real) + bce(out_fake, y_fake)
            loss_d.backward()
            opt_d.step()

            # Generator
            opt_g.zero_grad()
            out_fake_for_g = D(fake)
            loss_g = bce(out_fake_for_g, y_real)
            loss_g.backward()
            opt_g.step()

            epoch_loss_g += float(loss_g.detach().cpu())
            epoch_loss_d += float(loss_d.detach().cpu())
            steps += 1

        losses_g.append(epoch_loss_g / max(steps, 1))
        losses_d.append(epoch_loss_d / max(steps, 1))

    return GanTrainResult(
        generator_state=G.state_dict(),
        z_dim=z_dim,
        embed_dim=embed_dim,
        losses_g=losses_g,
        losses_d=losses_d,
    )


def generate_latents(
    generator_state: dict,
    *,
    z_dim: int,
    embed_dim: int,
    count: int,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample *count* noise vectors and corresponding fake embeddings."""
    torch = _require_torch()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    G = _MLPGenerator(z_dim, embed_dim).net.to(dev)
    G.load_state_dict(generator_state)
    G.eval()
    with torch.no_grad():
        z = torch.randn(count, z_dim, device=dev)
        emb = G(z)
    return z.cpu().numpy(), emb.cpu().numpy()
