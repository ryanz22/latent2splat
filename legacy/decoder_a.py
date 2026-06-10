"""Approach A — token-aligned decoder (primary).

Each of the 1536 latent tokens (T·H·W = 4·24·16) owns K Gaussians whose means
are bounded residuals to a canonical 3D position. The canonical positions are
a fixed (t,h,w)-indexed grid inside a unit cube — the single intentional
camera-shaped inductive bias. A small transformer mixes the tokens before the
per-token Gaussian head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gaussian_head import GaussianHead, split_gaussian_params

# Spatial/channel dims are fixed by the VAE (512/768 → 16/24, 128 ch). The
# TEMPORAL dim depends on the input frame count: T = (frames-1)//8 + 1.
# v1 data (25 frames) → T=4; v2 data (9 frames) → T=2. T is therefore a
# per-model argument, not a module constant.
LATENT_C, LATENT_H, LATENT_W = 128, 24, 16
LATENT_T_DEFAULT = 2  # v2 default (9-frame input)


def _canonical_grid(t: int, h: int, w: int) -> torch.Tensor:
    """(t·h·w, 3) grid of canonical means spanning [-0.5,0.5]^3.

    The temporal axis maps to a depth-ish third spatial axis: it is a prior,
    not ground truth, and the residual head moves means off it freely.
    """
    zs = torch.linspace(-0.5, 0.5, t)
    ys = torch.linspace(-0.5, 0.5, h)
    xs = torch.linspace(-0.5, 0.5, w)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


class TokenAlignedDecoder(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        k: int = 32,
        latent_t: int = LATENT_T_DEFAULT,
        offset_max: float = 1.0,
        scale_max: float = 0.3,
    ):
        super().__init__()
        self.k = k
        self.latent_t = latent_t
        self.num_tokens = latent_t * LATENT_H * LATENT_W
        self.offset_max = offset_max
        self.scale_max = scale_max

        self.in_proj = nn.Linear(LATENT_C, dim)
        self.pos_emb = nn.Parameter(torch.zeros(self.num_tokens, dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = GaussianHead(dim, k=k)

        self.register_buffer(
            "canonical", _canonical_grid(latent_t, LATENT_H, LATENT_W), persistent=False
        )

    def forward(self, latent: torch.Tensor) -> dict:
        """latent (B,128,T,24,16) -> dict of (B, N, ...) Gaussian params, N=tokens·K."""
        b = latent.shape[0]
        # (B,C,T,H,W) -> (B, T*H*W, C)
        x = latent.flatten(2).transpose(1, 2)
        x = self.in_proj(x) + self.pos_emb
        x = self.encoder(x)
        raw = self.head(x)  # (B, tokens, K, 14)

        # canonical mean per token, broadcast across K
        canon = self.canonical.to(raw.dtype)[None, :, None, :]  # (1,tokens,1,3)
        params = split_gaussian_params(
            raw, canonical_mean=canon, offset_max=self.offset_max, scale_max=self.scale_max
        )
        return {k_: v.reshape(b, -1, v.shape[-1]) for k_, v in params.items()}
