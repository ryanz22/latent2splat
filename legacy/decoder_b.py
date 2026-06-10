"""Approach B — learned-query decoder (ablation, no camera prior).

The latent is encoded into a context (same token transformer as A, minus the
canonical grid). N learnable query tokens cross-attend to the context and each
emits one free Gaussian whose mean is a bounded absolute position in a unit
cube. No (t,h,w) → 3D prior — this tests whether the latent alone carries the
geometry.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .decoder_a import LATENT_C, LATENT_H, LATENT_W, LATENT_T_DEFAULT
from .gaussian_head import GaussianHead, split_gaussian_params


class LearnedQueryDecoder(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        enc_depth: int = 6,
        dec_depth: int = 4,
        heads: int = 8,
        num_queries: int = 16384,
        latent_t: int = LATENT_T_DEFAULT,
        offset_max: float = 1.0,
        scale_max: float = 0.3,
    ):
        super().__init__()
        self.offset_max = offset_max
        self.scale_max = scale_max
        num_tokens = latent_t * LATENT_H * LATENT_W

        self.in_proj = nn.Linear(LATENT_C, dim)
        self.pos_emb = nn.Parameter(torch.zeros(num_tokens, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_depth)

        self.queries = nn.Parameter(torch.zeros(num_queries, dim))
        nn.init.normal_(self.queries, std=0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_depth)

        # K=1 per query: each query is exactly one Gaussian.
        self.head = GaussianHead(dim, k=1)

    def forward(self, latent: torch.Tensor) -> dict:
        """latent (B,128,T,24,16) -> dict of (B, num_queries, ...) Gaussian params."""
        b = latent.shape[0]
        ctx = latent.flatten(2).transpose(1, 2)
        ctx = self.in_proj(ctx) + self.pos_emb
        ctx = self.encoder(ctx)

        q = self.queries[None].expand(b, -1, -1)
        out = self.decoder(q, ctx)            # (B, num_queries, dim)
        raw = self.head(out)                  # (B, num_queries, 1, 14)

        params = split_gaussian_params(
            raw, canonical_mean=None, offset_max=self.offset_max, scale_max=self.scale_max
        )
        return {k_: v.reshape(b, -1, v.shape[-1]) for k_, v in params.items()}
