"""Method D — pixel-aligned decoder with GS-LRM activation shifts (anti-collapse).

Method C had the right geometry (pixel-aligned depth) but collapsed: at init its
Gaussians rendered worse than a blank frame, so the first gradient drove
opacity/scale to zero through the shared output head, into a zero-gradient
absorbing state (StableGS arXiv:2503.18458 formally diagnoses this equilibrium).

Method D keeps C's shared head + single optimizer (GS-LRM and Lyra both do) and
fixes the collapse purely with **GS-LRM's activation shifts** (arXiv:2404.19702,
the one foundational method that, like us, regresses opacity from a free shared
head — Lyra's code uses the same recipe):

  opacity = sigmoid(raw - 2.0)            -> ~0.12 at init: visible, gradient-live
  scale   = min(exp(raw - 2.3), cap=0.3)  -> ~0.10 at init: visible, bounded
  rgb     = 0.5 * tanh(raw) + 0.5         -> 0.5 mid-gray at init (init render not noise)
  depth   = (1-w)*d_near + w*d_far, w=sigmoid(raw)   -> pixel-aligned ray means
  quat    = normalize(raw)
  linear weights init N(0, 0.02)

The shifts put opacity/scale at moderate nonzero init values, so "blank beats
prediction -> kill opacity" never arises. Means stay pixel-aligned (the
probe-verified good init geometry). Count is decoupled from the latent grid via
k candidates per token: 24*16 tokens * k=27 = 10,368 Gaussians.

See docs memory: project-literature-landscape, project-collapse-root-cause.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_a import LATENT_C, LATENT_H, LATENT_W
from .decoder_c import ray_dirs_world

# Per-Gaussian raw channels: depth(1) + quat(4) + log-scale(3) + opacity(1) + rgb(3) = 12
PERGAUSS_DIM = 12


class PixelAlignedDecoderD(nn.Module):
    """Pixel-aligned-depth decoder with GS-LRM activation shifts.

    forward(latent, ref_K, ref_c2w) -> dict of (B, N, ...) params,
    N = LATENT_H * LATENT_W * k. Reference camera (frame 0 K + OpenGL c2w)
    supplies the per-pixel rays for unprojection.
    """

    def __init__(
        self,
        dim: int = 768,
        depth_layers: int = 12,
        heads: int = 12,
        k: int = 27,                  # 24*16*27 = 10,368 Gaussians
        radius: float = 2.0,          # measured from data at train time
        depth_halfrange: float = 1.3, # near/far = radius ± this (animals: ~[0.7, 3.3])
        scale_cap: float = 0.3,       # GS-LRM cap; stops long-line degenerates
        opacity_shift: float = 2.0,   # sigmoid(raw - shift): init opacity ~0.12
        logscale_shift: float = 2.3,  # min(exp(raw - shift), cap): init scale ~0.10
    ):
        super().__init__()
        self.k = k
        self.radius = radius
        self.d_near = max(radius - depth_halfrange, 1e-2)
        self.d_far = radius + depth_halfrange
        self.scale_cap = scale_cap
        self.opacity_shift = opacity_shift
        self.logscale_shift = logscale_shift

        self.in_proj = nn.Linear(LATENT_C, dim)
        self.pos_emb = nn.Parameter(torch.zeros(LATENT_H * LATENT_W, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth_layers)
        self.head = nn.Linear(dim, k * PERGAUSS_DIM)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        # GS-LRM init: N(0, 0.02) weights, zero bias. The activation SHIFTS
        # (subtracted constants in forward) do the visible-but-modest init,
        # NOT bias offsets — so raw outputs start ~0 and the shifts dominate.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, latent: torch.Tensor, ref_K: torch.Tensor, ref_c2w: torch.Tensor) -> dict:
        b = latent.shape[0]
        x = latent.mean(dim=2)                  # collapse T -> (B,C,H,W)
        x = x.flatten(2).transpose(1, 2)        # (B, H*W, C)
        x = self.in_proj(x) + self.pos_emb
        x = self.encoder(x)
        raw = self.head(x).unflatten(-1, (self.k, PERGAUSS_DIM))  # (B, HW, k, 12)

        dirs = ray_dirs_world(ref_K, ref_c2w, LATENT_H, LATENT_W)  # (HW, 3)
        cam_center = ref_c2w[:3, 3]                                # (3,)
        return self._assemble(raw, dirs, cam_center, b)

    def _assemble(self, raw, dirs, cam_center, b) -> dict:
        # channel layout: [0]=depth, [1:5]=quat, [5:8]=log-scale, [8]=opacity, [9:12]=rgb
        w = torch.sigmoid(raw[..., 0:1])                          # (B,HW,k,1) in (0,1)
        depth = (1 - w) * self.d_near + w * self.d_far            # pixel-aligned ray depth
        d = dirs[None, :, None, :]                                # (1,HW,1,3)
        mean = cam_center[None, None, None, :] + depth * d        # (B,HW,k,3)

        quat = F.normalize(raw[..., 1:5], dim=-1)
        scale = torch.clamp(torch.exp(raw[..., 5:8] - self.logscale_shift), max=self.scale_cap)
        opacity = torch.sigmoid(raw[..., 8:9] - self.opacity_shift)
        rgb = 0.5 * torch.tanh(raw[..., 9:12]) + 0.5

        params = {"mean": mean, "quat": quat, "scale": scale, "opacity": opacity, "rgb": rgb}
        return {kk: v.reshape(b, -1, v.shape[-1]) for kk, v in params.items()}
