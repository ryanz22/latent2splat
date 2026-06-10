"""Shared Gaussian-parameter head + activations for both decoders.

A Gaussian primitive is (mean μ∈ℝ³, quaternion q∈ℝ⁴, scale s∈ℝ³, opacity
α∈[0,1], color c∈ℝ³). v1 uses SH degree 0 (plain RGB).

Parameterization follows canonical 3DGS / gsplat: **scale in log-space
(`exp`) and opacity in logit-space (`sigmoid`)**, with NO saturating clamps.
The earlier `clamp(softplus(scale))` + `tanh(mean)` created flat dead-gradient
zones; combined with no incentive to keep Gaussians visible, opacity collapsed
to 0 (an invisible Gaussian has exactly-zero gradient — an unrecoverable
local min). The head's opacity-logit bias is initialized POSITIVE so every
Gaussian starts visible and the optimizer must work to remove it, not the
reverse.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 3 (mean/offset) + 4 (quat) + 3 (log-scale) + 1 (opacity-logit) + 3 (rgb) = 14
GAUSSIAN_DIM = 14

# Channel layout in the raw head output.
_MEAN, _QUAT, _LOGSCALE, _OPLOGIT, _RGB = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 11), slice(11, 14)


def split_gaussian_params(
    raw: torch.Tensor,
    canonical_mean: torch.Tensor | None = None,
    offset_max: float = 1.0,
    scale_max: float = 0.3,
) -> dict:
    """Map raw head outputs (...,14) to activated Gaussian params.

    - mean: bounded residual to canonical_mean (token-aligned) or bounded
      absolute position (learned-query). tanh*offset_max, generous (1.0).
    - scale: `scale_max * sigmoid(raw)` — BOUNDED in (0, scale_max). A raw
      `exp()` (the 3DGS convention) blew up to scale ~960 in a single bad
      step here because we use a single global LR rather than 3DGS's
      per-group LRs; the bounded sigmoid form cannot explode and keeps the
      gradient alive everywhere in between. See design doc §11.
    - opacity: `sigmoid(raw)` in logit-space; the head biases this positive.
    """
    offset = torch.tanh(raw[..., _MEAN]) * offset_max
    mean = offset if canonical_mean is None else canonical_mean + offset
    quat = F.normalize(raw[..., _QUAT], dim=-1)
    scale = scale_max * torch.sigmoid(raw[..., _LOGSCALE])
    opacity = torch.sigmoid(raw[..., _OPLOGIT])
    rgb = torch.sigmoid(raw[..., _RGB])
    return {"mean": mean, "quat": quat, "scale": scale, "opacity": opacity, "rgb": rgb}


class GaussianHead(nn.Module):
    """Linear head: hidden -> K Gaussians worth of raw params.

    `opacity_bias` initializes the opacity-logit channel so Gaussians start
    visible (default +2.0 → sigmoid≈0.88), breaking the opacity-death spiral.
    """

    def __init__(self, hidden: int, k: int = 1, opacity_bias: float = 2.0):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(hidden, k * GAUSSIAN_DIM)
        with torch.no_grad():
            self.proj.bias.view(k, GAUSSIAN_DIM)[:, _OPLOGIT] = opacity_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., hidden) -> (..., K, 14)
        out = self.proj(x)
        return out.unflatten(-1, (self.k, GAUSSIAN_DIM))
