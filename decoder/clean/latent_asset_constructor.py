"""Latent-conditioned source-view predictors for the next 3DGS decoder path.

This module is intentionally independent from the current RGBD fusion training
loop.  It gives the project a concrete component for the next architecture:

    LTX orbit latent + requested source cameras
      -> per-source RGB/depth/mask/confidence/features
      -> learned 3D fusion / Gaussian heads

The current production-quality path can use DA3/LTX decoded RGB as teachers
while this predictor learns to replace them.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def camera_token_features(
    K: torch.Tensor,
    c2w: torch.Tensor,
    radius: torch.Tensor | float,
) -> torch.Tensor:
    """Compact camera token features for source-view conditioning.

    Args:
        K: ``(B,V,3,3)`` intrinsics.
        c2w: ``(B,V,4,4)`` OpenGL camera-to-world matrices.
        radius: scalar or ``(B,)`` object radius.

    Returns:
        ``(B,V,16)`` features containing normalized intrinsics, camera center,
        forward/up axes, and distance.  These are inference-available for a
        planned orbit and do not use target-view ground truth.
    """
    if K.ndim != 4 or K.shape[-2:] != (3, 3):
        raise ValueError("K must have shape (B,V,3,3)")
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError("c2w must have shape (B,V,4,4)")
    if K.shape[:2] != c2w.shape[:2]:
        raise ValueError("K and c2w must agree on batch/view dimensions")

    dtype, device = K.dtype, K.device
    if not torch.is_tensor(radius):
        radius_t = torch.full((K.shape[0],), float(radius), dtype=dtype, device=device)
    else:
        radius_t = radius.to(device=device, dtype=dtype).reshape(K.shape[0])
    radius_t = radius_t.clamp_min(1e-6)[:, None, None]

    fx = K[..., 0, 0:1]
    fy = K[..., 1, 1:2]
    cx = K[..., 0, 2:3]
    cy = K[..., 1, 2:3]
    intr = torch.cat([fx, fy, cx, cy], dim=-1)
    intr = torch.log1p(intr.clamp_min(0.0)) / 10.0

    center_raw = c2w[..., :3, 3] / radius_t
    # OpenGL cameras look down local -Z.  Keep both forward and up explicit so
    # the predictor can distinguish orbit pose, elevation, and roll.
    forward = -c2w[..., :3, 2]
    up = c2w[..., :3, 1]
    dist = center_raw.norm(dim=-1, keepdim=True).clamp(0.0, 8.0) / 8.0
    az = torch.atan2(center_raw[..., 0:1], center_raw[..., 2:3].clamp_min(1e-6))
    az_feat = torch.cat([torch.sin(az), torch.cos(az)], dim=-1)
    center = center_raw.clamp(-4.0, 4.0) / 4.0
    return torch.cat([intr, center, forward, up, dist, az_feat], dim=-1)


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(16, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class LatentSourcePredictor(nn.Module):
    """Predict source-view evidence from an LTX orbit latent and cameras.

    The predictor emits per-view maps at a configurable scale relative to the
    latent spatial grid.  It is meant to be supervised first against source RGB,
    masks, and depth, then plugged into the existing fusion renderer.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        latent_t: int = 2,
        hidden: int = 128,
        out_scale: int = 4,
        depth_bins: int = 64,
        feature_channels: int = 32,
        blocks: int = 3,
    ):
        super().__init__()
        if out_scale < 1 or out_scale & (out_scale - 1):
            raise ValueError("out_scale must be a positive power of two")
        self.latent_t = int(latent_t)
        self.hidden = int(hidden)
        self.out_scale = int(out_scale)
        self.depth_bins = int(depth_bins)
        self.feature_channels = int(feature_channels)

        self.latent_proj = nn.Conv3d(latent_channels, hidden, kernel_size=1)
        self.temporal_merge = nn.Conv2d(hidden * self.latent_t, hidden, kernel_size=1)
        self.encoder = nn.Sequential(*[_ResBlock(hidden) for _ in range(max(blocks, 1))])
        self.camera_mlp = nn.Sequential(
            nn.Linear(16, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        ups = []
        for _ in range(self.out_scale.bit_length() - 1):
            ups.extend([
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(min(16, hidden), hidden),
                nn.GELU(),
            ])
        self.upsampler = nn.Sequential(*ups)
        out_channels = 3 + 1 + self.depth_bins + 1 + self.feature_channels
        self.head = nn.Conv2d(hidden, out_channels, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        latent: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        radius: torch.Tensor | float,
    ) -> dict[str, torch.Tensor]:
        """Run the source predictor.

        Args:
            latent: ``(B,128,T,H,W)`` LTX latent.
            K: ``(B,V,3,3)`` source-view intrinsics.
            c2w: ``(B,V,4,4)`` source-view camera-to-world matrices.
            radius: scalar or ``(B,)`` radius.

        Returns:
            Dict with shapes:
              - ``rgb_delta``: ``(B,V,3,Hs,Ws)``
              - ``mask_logit``: ``(B,V,1,Hs,Ws)``
              - ``depth_logits``: ``(B,V,D,Hs,Ws)``
              - ``confidence_logit``: ``(B,V,1,Hs,Ws)``
              - ``features``: ``(B,V,F,Hs,Ws)``
        """
        if latent.ndim != 5:
            raise ValueError("latent must have shape (B,C,T,H,W)")
        b, _, t, h, w = latent.shape
        if t != self.latent_t:
            raise ValueError(f"expected latent T={self.latent_t}, got {t}")
        if K.shape[0] != b or c2w.shape[0] != b:
            raise ValueError("latent, K, and c2w must agree on batch size")
        v = K.shape[1]

        x = self.latent_proj(latent)
        x = x.reshape(b, self.hidden * self.latent_t, h, w)
        x = self.encoder(self.temporal_merge(x))

        cam = camera_token_features(K, c2w, radius)
        film = self.camera_mlp(cam).view(b, v, self.hidden * 2, 1, 1)
        gamma, beta = film.chunk(2, dim=2)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        x_v = x[:, None].expand(-1, v, -1, -1, -1)
        x_v = (x_v * gamma + beta).reshape(b * v, self.hidden, h, w)
        y = self.head(self.upsampler(x_v))
        hs, ws = y.shape[-2:]
        y = y.view(b, v, y.shape[1], hs, ws)

        i = 0
        rgb_delta = torch.tanh(y[:, :, i:i + 3])
        i += 3
        mask_logit = y[:, :, i:i + 1]
        i += 1
        depth_logits = y[:, :, i:i + self.depth_bins]
        i += self.depth_bins
        confidence_logit = y[:, :, i:i + 1]
        i += 1
        features = y[:, :, i:i + self.feature_channels]
        return {
            "rgb_delta": rgb_delta,
            "mask_logit": mask_logit,
            "depth_logits": depth_logits,
            "confidence_logit": confidence_logit,
            "features": features,
        }
