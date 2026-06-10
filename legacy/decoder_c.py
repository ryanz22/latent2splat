"""Approach C — pixel-aligned depth decoder (Wonderland / Splatter-Image style).

The decoders A/B regressed FREE 3D means and never localized: 49k large
Gaussians diffused into full-frame fog because nothing tied them to surfaces
(per the project debug log).

The literature (Wonderland, GGS, Splatter-Image, pixelSplat) avoids this by
construction: a Gaussian is **pixel-aligned with a predicted depth** and its
3D position is computed by unprojecting its pixel along the camera ray:

    μ = cam_center + depth · ray_dir(u, v)

so a Gaussian can only move ALONG its ray. That single constraint forces
localization onto plausible surfaces.

Method A (this file, single-reference): treat the latent's (H=24, W=16)
spatial grid as pixel-aligned to ONE reference orbit frame (frame 0). Collapse
the temporal axis, predict per-latent-pixel depth + Gaussian params, unproject
through frame-0's camera. Known limitation: frame 0 only sees the front, so
back-facing surface is unconstrained — that is the motivation for Method B
(per-temporal-slot cameras), a later file.

Depth range is data-derived: all cameras sit at radius 0.95, fov_y 49.1°, so a
frame-filling object spans depth ~[0.52, 1.38]; we bound to radius ± 0.6.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .decoder_a import LATENT_C, LATENT_H, LATENT_W
from .gaussian_head import GAUSSIAN_DIM, split_gaussian_params

# Per-pixel raw output: depth(1) + the rest of GAUSSIAN_DIM EXCEPT the 3 mean
# channels (mean is computed by unprojection, not regressed) = 1 + (14-3) = 12.
PERPIX_DIM = 1 + (GAUSSIAN_DIM - 3)


def ray_dirs_world(K: torch.Tensor, c2w_opengl: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Per-pixel world-space ray directions for an OpenGL camera.

    K: (3,3) intrinsics for the FULL-RES image. h,w: the latent grid size
    (we sample pixel centers of the h×w grid mapped onto the full image).
    Returns (h*w, 3) unit ray directions in world space, and the implied
    pixel centers are the grid-cell centers of the full image.

    OpenGL camera: looks down -Z, +Y up. Camera-space ray for pixel (u,v):
        d_cam = ((u-cx)/fx, -(v-cy)/fy, -1)
    rotate by the c2w rotation (no translation) to get world direction.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    full_w, full_h = 2 * cx, 2 * cy  # principal point is image center here
    # pixel centers of an h×w grid spanning the full image
    us = (torch.arange(w, dtype=K.dtype, device=K.device) + 0.5) * (full_w / w)
    vs = (torch.arange(h, dtype=K.dtype, device=K.device) + 0.5) * (full_h / h)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")  # (h,w)
    d_cam = torch.stack([(uu - cx) / fx, -(vv - cy) / fy, -torch.ones_like(uu)], dim=-1)
    d_cam = d_cam.reshape(-1, 3)                       # (h*w, 3)
    R = c2w_opengl[:3, :3]
    d_world = d_cam @ R.T                              # rotate cam->world
    return d_world / d_world.norm(dim=-1, keepdim=True)


class PixelAlignedDecoder(nn.Module):
    """Method A: pixel-aligned-depth Gaussians anchored to a reference frame.

    forward(latent, ref_K, ref_c2w) -> dict of (B, N, ...) Gaussian params,
    N = LATENT_H * LATENT_W * k. The caller supplies the reference camera
    (frame 0's intrinsics + OpenGL c2w) so the geometry stays explicit.
    """

    def __init__(
        self,
        dim: int = 512,
        depth_layers: int = 6,
        heads: int = 8,
        k: int = 2,
        radius: float = 2.0,         # animals_v1 camera radius (was 0.95 for objaverse100_v1)
        depth_halfrange: float = 1.1,  # object is a unit sphere at radius 2 -> depth ~[1,3]
        scale_max: float = 0.05,
    ):
        super().__init__()
        self.k = k
        self.radius = radius
        self.depth_halfrange = depth_halfrange
        self.scale_max = scale_max

        self.in_proj = nn.Linear(LATENT_C, dim)
        self.pos_emb = nn.Parameter(torch.zeros(LATENT_H * LATENT_W, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth_layers)
        # per-pixel head: k * (depth + 11 non-mean gaussian channels)
        self.head = nn.Linear(dim, k * PERPIX_DIM)
        # opacity-logit bias positive so Gaussians start visible
        with torch.no_grad():
            self.head.bias.view(k, PERPIX_DIM)[:, 8] = 2.0  # opacity-logit slot (see _assemble)

    def forward(self, latent: torch.Tensor, ref_K: torch.Tensor, ref_c2w: torch.Tensor) -> dict:
        b = latent.shape[0]
        # collapse temporal axis (mean over T), keep spatial grid -> (B, H*W, C)
        x = latent.mean(dim=2)                       # (B,C,H,W)
        x = x.flatten(2).transpose(1, 2)             # (B, H*W, C)
        x = self.in_proj(x) + self.pos_emb
        x = self.encoder(x)
        raw = self.head(x).unflatten(-1, (self.k, PERPIX_DIM))  # (B, H*W, k, 12)

        # ray dirs for the H×W grid (shared across batch & k)
        dirs = ray_dirs_world(ref_K, ref_c2w, LATENT_H, LATENT_W)  # (H*W, 3)
        cam_center = ref_c2w[:3, 3]                                # (3,)

        return self._assemble(raw, dirs, cam_center, b)

    def _assemble(self, raw, dirs, cam_center, b) -> dict:
        # raw layout per pixel/k: [0]=depth, [1:5]=quat, [5:8]=log? scale, [8]=opacity, [9:12]=rgb
        depth = self.radius + torch.tanh(raw[..., 0:1]) * self.depth_halfrange  # (B,HW,k,1)
        # mean = cam_center + depth * ray_dir ; dirs broadcast over B and k
        d = dirs[None, :, None, :]                       # (1,HW,1,3)
        mean = cam_center[None, None, None, :] + depth * d  # (B,HW,k,3)

        # reuse the shared activations for the non-mean channels by packing a
        # 14-wide tensor whose first 3 (mean offset) are zero and canonical=mean.
        rest = raw[..., 1:]                              # (B,HW,k,11) = quat4+scale3+op1+rgb3
        packed = torch.cat([torch.zeros_like(mean), rest], dim=-1)  # (B,HW,k,14)
        params = split_gaussian_params(
            packed, canonical_mean=mean, offset_max=0.0, scale_max=self.scale_max
        )
        return {kk: v.reshape(b, -1, v.shape[-1]) for kk, v in params.items()}
