"""Method E — transformer encoder + transposed-conv upsampler + per-pixel
Gaussians. This is the verified latent-input recipe (Lyra / Wonderland), the
closest proven practice to our exact setup (a small frozen video-VAE latent →
many pixel-aligned Gaussians).

Why this backbone (research-verified):
- Lyra & Wonderland — the only methods that, like us, decode from a compressed
  video-VAE latent — run a transformer over the token grid then UPSAMPLE with a
  transposed-3D-conv and predict ONE Gaussian per upsampled pixel. Methods C/D
  predicted K Gaussians per latent token with no spatial coupling between them;
  the deconv upsampler gives neighboring Gaussians shared, smoothly-varying
  features (spatial coherence) — the piece C/D lacked.
- Mamba (Long-LRM/Lyra) is only for ~250k-token sequences; at our 384 tokens a
  pure transformer is sufficient, so we drop it.

Opacity — two variants (set by `opacity_mode`), both keep the rest identical:
- "pdf" (default, NO depth data needed): opacity = probability the network
  assigns to the Gaussian's depth bin (pixelSplat). Per pixel the head emits a
  softmax over K disparity bins; opacity = selected-bin prob. This couples
  "place a surface" and "be visible" into ONE parameter, so opacity cannot be
  driven to zero independently of geometry — the structural cure for the
  free-opacity collapse that killed C/D.
- "sigmoid" (for later, WITH depth supervision): opacity = sigmoid(raw - 2.0),
  Lyra's recipe — Lyra avoids collapse here via depth supervision, which we'll
  add when depth maps arrive.

Other activations are GS-LRM/Lyra: scale=min(exp(raw-2.3),0.3), rgb=0.5tanh+0.5,
quat=normalize. Means are pixel-aligned: unproject the ref-frame ray at the
predicted depth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_a import LATENT_C, LATENT_H, LATENT_W
from .decoder_c import ray_dirs_world


class PixelAlignedDecoderE(nn.Module):
    """forward(latent, ref_K, ref_c2w) -> dict of (B, N, ...) Gaussian params,
    N = (LATENT_H*up) * (LATENT_W*up). Reference camera supplies per-pixel rays
    at the UPSAMPLED resolution."""

    def __init__(
        self,
        dim: int = 512,
        depth_layers: int = 12,
        heads: int = 8,
        up: int = 5,                  # spatial upsample factor: 24x16 -> 120x80 = 9,600
        radius: float = 2.0,
        depth_halfrange: float = 1.3,
        scale_cap: float = 0.3,
        scale_floor: float = 0.015,   # min scale: stops the scale-death spiral
                                      # (a few px at radius 2). Analog of the
                                      # opacity fix — remove the zeroable channel.
        opacity_mode: str = "pdf",    # "pdf" (no depth needed) | "sigmoid" (needs depth sup.)
        opacity_shift: float = 2.0,   # used only in "sigmoid" mode
        num_depth_bins: int = 32,     # used only in "pdf" mode (pixelSplat: 32)
    ):
        super().__init__()
        assert opacity_mode in ("pdf", "sigmoid")
        self.up = up
        self.up_h, self.up_w = LATENT_H * up, LATENT_W * up
        self.radius = radius
        self.d_near = max(radius - depth_halfrange, 1e-2)
        self.d_far = radius + depth_halfrange
        self.scale_cap = scale_cap
        self.scale_floor = scale_floor
        self.opacity_mode = opacity_mode
        self.opacity_shift = opacity_shift
        self.K = num_depth_bins

        self.in_proj = nn.Linear(LATENT_C, dim)
        self.pos_emb = nn.Parameter(torch.zeros(LATENT_H * LATENT_W, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth_layers)

        # Transposed-conv spatial upsampler: (dim,24,16) -> (dim, 24*up, 16*up).
        # kernel=stride=up, padding=0 → output = in*up exactly (no off-by-one).
        self.deconv = nn.ConvTranspose2d(dim, dim, kernel_size=up, stride=up)
        self.act = nn.GELU()

        # Per-pixel Gaussian head. Channels:
        #   pdf mode:    depth-bin logits(K) + scale(3) + quat(4) + rgb(3)
        #   sigmoid mode: depth(1) + scale(3) + quat(4) + opacity(1) + rgb(3)
        if opacity_mode == "pdf":
            self.out_ch = self.K + 3 + 4 + 3
        else:
            self.out_ch = 1 + 3 + 4 + 1 + 3
        self.head = nn.Conv2d(dim, self.out_ch, kernel_size=1)
        self.apply(self._init_weights)

        # disparity bin centers in [1/d_far, 1/d_near] (uniform in disparity,
        # like pixelSplat); depth = 1/disparity.
        if opacity_mode == "pdf":
            disp = torch.linspace(1.0 / self.d_far, 1.0 / self.d_near, self.K)
            self.register_buffer("bin_depths", 1.0 / disp, persistent=False)  # (K,)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, latent: torch.Tensor, ref_K: torch.Tensor, ref_c2w: torch.Tensor) -> dict:
        b = latent.shape[0]
        x = latent.mean(dim=2)                         # collapse T -> (B,C,H,W)
        x = x.flatten(2).transpose(1, 2)               # (B, HW, C)
        x = self.in_proj(x) + self.pos_emb
        x = self.encoder(x)                            # (B, HW, dim)
        # back to image grid for the conv upsampler
        x = x.transpose(1, 2).reshape(b, -1, LATENT_H, LATENT_W)   # (B,dim,24,16)
        x = self.act(self.deconv(x))                   # (B,dim,120,80)
        raw = self.head(x)                             # (B,out_ch,120,80)
        # to (B, N, out_ch), N = 120*80
        raw = raw.flatten(2).transpose(1, 2)
        return self._assemble(raw, ref_K, ref_c2w, b)

    def _scale(self, raw_scale: torch.Tensor) -> torch.Tensor:
        # Bounded in [floor, cap] via sigmoid — can't vanish (floor) OR explode
        # (cap), and gradient-live across the whole range. The floor is what
        # stops the scale-death spiral that blanked Method E's first version.
        return self.scale_floor + (self.scale_cap - self.scale_floor) * torch.sigmoid(raw_scale)

    def _assemble(self, raw, ref_K, ref_c2w, b) -> dict:
        dirs = ray_dirs_world(ref_K, ref_c2w, self.up_h, self.up_w)  # (N,3)
        cam_center = ref_c2w[:3, 3]

        if self.opacity_mode == "pdf":
            logits = raw[..., : self.K]                              # (B,N,K)
            pdf = torch.softmax(logits, dim=-1)
            # expected depth over bins (differentiable, smooth); opacity = max prob
            depth = (pdf * self.bin_depths.to(pdf.dtype)).sum(-1, keepdim=True)  # (B,N,1)
            opacity = pdf.max(dim=-1, keepdim=True).values           # (B,N,1) in (0,1)
            rest = raw[..., self.K:]                                 # scale3+quat4+rgb3
            scale = self._scale(rest[..., 0:3])
            quat = F.normalize(rest[..., 3:7], dim=-1)
            rgb = 0.5 * torch.tanh(rest[..., 7:10]) + 0.5
        else:  # sigmoid
            w = torch.sigmoid(raw[..., 0:1])
            depth = (1 - w) * self.d_near + w * self.d_far
            scale = self._scale(raw[..., 1:4])
            quat = F.normalize(raw[..., 4:8], dim=-1)
            opacity = torch.sigmoid(raw[..., 8:9] - self.opacity_shift)
            rgb = 0.5 * torch.tanh(raw[..., 9:12]) + 0.5

        mean = cam_center[None, None, :] + depth * dirs[None, :, :]  # (B,N,3)
        params = {"mean": mean, "quat": quat, "scale": scale, "opacity": opacity,
                  "rgb": rgb, "depth": depth}  # depth (B,N,1): predicted ray distance, for supervision
        return {kk: v.reshape(b, -1, v.shape[-1]) for kk, v in params.items()}
