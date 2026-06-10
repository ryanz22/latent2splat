"""Clean-slate latent→3DGS decoder (single-object instantiation of the
Wonderland/Lyra recipe + GS-LRM parameterization). K=1 reference-view anchoring.

Shape trace (B omitted):
  z (128,2,24,16) → 768 tokens(128) → +linear(768)+posemb+Plücker → ViT×L
  → reshape (2,24,16,768) → concat T → 1×1 → (768,24,16)
  → upsample ×4 → (64,384,256) → 1×1 head → (12,384,256) → 98,304 Gaussians.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder.clean.geometry import ray_dirs_world
from .geometry import depth_bounds, plucker_embedding
from .gaussians import activate, B_ALPHA, B_SCALE, SCALE_CAP_FRAC

LATENT_C, T, LH, LW = 128, 2, 24, 16
# ups_stages stride-2 stages upsample the (LH,LW) grid by 2**ups_stages:
#   4 → 384×256 = 98,304 Gaussians (~0.25/render-px);  5 → 768×512 = 393,216 (~1/px)


class CleanGSDecoder(nn.Module):
    def __init__(self, dim: int = 768, depth: int = 12, heads: int = 16,
                 half_frac: float = 0.5, scale_cap_frac: float = SCALE_CAP_FRAC,
                 ups_stages: int = 4, mean_offset_frac: float = 0.0,
                 upsample_mode: str = "deconv", latent_skip: bool = False,
                 coord_inject: bool = False, coord_fourier: int = 4,
                 image_cond_channels: int = 0, image_head_skip: bool = False,
                 image_opacity_fg: float = 0.2, image_opacity_bg: float = 0.001,
                 image_residual_scale: float = 0.1, image_scale_frac: float = 0.0,
                 image_geom_residual_scale: float = 1.0,
                 image_rgb_residual_scale: float | None = None,
                 image_opacity_residual_scale: float | None = None,
                 explicit_depth_head: bool = False,
                 explicit_visibility_head: bool = False,
                 image_depth_prior_frac: float = 0.0,
                 image_depth_skip: bool = False,
                 image_depth_residual_scale: float = 0.0,
                 zero_init_head: bool = False,
                 image_visibility_skip: bool = False,
                 image_normal_scale_frac: float = 0.0,
                 image_boundary_scale_mult: float = 1.0,
                 image_boundary_width: int = 0,
                 image_camera_quat: bool = False,
                 image_normal_quat: bool = False,
                 depth_head_scale: float = 1.0,
                 visibility_head_scale: float = 1.0,
                 latent_t: int = T,
                 latent_h: int = LH,
                 latent_w: int = LW):
        super().__init__()
        if upsample_mode not in {"deconv", "resize"}:
            raise ValueError("upsample_mode must be 'deconv' or 'resize'")
        self.latent_t = int(latent_t)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        if self.latent_t <= 0 or self.latent_h <= 0 or self.latent_w <= 0:
            raise ValueError("latent_t, latent_h, and latent_w must be positive")
        self.dim, self.half_frac, self.scale_cap_frac = dim, half_frac, scale_cap_frac
        self.mean_offset_frac = mean_offset_frac
        self.upsample_mode = upsample_mode
        self.latent_skip = latent_skip
        self.coord_inject = coord_inject
        self.coord_fourier = coord_fourier
        self.image_cond_channels = image_cond_channels
        self.image_head_skip = image_head_skip
        self.image_opacity_fg = image_opacity_fg
        self.image_opacity_bg = image_opacity_bg
        self.image_residual_scale = image_residual_scale
        self.image_rgb_residual_scale = (
            image_residual_scale if image_rgb_residual_scale is None else image_rgb_residual_scale
        )
        self.image_opacity_residual_scale = (
            image_residual_scale if image_opacity_residual_scale is None else image_opacity_residual_scale
        )
        self.image_scale_frac = image_scale_frac
        self.image_geom_residual_scale = image_geom_residual_scale
        self.explicit_depth_head = explicit_depth_head
        self.explicit_visibility_head = explicit_visibility_head
        self.image_depth_prior_frac = image_depth_prior_frac
        self.image_depth_skip = image_depth_skip
        self.image_depth_residual_scale = image_depth_residual_scale
        self.zero_init_head = zero_init_head
        self.image_visibility_skip = image_visibility_skip
        self.image_normal_scale_frac = image_normal_scale_frac
        self.image_boundary_scale_mult = image_boundary_scale_mult
        self.image_boundary_width = image_boundary_width
        self.image_camera_quat = image_camera_quat
        self.image_normal_quat = image_normal_quat
        self.depth_head_scale = depth_head_scale
        self.visibility_head_scale = visibility_head_scale
        self.map_h = self.latent_h * 2 ** ups_stages
        self.map_w = self.latent_w * 2 ** ups_stages
        self.out_channels = 15 if mean_offset_frac > 0 else 12
        self.in_proj = nn.Linear(LATENT_C, dim)
        self.pos_emb = nn.Parameter(torch.zeros(self.latent_t * self.latent_h * self.latent_w, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.plucker_proj = nn.Linear(6, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True, activation="gelu")
        self.trunk = nn.TransformerEncoder(layer, num_layers=depth)
        self.temporal = nn.Conv2d(self.latent_t * dim, dim, kernel_size=1)
        # channel schedule: taper dim→256→128→64, then hold at 64 for extra stages
        taper = [256, 128, 64]
        chans = [dim] + taper[:ups_stages] + [64] * max(0, ups_stages - len(taper))
        ups = []
        skip_projs = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            block = []
            if upsample_mode == "deconv":
                block.append(nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2))
            else:
                block += [nn.Upsample(scale_factor=2, mode="nearest"),
                          nn.Conv2d(cin, cout, kernel_size=3, padding=1)]
            block += [nn.GroupNorm(min(32, cout), cout), nn.GELU()]
            ups.append(nn.Sequential(*block))
            if latent_skip:
                skip_projs.append(nn.Sequential(
                    nn.Conv2d(dim, cout, kernel_size=1),
                    nn.GroupNorm(min(32, cout), cout),
                ))
            else:
                skip_projs.append(nn.Identity())
        self.upsampler = nn.ModuleList(ups)      # ×2**ups_stages
        self.skip_projs = nn.ModuleList(skip_projs)
        if latent_skip:
            self.skip_gates = nn.Parameter(torch.full((len(skip_projs),), -2.0))
        else:
            self.register_parameter("skip_gates", None)
        coord_ch = 8 + 4 * max(coord_fourier, 0)  # xy + Plucker6 + sin/cos for x/y
        if coord_inject:
            self.coord_proj = nn.Sequential(
                nn.Conv2d(coord_ch, chans[-1], kernel_size=1),
                nn.GroupNorm(min(32, chans[-1]), chans[-1]),
                nn.GELU(),
                nn.Conv2d(chans[-1], chans[-1], kernel_size=1),
            )
            self.coord_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.coord_proj = None
            self.register_parameter("coord_gate", None)
        if image_cond_channels > 0:
            self.image_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(image_cond_channels, cout, kernel_size=3, padding=1),
                    nn.GroupNorm(min(32, cout), cout),
                    nn.GELU(),
                    nn.Conv2d(cout, cout, kernel_size=1),
                )
                for cout in chans[1:]
            ])
            self.image_gates = nn.Parameter(torch.full((len(self.image_projs),), -1.0))
        else:
            self.image_projs = nn.ModuleList()
            self.register_parameter("image_gates", None)
        self.head = nn.Conv2d(chans[-1], self.out_channels, kernel_size=1)
        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.depth_head = self._make_scalar_head(chans[-1]) if explicit_depth_head else None
        self.visibility_head = self._make_scalar_head(chans[-1]) if explicit_visibility_head else None

    @staticmethod
    def _make_scalar_head(channels: int) -> nn.Sequential:
        mid = max(channels // 2, 32)
        head = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, mid), mid),
            nn.GELU(),
            nn.Conv2d(mid, 1, kernel_size=1),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @staticmethod
    def _matrix_to_quat_wxyz(rot: torch.Tensor) -> torch.Tensor:
        m = rot
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if float(tr.detach().cpu()) > 0.0:
            s = torch.sqrt((tr + 1.0).clamp_min(1e-8)) * 2.0
            q = torch.stack([
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ])
        elif bool(((m[0, 0] > m[1, 1]) & (m[0, 0] > m[2, 2])).detach().cpu()):
            s = torch.sqrt((1.0 + m[0, 0] - m[1, 1] - m[2, 2]).clamp_min(1e-8)) * 2.0
            q = torch.stack([
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ])
        elif bool((m[1, 1] > m[2, 2]).detach().cpu()):
            s = torch.sqrt((1.0 + m[1, 1] - m[0, 0] - m[2, 2]).clamp_min(1e-8)) * 2.0
            q = torch.stack([
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ])
        else:
            s = torch.sqrt((1.0 + m[2, 2] - m[0, 0] - m[1, 1]).clamp_min(1e-8)) * 2.0
            q = torch.stack([
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ])
        q = F.normalize(q, dim=0)
        return torch.where(q[:1] < 0, -q, q)

    @staticmethod
    def _normal_map_to_quat_wxyz(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World normal map (B,3,H,W) -> quaternion map (B,4,H,W) + validity."""
        n_norm = normal.norm(dim=1, keepdim=True)
        valid = n_norm > 0.1
        n = normal / n_norm.clamp_min(1e-6)
        up_z = torch.zeros_like(n)
        up_z[:, 2:3] = 1.0
        up_y = torch.zeros_like(n)
        up_y[:, 1:2] = 1.0
        near_z = (n[:, 2:3].abs() > 0.95)
        up = torch.where(near_z, up_y, up_z)
        t1 = F.normalize(torch.cross(up, n, dim=1), dim=1)
        t2 = F.normalize(torch.cross(n, t1, dim=1), dim=1)
        # Rotation matrix columns are local x/y/z axes in world space.
        m00, m10, m20 = t1[:, 0:1], t1[:, 1:2], t1[:, 2:3]
        m01, m11, m21 = t2[:, 0:1], t2[:, 1:2], t2[:, 2:3]
        m02, m12, m22 = n[:, 0:1], n[:, 1:2], n[:, 2:3]
        qw = 0.5 * torch.sqrt((1.0 + m00 + m11 + m22).clamp_min(1e-8))
        qx = 0.5 * torch.copysign(torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-8)), m21 - m12)
        qy = 0.5 * torch.copysign(torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(1e-8)), m02 - m20)
        qz = 0.5 * torch.copysign(torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(1e-8)), m10 - m01)
        q = F.normalize(torch.cat([qw, qx, qy, qz], dim=1), dim=1)
        q = torch.where(q[:, :1] < 0, -q, q)
        return q, valid

    @staticmethod
    def _raw_for_soft_capped_scale(target: float, cap: float) -> float:
        target = min(max(target, 1e-8), cap * 0.95)
        raw_pos = target * cap / max(cap - target, 1e-8)
        return math.log(max(raw_pos, 1e-8)) - B_SCALE

    @staticmethod
    def _raw_for_soft_capped_scale_tensor(target: torch.Tensor, cap: float) -> torch.Tensor:
        target = target.clamp(1e-8, cap * 0.95)
        raw_pos = target * cap / (cap - target).clamp_min(1e-8)
        return torch.log(raw_pos.clamp_min(1e-8)) - B_SCALE

    def _coord_features(self, anchor_K: torch.Tensor, anchor_c2w: torch.Tensor,
                        dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        h, w = self.map_h, self.map_w
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        feats = [xx, yy]
        for i in range(max(self.coord_fourier, 0)):
            f = math.pi * (2 ** i)
            feats.extend([torch.sin(f * xx), torch.cos(f * xx),
                          torch.sin(f * yy), torch.cos(f * yy)])
        pl = plucker_embedding(anchor_K, anchor_c2w, h, w).to(device=device, dtype=dtype)
        pl = pl.reshape(h, w, 6).permute(2, 0, 1)
        return torch.cat([torch.stack(feats, 0), pl], 0)

    def forward(self, latent: torch.Tensor, anchor_K: torch.Tensor,
                anchor_c2w: torch.Tensor, radius: float,
                image_cond: torch.Tensor | None = None) -> dict:
        b = latent.shape[0]
        if latent.shape[1] != LATENT_C:
            raise ValueError(f"expected latent channels={LATENT_C}, got {latent.shape[1]}")
        if tuple(latent.shape[2:]) != (self.latent_t, self.latent_h, self.latent_w):
            raise ValueError(
                "latent grid mismatch: model expects "
                f"(T,H,W)=({self.latent_t},{self.latent_h},{self.latent_w}), "
                f"got {tuple(latent.shape[2:])}"
            )
        # tokens: (B, T*LH*LW, C)
        x = latent.permute(0, 2, 3, 4, 1).reshape(
            b, self.latent_t * self.latent_h * self.latent_w, LATENT_C
        )
        x = self.in_proj(x) + self.pos_emb
        # Plücker tokens for the (single) anchor view at latent resolution, tiled over T
        pl = plucker_embedding(anchor_K, anchor_c2w, self.latent_h, self.latent_w)
        pl = pl.to(x.dtype).repeat(self.latent_t, 1)
        x = x + self.plucker_proj(pl)[None]
        x = self.trunk(x)
        # reshape, concat the two temporal slices along channels
        x = x.reshape(
            b, self.latent_t, self.latent_h, self.latent_w, self.dim
        ).permute(0, 1, 4, 2, 3)
        x = x.reshape(b, self.latent_t * self.dim, self.latent_h, self.latent_w)
        x = self.temporal(x)
        base = x
        for i, (block, skip_proj) in enumerate(zip(self.upsampler, self.skip_projs)):
            x = block(x)
            if self.latent_skip:
                skip = F.interpolate(base, size=x.shape[-2:], mode="bilinear", align_corners=False)
                gate = torch.sigmoid(self.skip_gates[i])
                x = x + gate * skip_proj(skip)
            if self.image_cond_channels > 0:
                if image_cond is None:
                    raise ValueError("image_cond is required when image_cond_channels > 0")
                cond = F.interpolate(image_cond.to(dtype=x.dtype), size=x.shape[-2:],
                                     mode="bilinear", align_corners=False)
                x = x + torch.sigmoid(self.image_gates[i]) * self.image_projs[i](cond)
        if self.coord_inject:
            coord = self._coord_features(anchor_K, anchor_c2w, x.dtype, x.device)
            x = x + torch.sigmoid(self.coord_gate) * self.coord_proj(coord[None].expand(b, -1, -1, -1))
        head_raw = self.head(x)                                  # (B,C,map_h,map_w)
        raw_residual_l2 = {
            "_raw_residual_l2_rgb": head_raw[:, 0:3].float().pow(2).mean(dim=(1, 2, 3)),
            "_raw_residual_l2_geom": head_raw[:, 3:10].float().pow(2).mean(dim=(1, 2, 3)),
            "_raw_residual_l2_opacity": head_raw[:, 10:11].float().pow(2).mean(dim=(1, 2, 3)),
            "_raw_residual_l2_depth": head_raw[:, 11:12].float().pow(2).mean(dim=(1, 2, 3)),
        }
        if head_raw.shape[1] >= 15:
            raw_residual_l2["_raw_residual_l2_offset"] = (
                head_raw[:, 12:15].float().pow(2).mean(dim=(1, 2, 3))
            )
        # The skip path writes deterministic RGBD priors into raw channels in-place.
        # Keep the unmodified head output for direct residual regularization.
        raw = head_raw.clone()
        if self.image_head_skip:
            if image_cond is None:
                raise ValueError("image_cond is required when image_head_skip is enabled")
            cond = F.interpolate(image_cond.to(dtype=raw.dtype), size=raw.shape[-2:],
                                 mode="bilinear", align_corners=False)
            mask = cond[:, 3:4].clamp(0.0, 1.0)
            visibility = 1.0
            if self.image_visibility_skip:
                if image_cond.shape[1] < 7:
                    raise ValueError("image_visibility_skip expects image_cond channel 6 = visibility")
                visibility = cond[:, 6:7].clamp(0.0, 1.0)
            rgb = cond[:, :3] / mask.clamp_min(1e-4)
            rgb = rgb.clamp(1e-4, 1.0 - 1e-4)
            raw[:, 0:3] = self.image_rgb_residual_scale * raw[:, 0:3] + torch.logit(rgb) * mask
            op_t = self.image_opacity_bg + (self.image_opacity_fg - self.image_opacity_bg) * mask * visibility
            raw[:, 10:11] = (
                self.image_opacity_residual_scale * raw[:, 10:11]
                + torch.logit(op_t.clamp(1e-4, 1.0 - 1e-4)) - B_ALPHA
            )
            if self.image_scale_frac > 0:
                cap = self.scale_cap_frac * radius
                z_frac = self.image_normal_scale_frac or self.image_scale_frac
                if self.image_boundary_width > 0 and self.image_boundary_scale_mult < 1.0:
                    bw = max(int(self.image_boundary_width), 1)
                    eroded = 1.0 - F.max_pool2d(
                        1.0 - mask, kernel_size=2 * bw + 1, stride=1, padding=bw
                    )
                    eroded = eroded.clamp(0.0, 1.0)
                    mult = min(max(float(self.image_boundary_scale_mult), 0.01), 1.0)
                    edge_scale = mult + (1.0 - mult) * eroded
                    xy_raw = self._raw_for_soft_capped_scale_tensor(
                        self.image_scale_frac * radius * edge_scale, cap
                    )
                    z_raw = self._raw_for_soft_capped_scale_tensor(
                        z_frac * radius * edge_scale, cap
                    )
                    raw_scale = torch.cat([xy_raw, xy_raw, z_raw], dim=1)
                else:
                    xy = self._raw_for_soft_capped_scale(self.image_scale_frac * radius, cap)
                    z = self._raw_for_soft_capped_scale(z_frac * radius, cap)
                    raw_scale = raw.new_tensor([xy, xy, z]).view(1, 3, 1, 1)
                raw[:, 3:6] = self.image_geom_residual_scale * raw[:, 3:6] + raw_scale
            if self.image_normal_quat:
                if image_cond.shape[1] < 9:
                    raise ValueError("image_normal_quat expects image_cond to end with normal xyz channels")
                normal = cond[:, -3:]
                q, n_valid = self._normal_map_to_quat_wxyz(normal)
                q_raw = q - raw.new_tensor([1.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1)
                raw_q = self.image_geom_residual_scale * raw[:, 6:10] + q_raw
                raw[:, 6:10] = torch.where(n_valid.expand_as(raw_q), raw_q, raw[:, 6:10])
            elif self.image_camera_quat:
                q = self._matrix_to_quat_wxyz(anchor_c2w[:3, :3].to(raw.dtype))
                q_raw = q - raw.new_tensor([1.0, 0.0, 0.0, 0.0])
                raw[:, 6:10] = (
                    self.image_geom_residual_scale * raw[:, 6:10]
                    + q_raw.view(1, 4, 1, 1)
                )
            elif self.image_geom_residual_scale != 1.0:
                raw[:, 6:10] = self.image_geom_residual_scale * raw[:, 6:10]
            if self.image_geom_residual_scale != 1.0:
                if not self.image_depth_skip:
                    raw[:, 11:12] = self.image_geom_residual_scale * raw[:, 11:12]
                if raw.shape[1] >= 15:
                    raw[:, 12:15] = self.image_geom_residual_scale * raw[:, 12:15]
        if self.image_depth_skip:
            if image_cond is None:
                raise ValueError("image_cond is required when image_depth_skip is enabled")
            if image_cond.shape[1] < 6:
                raise ValueError("image_depth_skip expects image_cond channels [rgb*mask, mask, depth, valid]")
            cond = F.interpolate(image_cond.to(dtype=raw.dtype), size=raw.shape[-2:],
                                 mode="bilinear", align_corners=False)
            depth_frac = cond[:, 4:5].clamp(1e-4, 1.0 - 1e-4)
            depth_valid = cond[:, 5:6].clamp(0.0, 1.0)
            depth_raw = torch.logit(depth_frac)
            raw[:, 11:12] = (
                self.image_depth_residual_scale * raw[:, 11:12]
                + depth_valid * depth_raw
            )
        if self.image_depth_prior_frac > 0:
            frac = min(max(self.image_depth_prior_frac, 1e-4), 1.0 - 1e-4)
            raw[:, 11:12] = raw[:, 11:12] + math.log(frac / (1.0 - frac))
        if self.depth_head is not None:
            raw[:, 11:12] = raw[:, 11:12] + self.depth_head_scale * self.depth_head(x)
        if self.visibility_head is not None:
            raw[:, 10:11] = raw[:, 10:11] + self.visibility_head_scale * self.visibility_head(x)
        raw = raw.permute(0, 2, 3, 1).reshape(b, self.map_h * self.map_w, self.out_channels)
        # ray-anchor to the reference camera
        dirs = ray_dirs_world(anchor_K, anchor_c2w, self.map_h, self.map_w).to(raw.dtype)  # (N,3)
        origins = anchor_c2w[:3, 3].to(raw.dtype).expand_as(dirs)
        d_near, d_far = depth_bounds(anchor_c2w, radius, self.half_frac)
        out = [activate(raw[i], origins, dirs, d_near, d_far, radius, self.scale_cap_frac,
                        self.mean_offset_frac)
               for i in range(b)]
        batched = {k: torch.stack([o[k] for o in out], 0) for k in out[0]}
        batched.update(raw_residual_l2)
        return batched
