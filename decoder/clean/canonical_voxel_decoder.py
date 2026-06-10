"""Canonical learned RGBD/latent voxel decoder.

This decoder is a more learned alternative to the surface-token path. It
first consolidates sampled RGBD observations into occupied canonical 3D voxels,
then runs learned occupied-neighborhood message passing plus cross-attention
to the full LTX latent grid before emitting one Gaussian per occupied voxel.

The initialization is intentionally close to a useful feed-forward RGBD prior:
voxel centers/colors/normals come from the input observations and the final
head starts at zero. Training owns the residual geometry, color, opacity, and
scale without per-object optimization.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder.clean.sparse_voxel_fusion import (
    DenseVoxelMessageBlock,
    dense_voxel_message_pairs,
)
from decoder.clean.surface_token_decoder import (
    _quat_from_normals,
    build_rgbd_surface_tokens,
)


def _project_points(points: torch.Tensor, w2c: torch.Tensor,
                    K: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    u = K[0, 0] * (cam[:, 0] / z.clamp_min(1e-6)) + K[0, 2]
    v = K[1, 1] * (cam[:, 1] / z.clamp_min(1e-6)) + K[1, 2]
    inb = (z > 1e-6) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    grid_x = (u / max(w - 1, 1)) * 2.0 - 1.0
    grid_y = (v / max(h - 1, 1)) * 2.0 - 1.0
    return z, torch.stack([grid_x, grid_y], dim=-1), inb


def _sample_map(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    n = grid.shape[0]
    return F.grid_sample(
        x, grid.view(1, n, 1, 2), mode="bilinear", padding_mode="zeros",
        align_corners=True,
    ).view(x.shape[1], n).T


def _scatter_mean(values: torch.Tensor, inverse: torch.Tensor,
                  weights: torch.Tensor, n_out: int) -> torch.Tensor:
    flat = values.reshape(values.shape[0], -1)
    w = weights.reshape(-1, 1).to(dtype=flat.dtype, device=flat.device)
    out = torch.zeros(n_out, flat.shape[1], dtype=flat.dtype, device=flat.device)
    den = torch.zeros(n_out, 1, dtype=flat.dtype, device=flat.device)
    out.scatter_add_(0, inverse[:, None].expand(-1, flat.shape[1]), flat * w)
    den.scatter_add_(0, inverse[:, None], w)
    out = out / den.clamp_min(1e-6)
    return out.reshape((n_out,) + values.shape[1:])


def _decode_shifted_keys(unique: torch.Tensor,
                         dims: torch.Tensor) -> torch.Tensor:
    qx = unique % dims[0].clamp_min(1)
    tmp = unique // dims[0].clamp_min(1)
    qy = tmp % dims[1].clamp_min(1)
    qz = tmp // dims[1].clamp_min(1)
    return torch.stack([qx, qy, qz], dim=-1)


def _normal_tangent_basis(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = F.normalize(normals, dim=-1)
    up_z = torch.zeros_like(n)
    up_z[:, 2] = 1.0
    up_y = torch.zeros_like(n)
    up_y[:, 1] = 1.0
    up = torch.where((n[:, 2:3].abs() > 0.95), up_y, up_z)
    t1 = F.normalize(torch.cross(up, n, dim=-1), dim=-1)
    t2 = F.normalize(torch.cross(n, t1, dim=-1), dim=-1)
    return t1, t2


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ViewFeatureEncoder(nn.Module):
    """Small shared CNN for high-resolution conditioning-view features."""

    def __init__(self, out_channels: int):
        super().__init__()
        out_channels = max(int(out_channels), 8)
        self.in_proj = nn.Sequential(
            nn.Conv2d(8, out_channels, 3, padding=1),
            _group_norm(out_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                _group_norm(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                _group_norm(out_channels),
            ),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2),
                _group_norm(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                _group_norm(out_channels),
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block(h)
            h = F.gelu(h)
        return h


def _voxelize_rgbd_surface(
    source_base: dict[str, torch.Tensor],
    radius: float,
    voxel_size: float,
    n_views: int,
    opacity_init: float,
    support_floor: float,
    support_target: float,
    tangent_scale_mult: float,
    normal_scale_mult: float,
    max_voxels: int,
) -> dict[str, torch.Tensor]:
    mean_src = source_base["mean"]
    device = mean_src.device
    dtype = mean_src.dtype
    valid = source_base["valid"].reshape(-1) > 0.0
    if not bool(valid.any()):
        empty = mean_src.new_zeros((0, 3))
        return {
            "features": mean_src.new_zeros((0, CanonicalVoxelDecoder.FEATURE_DIM)),
            "mean_anchor": empty,
            "rgb_anchor": empty,
            "quat_anchor": mean_src.new_zeros((0, 4)),
            "tangent1": empty,
            "tangent2": empty,
            "scale_base": empty,
            "opacity_prior": mean_src.new_zeros((0, 1)),
            "support": mean_src.new_zeros((0, 1)),
            "count": mean_src.new_zeros((0, 1)),
        }

    mean_v = mean_src[valid]
    rgb_v = source_base["rgb"][valid]
    ray_v = source_base["direction"][valid]
    mask_v = source_base["mask"][valid].reshape(-1, 1).clamp(0.0, 1.0)
    weights = mask_v.clamp_min(1e-4)
    voxel = max(float(voxel_size), 1e-8)
    radius_t = mean_src.new_tensor(max(float(radius), 1e-6))

    q_abs = torch.floor(mean_v.detach() / voxel).to(torch.long)
    q_min = q_abs.amin(dim=0, keepdim=True)
    q = q_abs - q_min
    dims = q.amax(dim=0) + 1
    key = q[:, 0] + dims[0] * (q[:, 1] + dims[1] * q[:, 2])
    unique, inverse = torch.unique(key, sorted=False, return_inverse=True)
    n_vox = int(unique.shape[0])

    count = torch.zeros(n_vox, 1, dtype=dtype, device=device)
    count.scatter_add_(0, inverse[:, None], torch.ones_like(weights))
    weight_sum = torch.zeros(n_vox, 1, dtype=dtype, device=device)
    weight_sum.scatter_add_(0, inverse[:, None], weights)
    mean = _scatter_mean(mean_v, inverse, weights, n_vox)
    rgb = _scatter_mean(rgb_v, inverse, weights, n_vox).clamp(0.0, 1.0)
    ray = F.normalize(_scatter_mean(ray_v, inverse, weights, n_vox), dim=-1)
    normal = F.normalize(-ray, dim=-1)

    mean_sq = _scatter_mean(mean_v.square(), inverse, weights, n_vox)
    rgb_sq = _scatter_mean(rgb_v.square(), inverse, weights, n_vox)
    pos_std = (mean_sq - mean.square()).clamp_min(0.0).sqrt()
    rgb_std = (rgb_sq - rgb.square()).clamp_min(0.0).sqrt()

    q_shift = _decode_shifted_keys(unique, dims)
    q_abs_unique = q_shift + q_min.reshape(1, 3)
    center = (q_abs_unique.to(dtype=dtype, device=device) + 0.5) * voxel

    view_id = source_base.get("view_id")
    if view_id is not None:
        view_v = view_id.reshape(-1).to(device=device)[valid].long()
        view_v = view_v.clamp(0, max(int(n_views) - 1, 0))
        one_hot = F.one_hot(view_v, num_classes=max(int(n_views), 1)).to(dtype=dtype)
        view_seen = torch.zeros(n_vox, one_hot.shape[-1], dtype=dtype, device=device)
        view_seen.scatter_add_(0, inverse[:, None].expand(-1, one_hot.shape[-1]), one_hot)
        view_support = (view_seen > 0).to(dtype).sum(dim=-1, keepdim=True)
    else:
        view_support = torch.ones(n_vox, 1, dtype=dtype, device=device)

    if max_voxels > 0 and n_vox > int(max_voxels):
        # Prefer voxels with multi-view support, then dense local evidence.
        score = view_support.reshape(-1) * 1000.0 + count.reshape(-1)
        keep = torch.topk(score, k=int(max_voxels), largest=True).indices
        mean = mean[keep]
        rgb = rgb[keep]
        ray = ray[keep]
        normal = normal[keep]
        pos_std = pos_std[keep]
        rgb_std = rgb_std[keep]
        center = center[keep]
        count = count[keep]
        weight_sum = weight_sum[keep]
        view_support = view_support[keep]
        n_vox = int(keep.numel())

    mean_norm = (mean / radius_t).clamp(-2.0, 2.0) / 2.0
    center_norm = (center / radius_t).clamp(-2.0, 2.0) / 2.0
    offset_norm = ((mean - center) / voxel).clamp(-2.0, 2.0) / 2.0
    pos_std_norm = (pos_std / (2.0 * voxel)).clamp(0.0, 1.0)
    count_norm = (torch.log1p(count) / math.log1p(64.0)).clamp(0.0, 2.0) / 2.0
    support_norm = (view_support / max(float(n_views), 1.0)).clamp(0.0, 1.0)
    weight_mean = (weight_sum / count.clamp_min(1.0)).clamp(0.0, 1.0)
    count_lin = (count / 32.0).clamp(0.0, 1.0)
    voxel_frac = mean.new_full((n_vox, 1), voxel / max(float(radius), 1e-6))

    tangent1, tangent2 = _normal_tangent_basis(normal)

    features = torch.cat([
        rgb,
        mean_norm,
        center_norm,
        offset_norm,
        normal,
        pos_std_norm,
        rgb_std.clamp(0.0, 1.0),
        count_norm,
        support_norm,
        weight_mean,
        count_lin,
        voxel_frac.clamp(0.0, 1.0),
    ], dim=-1)
    if features.shape[-1] != CanonicalVoxelDecoder.FEATURE_DIM:
        raise RuntimeError(f"canonical feature dim mismatch: {features.shape[-1]}")

    support_target = max(float(support_target), 1e-6)
    support_gate = (view_support / support_target).clamp(0.0, 1.0)
    support_gate = max(float(support_floor), 0.0) + (1.0 - max(float(support_floor), 0.0)) * support_gate
    opacity_prior = (float(opacity_init) * support_gate).clamp(1e-4, 1.0 - 1e-4)
    tangent = max(float(tangent_scale_mult), 1e-6) * voxel
    normal_s = max(float(normal_scale_mult), 1e-6) * voxel
    scale_base = mean.new_tensor([tangent, tangent, normal_s]).reshape(1, 3).expand(n_vox, 3)

    return {
        "features": features,
        "mean_anchor": mean,
        "rgb_anchor": rgb,
        "quat_anchor": _quat_from_normals(normal),
        "tangent1": tangent1,
        "tangent2": tangent2,
        "scale_base": scale_base,
        "opacity_prior": opacity_prior,
        "support": view_support,
        "count": count,
    }


class _CanonicalVoxelBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, message_radius: int = 1,
                 mlp_ratio: int = 4, scene_slots: int = 0):
        super().__init__()
        n_offsets = (2 * max(int(message_radius), 0) + 1) ** 3
        self.message = DenseVoxelMessageBlock(hidden, n_offsets)
        self.cross_norm = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.has_scene = int(scene_slots) > 0
        if self.has_scene:
            self.scene_q_norm = nn.LayerNorm(hidden)
            self.scene_kv_norm = nn.LayerNorm(hidden)
            self.scene_from_vox = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.vox_scene_q_norm = nn.LayerNorm(hidden)
            self.vox_scene_kv_norm = nn.LayerNorm(hidden)
            self.vox_from_scene = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden * mlp_ratio, hidden),
        )

    def forward(self, vox: torch.Tensor, latent_tokens: torch.Tensor,
                pairs: list[tuple[torch.Tensor, torch.Tensor]],
                scene_tokens: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        vox = self.message(vox, pairs)
        q = self.cross_norm(vox)[None]
        k = latent_tokens[None]
        msg, _ = self.cross_attn(q, k, k, need_weights=False)
        vox = vox + msg[0]
        if self.has_scene and scene_tokens is not None:
            scene_q = self.scene_q_norm(scene_tokens)[None]
            vox_kv = self.scene_kv_norm(vox)[None]
            scene_msg, _ = self.scene_from_vox(scene_q, vox_kv, vox_kv, need_weights=False)
            scene_tokens = scene_tokens + scene_msg[0]

            vox_q = self.vox_scene_q_norm(vox)[None]
            scene_kv = self.vox_scene_kv_norm(scene_tokens)[None]
            vox_msg, _ = self.vox_from_scene(vox_q, scene_kv, scene_kv, need_weights=False)
            vox = vox + vox_msg[0]
        vox = vox + self.ff(self.ff_norm(vox))
        return vox, scene_tokens


class CanonicalVoxelDecoder(nn.Module):
    """Learned canonical occupied-voxel Gaussian decoder."""

    FEATURE_DIM = 26
    DETAIL_DIM = 13
    CONSISTENCY_DIM = 18

    def __init__(
        self,
        latent_channels: int = 128,
        hidden: int = 384,
        layers: int = 5,
        heads: int = 8,
        latent_layers: int = 0,
        scene_slots: int = 0,
        grid_h: int = 72,
        grid_w: int = 108,
        latent_pool: int = 2,
        message_radius: int = 1,
        voxel_size_frac: float = 0.003,
        max_voxels: int = 60000,
        gaussians_per_voxel: int = 1,
        child_offset_mult: float = 0.35,
        mean_res_voxels: float = 0.75,
        rgb_res_scale: float = 0.25,
        tangent_scale_mult: float = 0.45,
        normal_scale_mult: float = 0.10,
        scale_res_scale: float = 0.75,
        quat_res_scale: float = 0.20,
        opacity_init: float = 0.82,
        opacity_support_floor: float = 0.35,
        opacity_support_target: float = 2.0,
        detail_sampling: bool = False,
        detail_color_mix: float = 0.75,
        detail_depth_tol_frac: float = 0.015,
        detail_score_temp: float = 0.75,
        detail_chunk: int = 16384,
        view_feature_channels: int = 0,
        view_feature_scale: float = 0.5,
        opacity_prior_weight: float = 1.0,
        zero_init_head: bool = True,
        source_consistency_refine: bool = False,
        source_consistency_hidden: int = 128,
        source_consistency_opacity_strength: float = 1.0,
        source_consistency_rgb_scale: float = 0.10,
        source_consistency_scale_res_scale: float = 0.25,
        source_consistency_zero_init: bool = True,
    ):
        super().__init__()
        hidden = max(int(hidden), 32)
        layers = max(int(layers), 1)
        heads = max(int(heads), 1)
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.grid_h = max(int(grid_h), 1)
        self.grid_w = max(int(grid_w), 1)
        self.latent_pool = max(int(latent_pool), 1)
        self.message_radius = max(int(message_radius), 0)
        self.voxel_size_frac = float(voxel_size_frac)
        self.max_voxels = max(int(max_voxels), 0)
        self.gaussians_per_voxel = max(int(gaussians_per_voxel), 1)
        self.child_offset_mult = float(child_offset_mult)
        self.mean_res_voxels = float(mean_res_voxels)
        self.rgb_res_scale = float(rgb_res_scale)
        self.tangent_scale_mult = float(tangent_scale_mult)
        self.normal_scale_mult = float(normal_scale_mult)
        self.scale_res_scale = float(scale_res_scale)
        self.quat_res_scale = float(quat_res_scale)
        self.opacity_init = float(opacity_init)
        self.opacity_support_floor = float(opacity_support_floor)
        self.opacity_support_target = float(opacity_support_target)
        self.detail_sampling = bool(detail_sampling)
        self.detail_color_mix = float(detail_color_mix)
        self.detail_depth_tol_frac = float(detail_depth_tol_frac)
        self.detail_score_temp = float(detail_score_temp)
        self.detail_chunk = max(int(detail_chunk), 1024)
        self.view_feature_channels = max(int(view_feature_channels), 0)
        self.view_feature_scale = float(view_feature_scale)
        self.opacity_prior_weight = float(opacity_prior_weight)
        self.zero_init_head = bool(zero_init_head)
        self.scene_slots = max(int(scene_slots), 0)
        self.source_consistency_opacity_strength = float(source_consistency_opacity_strength)
        self.source_consistency_rgb_scale = float(source_consistency_rgb_scale)
        self.source_consistency_scale_res_scale = float(source_consistency_scale_res_scale)

        self.source_proj = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(int(latent_channels) + 6, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        latent_layers = max(int(latent_layers), 0)
        if latent_layers > 0:
            self.latent_context = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=heads,
                    dim_feedforward=hidden * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=latent_layers,
            )
        else:
            self.latent_context = None
        if self.scene_slots > 0:
            self.scene_tokens = nn.Parameter(torch.randn(self.scene_slots, hidden) * 0.02)
            self.scene_latent = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
        else:
            self.scene_tokens = None
            self.scene_latent = None
        self.blocks = nn.ModuleList([
            _CanonicalVoxelBlock(hidden, heads, self.message_radius,
                                 scene_slots=self.scene_slots)
            for _ in range(layers)
        ])
        if self.detail_sampling:
            if self.view_feature_channels > 0:
                self.view_feature_encoder = _ViewFeatureEncoder(self.view_feature_channels)
            else:
                self.view_feature_encoder = None
            detail_dim = self.DETAIL_DIM + self.view_feature_channels
            self.detail_view_proj = nn.Sequential(
                nn.Linear(detail_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.detail_score = nn.Linear(hidden, 1)
            self.detail_fuse = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 14 * self.gaussians_per_voxel),
        )
        if self.zero_init_head:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        if source_consistency_refine:
            consistency_hidden = max(int(source_consistency_hidden), 32)
            self.source_consistency_refine = nn.Sequential(
                nn.LayerNorm(self.CONSISTENCY_DIM),
                nn.Linear(self.CONSISTENCY_DIM, consistency_hidden),
                nn.GELU(),
                nn.Linear(consistency_hidden, consistency_hidden),
                nn.GELU(),
                nn.Linear(consistency_hidden, 7),
            )
            if source_consistency_zero_init:
                nn.init.zeros_(self.source_consistency_refine[-1].weight)
                nn.init.zeros_(self.source_consistency_refine[-1].bias)
        else:
            self.source_consistency_refine = None
        self.child_offsets = nn.Parameter(
            self._child_pattern(self.gaussians_per_voxel, torch.float32, torch.device("cpu"))
        )

    def _latent_tokens(self, latent: torch.Tensor, dtype: torch.dtype,
                       device: torch.device) -> torch.Tensor:
        if latent.ndim == 5:
            latent = latent[0]
        if latent.ndim != 4:
            raise ValueError("latent must have shape (C,T,H,W) or (B,C,T,H,W)")
        z = latent.to(device=device, dtype=dtype)
        if self.latent_pool > 1:
            k = self.latent_pool
            z = F.avg_pool3d(z[None], kernel_size=(1, k, k), stride=(1, k, k))[0]
        c, t, h, w = z.shape
        vals = z.permute(1, 2, 3, 0).reshape(-1, c)
        tt = torch.linspace(-1.0, 1.0, t, dtype=dtype, device=device)
        yy = torch.linspace(-1.0, 1.0, h, dtype=dtype, device=device)
        xx = torch.linspace(-1.0, 1.0, w, dtype=dtype, device=device)
        g_t, g_y, g_x = torch.meshgrid(tt, yy, xx, indexing="ij")
        pos = torch.stack([
            g_t.reshape(-1),
            g_y.reshape(-1),
            g_x.reshape(-1),
            torch.sin(math.pi * g_t).reshape(-1),
            torch.sin(math.pi * g_y).reshape(-1),
            torch.sin(math.pi * g_x).reshape(-1),
        ], dim=-1)
        tokens = self.latent_proj(torch.cat([vals, pos], dim=-1))
        if self.latent_context is not None:
            tokens = self.latent_context(tokens[None])[0]
        return tokens

    def _view_feature_maps(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        radius: float,
    ) -> torch.Tensor | None:
        if not self.detail_sampling or self.view_feature_encoder is None:
            return None
        device, dtype = frames.device, frames.dtype
        v, h, w, _ = frames.shape
        rgb = frames.permute(0, 3, 1, 2).to(device=device, dtype=dtype)
        mask = masks.permute(0, 3, 1, 2).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        depth = depths[:, None].to(device=device, dtype=dtype)
        finite = ((depth > 1e-6) & (depth < 1e5)).to(dtype)
        depth_norm = (depth / max(float(radius), 1e-6)).clamp(0.0, 4.0) / 4.0
        yy = torch.linspace(-1.0, 1.0, h, dtype=dtype, device=device)
        xx = torch.linspace(-1.0, 1.0, w, dtype=dtype, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        pos = torch.stack([gx, gy], dim=0).reshape(1, 2, h, w).expand(v, -1, -1, -1)
        x = torch.cat([rgb * mask, mask, depth_norm * finite, finite, pos], dim=1)
        scale = float(self.view_feature_scale)
        if scale > 0.0 and abs(scale - 1.0) > 1e-6:
            size = (max(int(round(h * scale)), 1), max(int(round(w * scale)), 1))
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.view_feature_encoder(x)

    def _sample_detail_views(
        self,
        points: torch.Tensor,
        frames: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        K: torch.Tensor,
        w2c: torch.Tensor,
        radius: float,
        view_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device, dtype = points.device, points.dtype
        v, h, w, _ = frames.shape
        frame_imgs = frames.permute(0, 3, 1, 2).to(device=device, dtype=dtype)
        mask_imgs = masks.permute(0, 3, 1, 2).to(device=device, dtype=dtype)
        depth_imgs = depths[:, None].to(device=device, dtype=dtype)
        K = K.to(device=device, dtype=dtype)
        w2c = w2c.to(device=device, dtype=dtype)
        tol = max(float(self.detail_depth_tol_frac) * max(float(radius), 1e-6), 1e-6)
        feats = []
        rgbs = []
        valids = []
        for i in range(v):
            z, grid, inb = _project_points(points, w2c[i], K[i], h, w)
            samp_rgb = _sample_map(frame_imgs[i:i + 1], grid).clamp(0.0, 1.0)
            samp_m = _sample_map(mask_imgs[i:i + 1], grid)[:, 0:1].clamp(0.0, 1.0)
            samp_d = _sample_map(depth_imgs[i:i + 1], grid)[:, 0:1]
            d_rel = ((samp_d - z[:, None]) / tol).clamp(-4.0, 4.0) / 4.0
            d_conf = torch.exp(-((samp_d - z[:, None]).abs() / tol).clamp(0.0, 20.0))
            inb_f = inb.to(dtype).reshape(-1, 1)
            finite = ((samp_d > 1e-6) & (samp_d < 1e5)).to(dtype)
            valid = inb_f * finite * (samp_m > 0.05).to(dtype)
            phase = points.new_full((points.shape[0], 1), 2.0 * math.pi * float(i) / max(float(v), 1.0))
            z_norm = (z[:, None] / max(float(radius), 1e-6)).clamp(0.0, 4.0) / 4.0
            feat = torch.cat([
                samp_rgb * samp_m,
                samp_m,
                d_rel,
                d_conf,
                inb_f,
                valid,
                torch.sin(phase),
                torch.cos(phase),
                z_norm,
                grid.clamp(-1.0, 1.0),
            ], dim=-1)
            if view_features is not None:
                samp_feat = _sample_map(view_features[i:i + 1], grid)
                feat = torch.cat([feat, samp_feat], dim=-1)
            feats.append(feat)
            rgbs.append(samp_rgb)
            valids.append(valid)
        return (
            torch.stack(feats, dim=1),
            torch.stack(rgbs, dim=1),
            torch.stack(valids, dim=1),
        )

    def _detail_condition(
        self,
        h: torch.Tensor,
        points: torch.Tensor,
        frames: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        K: torch.Tensor,
        w2c: torch.Tensor,
        radius: float,
        view_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summaries = []
        rgbs = []
        confs = []
        temp = max(float(self.detail_score_temp), 1e-4)
        for start in range(0, points.shape[0], self.detail_chunk):
            end = min(start + self.detail_chunk, points.shape[0])
            detail, sampled_rgb, valid = self._sample_detail_views(
                points[start:end], frames, masks, depths, K, w2c, radius,
                view_features=view_features,
            )
            n, v, _ = detail.shape
            view_h = self.detail_view_proj(detail.reshape(n * v, -1)).reshape(n, v, -1)
            q = h[start:end, None, :]
            scores = self.detail_score(torch.tanh(view_h + q)).squeeze(-1)
            scores = scores + valid.squeeze(-1).clamp_min(1e-4).log()
            weights = torch.softmax(scores / temp, dim=1)
            conf = (weights[..., None] * valid).sum(dim=1).clamp(0.0, 1.0)
            summary = (weights[..., None] * view_h).sum(dim=1)
            summary = self.detail_fuse(summary) * conf
            rgb = (weights[..., None] * sampled_rgb).sum(dim=1)
            summaries.append(summary)
            rgbs.append(rgb)
            confs.append(conf)
        return torch.cat(summaries, dim=0), torch.cat(rgbs, dim=0), torch.cat(confs, dim=0)

    def forward(
        self,
        latent: torch.Tensor,
        frames: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        radius: float,
        w2c: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, source_base = build_rgbd_surface_tokens(
            frames, masks, depths, K, c2w, radius, self.grid_h, self.grid_w
        )
        source_base["view_id"] = source_base["view_id"].to(device=frames.device)
        voxel_size = max(float(self.voxel_size_frac) * max(float(radius), 1e-6), 1e-8)
        vox = _voxelize_rgbd_surface(
            source_base,
            radius=radius,
            voxel_size=voxel_size,
            n_views=int(frames.shape[0]),
            opacity_init=self.opacity_init,
            support_floor=self.opacity_support_floor,
            support_target=self.opacity_support_target,
            tangent_scale_mult=self.tangent_scale_mult,
            normal_scale_mult=self.normal_scale_mult,
            max_voxels=self.max_voxels,
        )
        if vox["features"].shape[0] == 0:
            empty = frames.new_zeros((0, 3))
            return {
                "mean": empty,
                "quat": frames.new_zeros((0, 4)),
                "scale": empty,
                "opacity": frames.new_zeros((0, 1)),
                "rgb": empty,
                "depth": frames.new_zeros((0, 1)),
                "mean_anchor": empty,
                "mean_offset": empty,
                "scale_raw": empty,
                "_canonical_voxel_valid": frames.new_zeros((0, 1)),
            }

        h = self.source_proj(vox["features"])
        latent_tokens = self._latent_tokens(latent, h.dtype, h.device)
        scene_tokens = None
        if self.scene_tokens is not None and self.scene_latent is not None:
            scene_bias = self.scene_latent(latent_tokens.mean(dim=0, keepdim=True))
            scene_tokens = self.scene_tokens.to(dtype=h.dtype, device=h.device) + scene_bias
        pairs = dense_voxel_message_pairs(
            {"mean": vox["mean_anchor"].detach()},
            voxel_size=voxel_size,
            neighbor_radius=self.message_radius,
        )
        for block in self.blocks:
            h, scene_tokens = block(h, latent_tokens, pairs, scene_tokens)
        rgb_anchor = vox["rgb_anchor"]
        if self.detail_sampling:
            if w2c is None:
                raise ValueError("canonical detail sampling requires w2c cameras")
            view_features = self._view_feature_maps(frames, masks, depths, radius)
            detail_h, detail_rgb, detail_conf = self._detail_condition(
                h, vox["mean_anchor"], frames, masks, depths, K, w2c, radius,
                view_features=view_features,
            )
            h = h + detail_h
            mix = (float(self.detail_color_mix) * detail_conf).clamp(0.0, 1.0)
            rgb_anchor = (rgb_anchor * (1.0 - mix) + detail_rgb * mix).clamp(0.0, 1.0)
        raw = self.head(h).reshape(h.shape[0], self.gaussians_per_voxel, 14)

        voxel_t = h.new_tensor(voxel_size)
        pattern = self.child_offsets.to(dtype=h.dtype, device=h.device) * (
            self.child_offset_mult * voxel_t
        )
        base_offset = (
            pattern[:, 0].reshape(1, -1, 1) * vox["tangent1"][:, None, :]
            + pattern[:, 1].reshape(1, -1, 1) * vox["tangent2"][:, None, :]
        )
        mean_anchor = vox["mean_anchor"][:, None, :] + base_offset
        mean_res = torch.tanh(raw[:, :, 0:3]) * (self.mean_res_voxels * voxel_t)
        mean = mean_anchor + mean_res
        rgb = (
            rgb_anchor[:, None, :]
            + torch.tanh(raw[:, :, 3:6]) * self.rgb_res_scale
        ).clamp(0.0, 1.0)
        scale_mult = torch.exp(torch.tanh(raw[:, :, 6:9]) * self.scale_res_scale)
        scale = (vox["scale_base"][:, None, :] * scale_mult).clamp_min(1e-6)
        quat = F.normalize(
            vox["quat_anchor"][:, None, :]
            + self.quat_res_scale * torch.tanh(raw[:, :, 9:13]),
            dim=-1,
        )
        prior = vox["opacity_prior"].clamp(1e-4, 1.0 - 1e-4)
        if self.gaussians_per_voxel > 1:
            # Keep roughly the same combined opacity if children overlap while
            # allowing them to cover different parts of the voxel footprint.
            prior = 1.0 - (1.0 - prior).pow(1.0 / float(self.gaussians_per_voxel))
        prior_logit = torch.log(prior / (1.0 - prior))
        opacity = torch.sigmoid(
            raw[:, :, 13:14] + float(self.opacity_prior_weight) * prior_logit[:, None, :]
        )
        n, k = mean.shape[:2]
        mean = mean.reshape(n * k, 3)
        mean_anchor = mean_anchor.reshape(n * k, 3)
        mean_res = mean_res.reshape(n * k, 3)
        rgb = rgb.reshape(n * k, 3)
        scale = scale.reshape(n * k, 3)
        quat = quat.reshape(n * k, 4)
        opacity = opacity.reshape(n * k, 1)
        depth = mean.norm(dim=-1, keepdim=True)
        return {
            "mean": mean,
            "quat": quat,
            "scale": scale,
            "opacity": opacity,
            "rgb": rgb,
            "depth": depth,
            "mean_anchor": mean_anchor,
            "mean_offset": mean_res,
            "scale_raw": scale,
            "_canonical_voxel_valid": torch.ones_like(opacity),
            "_canonical_voxel_support": vox["support"].repeat_interleave(k, dim=0),
            "_canonical_voxel_count": vox["count"].repeat_interleave(k, dim=0),
        }

    def _source_consistency_features(self, p: dict[str, torch.Tensor],
                                     radius: float) -> torch.Tensor:
        mean = p["mean"]
        like = mean[:, :1]

        def field(name: str) -> torch.Tensor:
            x = p.get(name)
            if x is None:
                return torch.zeros_like(like)
            return x.reshape(mean.shape[0], -1)[:, :1].to(
                device=mean.device, dtype=mean.dtype
            )

        support = field("_fusion_support")
        conflict = field("_fusion_conflict")
        coverage = field("_fusion_coverage")
        color_support = field("_fusion_color_support")
        depth_error = field("_fusion_depth_error")
        color_error = field("_fusion_color_error")
        front_conflict = field("_fusion_front_conflict")
        sil_conflict = field("_fusion_silhouette_conflict")
        voxel_support = field("_canonical_voxel_support")
        voxel_count = field("_canonical_voxel_count")
        denom = coverage.clamp_min(1.0)
        radius_t = max(float(radius), 1e-6)
        opacity = p["opacity"].clamp(1e-4, 1.0 - 1e-4)
        opacity_logit = torch.log(opacity / (1.0 - opacity))
        scale = p["scale"].clamp_min(1e-8)
        scale_mean = scale.mean(dim=-1, keepdim=True)
        scale_max = scale.amax(dim=-1, keepdim=True)
        mean_norm = (mean.norm(dim=-1, keepdim=True) / radius_t).clamp(0.0, 4.0) / 4.0
        return torch.cat([
            (support / 8.0).clamp(0.0, 2.0),
            (conflict / 8.0).clamp(0.0, 2.0),
            (coverage / 8.0).clamp(0.0, 2.0),
            ((support - conflict) / 8.0).clamp(-2.0, 2.0),
            (color_support / 8.0).clamp(0.0, 2.0),
            (depth_error / denom / 4.0).clamp(0.0, 2.0),
            (color_error / denom).clamp(0.0, 2.0),
            (front_conflict / 8.0).clamp(0.0, 2.0),
            (sil_conflict / 8.0).clamp(0.0, 2.0),
            (voxel_support / 8.0).clamp(0.0, 2.0),
            (torch.log1p(voxel_count) / math.log1p(256.0)).clamp(0.0, 2.0),
            opacity,
            (opacity_logit / 8.0).clamp(-2.0, 2.0),
            (scale_mean / (0.006 * radius_t)).clamp(0.0, 4.0) / 4.0,
            (scale_max / (0.006 * radius_t)).clamp(0.0, 4.0) / 4.0,
            mean_norm,
            (support / denom).clamp(0.0, 2.0),
            (conflict / denom).clamp(0.0, 2.0),
        ], dim=-1)

    def refine_source_consistency(self, p: dict[str, torch.Tensor],
                                  radius: float) -> dict[str, torch.Tensor]:
        """Learn opacity/color/scale corrections from source-view consistency.

        The train loop computes multi-view support/conflict counts by projecting
        emitted Gaussians into the conditioning RGBD views. This method keeps
        that evidence inside the learned decoder instead of applying a fixed
        hand-tuned opacity threshold.
        """
        if self.source_consistency_refine is None or p["mean"].numel() == 0:
            return p
        delta = self.source_consistency_refine(
            self._source_consistency_features(p, radius)
        )
        out = dict(p)
        opacity = p["opacity"].clamp(1e-4, 1.0 - 1e-4)
        logit = torch.log(opacity / (1.0 - opacity))
        logit = logit + float(self.source_consistency_opacity_strength) * delta[:, 0:1]
        out["opacity"] = torch.sigmoid(logit)
        rgb_scale = float(self.source_consistency_rgb_scale)
        if rgb_scale > 0.0:
            out["rgb"] = (
                p["rgb"] + rgb_scale * torch.tanh(delta[:, 1:4])
            ).clamp(0.0, 1.0)
        scale_res = float(self.source_consistency_scale_res_scale)
        if scale_res > 0.0:
            scale_mult = torch.exp(scale_res * torch.tanh(delta[:, 4:7]))
            out["scale"] = (p["scale"] * scale_mult).clamp_min(1e-6)
            out["scale_raw"] = out["scale"]
        out["_canonical_source_vis_refine_delta"] = delta
        return out

    @staticmethod
    def _child_pattern(k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if k <= 1:
            vals = [[0.0, 0.0]]
        elif k == 2:
            vals = [[-1.0, 0.0], [1.0, 0.0]]
        elif k == 3:
            vals = [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]
        elif k == 4:
            vals = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
        else:
            side = int(math.ceil(math.sqrt(float(k))))
            coords = torch.linspace(-1.0, 1.0, side, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:k]
        return torch.tensor(vals, dtype=dtype, device=device)
