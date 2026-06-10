"""Learned RGBD/latent surface-token decoder.

This module is the first step away from post-hoc cleanup of the deterministic
RGBD fusion scaffold. It consumes sampled conditioning RGBD views plus a latent
summary, exchanges information through a small learned slot bottleneck, then
emits one Gaussian per sampled source surface token.

The geometry prior is still useful: source tokens start at RGBD-unprojected
surface points. The important difference from the previous fusion stack is that
the trainable network owns the Gaussian residuals before any voxel selection or
candidate pruning can freeze the surface.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder.data import zdepth_to_raydist
from decoder.clean.geometry import ray_dirs_world


def _scale_intrinsics(K: torch.Tensor, src_h: int, src_w: int,
                      dst_h: int, dst_w: int) -> torch.Tensor:
    K_s = K.clone()
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    K_s[..., 0, 0] = K_s[..., 0, 0] * sx
    K_s[..., 0, 2] = K_s[..., 0, 2] * sx
    K_s[..., 1, 1] = K_s[..., 1, 1] * sy
    K_s[..., 1, 2] = K_s[..., 1, 2] * sy
    return K_s


def _resize_nhwc(x: torch.Tensor, h: int, w: int,
                 mode: str = "bilinear") -> torch.Tensor:
    x_nchw = x.permute(0, 3, 1, 2)
    if mode == "nearest":
        y = F.interpolate(x_nchw, size=(h, w), mode=mode)
    else:
        y = F.interpolate(x_nchw, size=(h, w), mode=mode, align_corners=False)
    return y.permute(0, 2, 3, 1)


def _quat_from_normals(normals: torch.Tensor) -> torch.Tensor:
    """World normals ``(N,3)`` -> quaternions whose local z follows normal."""
    n = F.normalize(normals, dim=-1)
    up_z = torch.zeros_like(n)
    up_z[:, 2] = 1.0
    up_y = torch.zeros_like(n)
    up_y[:, 1] = 1.0
    up = torch.where((n[:, 2:3].abs() > 0.95), up_y, up_z)
    t1 = F.normalize(torch.cross(up, n, dim=-1), dim=-1)
    t2 = F.normalize(torch.cross(n, t1, dim=-1), dim=-1)
    m00, m10, m20 = t1[:, 0], t1[:, 1], t1[:, 2]
    m01, m11, m21 = t2[:, 0], t2[:, 1], t2[:, 2]
    m02, m12, m22 = n[:, 0], n[:, 1], n[:, 2]
    qw = 0.5 * torch.sqrt((1.0 + m00 + m11 + m22).clamp_min(1e-8))
    qx = 0.5 * torch.copysign(torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-8)), m21 - m12)
    qy = 0.5 * torch.copysign(torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(1e-8)), m02 - m20)
    qz = 0.5 * torch.copysign(torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(1e-8)), m10 - m01)
    q = F.normalize(torch.stack([qw, qx, qy, qz], dim=-1), dim=-1)
    return torch.where(q[:, :1] < 0, -q, q)


def _normals_from_depth_points(
    points: torch.Tensor,
    ray_dirs: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Estimate world-space surface normals on a sampled RGBD grid."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (H,W,3)")
    h, w, _ = points.shape
    fallback = F.normalize(-ray_dirs.reshape(h, w, 3), dim=-1)
    if h < 2 or w < 2:
        return fallback.reshape(-1, 3)
    valid_hw = valid.reshape(h, w).bool()
    dx = torch.zeros_like(points)
    dy = torch.zeros_like(points)
    if w > 2:
        dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dx[:, 0] = points[:, 1] - points[:, 0]
    dx[:, -1] = points[:, -1] - points[:, -2]
    if h > 2:
        dy[1:-1] = points[2:] - points[:-2]
    dy[0] = points[1] - points[0]
    dy[-1] = points[-1] - points[-2]
    normal = torch.cross(dx, dy, dim=-1)
    normal_norm = normal.norm(dim=-1, keepdim=True)
    normal = normal / normal_norm.clamp_min(1e-8)
    align = (normal * fallback).sum(dim=-1, keepdim=True)
    normal = torch.where(align < 0, -normal, normal)
    good = valid_hw[..., None] & torch.isfinite(normal).all(dim=-1, keepdim=True)
    good = good & (normal_norm > 1e-8)
    return torch.where(good, normal, fallback).reshape(-1, 3)


def _proposal_base_grid(count: int, extent: float) -> torch.Tensor:
    """Deterministic trainable proposal seeds in normalized object space."""
    count = max(int(count), 1)
    side = math.ceil(count ** (1.0 / 3.0))
    coords_1d = torch.linspace(-float(extent), float(extent), side, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    if coords.shape[0] == count:
        return coords
    ids = torch.linspace(0, coords.shape[0] - 1, count, dtype=torch.float32).round().long()
    return coords.index_select(0, ids)


def build_rgbd_surface_tokens(
    frames: torch.Tensor,
    masks: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    radius: float,
    grid_h: int,
    grid_w: int,
    use_depth_normals: bool = False,
    depth_normal_blend: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample RGBD views into fixed-size source surface tokens.

    Args:
        frames: ``(V,H,W,3)`` RGB in [0,1].
        masks: ``(V,H,W,1)`` foreground mask.
        depths: ``(V,H,W)`` camera z-depth.
        K: ``(V,3,3)`` intrinsics for the full-resolution frames.
        c2w: ``(V,4,4)`` OpenGL camera-to-world matrices.
        radius: object/camera radius used for normalization.
        grid_h/grid_w: sampled token map resolution per conditioning view.

    Returns:
        ``features`` of shape ``(V*grid_h*grid_w, 20)`` and base tensors used
        by the decoder heads.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (V,H,W,3)")
    if masks.ndim != 4 or masks.shape[-1] != 1:
        raise ValueError("masks must have shape (V,H,W,1)")
    if depths.ndim != 3:
        raise ValueError("depths must have shape (V,H,W)")
    if frames.shape[:3] != masks.shape[:3] or frames.shape[:3] != depths.shape:
        raise ValueError("frames, masks, and depths must agree on V,H,W")
    v, src_h, src_w, _ = frames.shape
    device = frames.device
    dtype = frames.dtype
    grid_h = max(int(grid_h), 1)
    grid_w = max(int(grid_w), 1)
    radius_t = frames.new_tensor(max(float(radius), 1e-6))

    rgb = _resize_nhwc(frames, grid_h, grid_w, mode="bilinear").clamp(0.0, 1.0)
    mask = _resize_nhwc(masks, grid_h, grid_w, mode="bilinear").clamp(0.0, 1.0)
    depth = F.interpolate(
        depths[:, None], size=(grid_h, grid_w), mode="nearest"
    )[:, 0].to(dtype=dtype)
    valid = ((depth > 1e-6) & (depth < 1e5) & (mask[..., 0] > 0.02)).to(dtype)

    K_s = _scale_intrinsics(K.to(device=device, dtype=dtype), src_h, src_w, grid_h, grid_w)
    means = []
    dirs = []
    ray_t = []
    origins = []
    view_ids = []
    normals = []
    for i in range(v):
        dirs_i = ray_dirs_world(K_s[i], c2w[i].to(device=device, dtype=dtype), grid_h, grid_w)
        t_i = zdepth_to_raydist(depth[i], K_s[i]).reshape(-1, 1)
        origin_i = c2w[i, :3, 3].to(device=device, dtype=dtype).reshape(1, 3)
        mean_i = origin_i + t_i * dirs_i
        means.append(mean_i)
        dirs.append(dirs_i)
        ray_t.append(t_i)
        origins.append(origin_i.expand_as(dirs_i))
        view_ids.append(torch.full((grid_h * grid_w, 1), float(i), device=device, dtype=dtype))
        if use_depth_normals:
            normals.append(
                _normals_from_depth_points(
                    mean_i.reshape(grid_h, grid_w, 3),
                    dirs_i,
                    valid[i].reshape(grid_h, grid_w),
                )
            )
    mean = torch.cat(means, dim=0)
    direction = torch.cat(dirs, dim=0)
    raydist = torch.cat(ray_t, dim=0)
    origin = torch.cat(origins, dim=0)
    view_id = torch.cat(view_ids, dim=0)
    ray_normal = -direction
    if normals:
        depth_normal = torch.cat(normals, dim=0)
        if torch.is_tensor(depth_normal_blend):
            blend = depth_normal_blend.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        else:
            blend = mean.new_tensor(min(max(float(depth_normal_blend), 0.0), 1.0))
        normal = F.normalize(ray_normal + blend * (depth_normal - ray_normal), dim=-1)
    else:
        depth_normal = ray_normal
        normal = ray_normal

    rgb_f = rgb.reshape(-1, 3)
    mask_f = mask.reshape(-1, 1)
    valid_f = valid.reshape(-1, 1)
    view_phase = 2.0 * math.pi * view_id / max(float(v), 1.0)
    mean_norm = (mean / radius_t).clamp(-2.0, 2.0) / 2.0
    origin_norm = (origin / radius_t).clamp(-2.0, 2.0) / 2.0
    ray_norm = (raydist / radius_t).clamp(0.0, 4.0) / 4.0
    rgb_masked = rgb_f * mask_f
    features = torch.cat([
        rgb_masked,                 # 3
        mask_f,                     # 1
        valid_f,                    # 1
        mean_norm,                  # 3
        direction,                  # 3
        origin_norm,                # 3
        ray_norm,                   # 1
        torch.sin(view_phase),      # 1
        torch.cos(view_phase),      # 1
        rgb_f,                      # 3
    ], dim=-1)
    base = {
        "mean": mean,
        "rgb": rgb_f,
        "mask": mask_f,
        "valid": valid_f,
        "raydist": raydist,
        "direction": direction,
        "ray_normal": ray_normal,
        "depth_normal": depth_normal,
        "view_id": view_id,
        "quat": _quat_from_normals(normal),
    }
    return features, base


class _SlotSurfaceBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.slot_from_src = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.src_from_slot = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.slot_norm1 = nn.LayerNorm(hidden)
        self.src_norm1 = nn.LayerNorm(hidden)
        self.slot_norm2 = nn.LayerNorm(hidden)
        self.src_norm2 = nn.LayerNorm(hidden)
        self.slot_ff = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden * mlp_ratio, hidden),
        )
        self.src_ff = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden * mlp_ratio, hidden),
        )

    def forward(self, src: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slot_msg, _ = self.slot_from_src(slots, src, src, need_weights=False)
        slots = self.slot_norm1(slots + slot_msg)
        slots = self.slot_norm2(slots + self.slot_ff(slots))
        src_msg, _ = self.src_from_slot(src, slots, slots, need_weights=False)
        src = self.src_norm1(src + src_msg)
        src = self.src_norm2(src + self.src_ff(src))
        return src, slots


class _LatentSlotBlock(nn.Module):
    """Let scene slots read the full latent grid before touching source splats."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.slot_from_latent = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden * mlp_ratio, hidden),
        )
        nn.init.zeros_(self.slot_from_latent.out_proj.weight)
        nn.init.zeros_(self.slot_from_latent.out_proj.bias)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, slots: torch.Tensor, latent_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm1(slots)
        msg, _ = self.slot_from_latent(q, latent_tokens, latent_tokens, need_weights=False)
        slots = slots + msg
        return slots + self.ff(self.norm2(slots))


class _SlotSelfBlock(nn.Module):
    """Parameter-heavy slot-only refinement with modest activation cost."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        mid = max(int(hidden) * max(int(mlp_ratio), 1), hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, mid),
            nn.GELU(),
            nn.Linear(mid, hidden),
        )
        nn.init.zeros_(self.self_attn.out_proj.weight)
        nn.init.zeros_(self.self_attn.out_proj.bias)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        q = self.norm1(slots)
        msg, _ = self.self_attn(q, q, q, need_weights=False)
        slots = slots + msg
        return slots + self.ff(self.norm2(slots))


class SurfaceTokenViewSelector(nn.Module):
    """Zero-init global source-view selector for the surface-token decoder.

    The same scalar controls both the top-k score and the opacity gate for a
    selected view. Render loss can therefore train the score through the gate:
    bad selected views get downweighted, their score falls, and unselected
    views can enter once they beat the tiny scaffold prior.
    """

    FEATURE_DIM = 15

    def __init__(
        self,
        latent_channels: int = 128,
        hidden: int = 128,
        score_scale: float = 1.0,
        gate_scale: float = 0.75,
    ):
        super().__init__()
        hidden = max(int(hidden), 16)
        self.score_scale = float(score_scale)
        self.gate_scale = float(gate_scale)
        self.latent_proj = nn.Sequential(
            nn.Linear(int(latent_channels), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.view_proj = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _latent_summary(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            return latent.mean(dim=(1, 2, 3))
        if latent.ndim == 5:
            return latent.mean(dim=(2, 3, 4))[0]
        raise ValueError("latent must have shape (C,T,H,W) or (B,C,T,H,W)")

    @staticmethod
    def _view_features(
        frames: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        del K
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape (V,H,W,3)")
        if masks.ndim != 4 or masks.shape[-1] != 1:
            raise ValueError("masks must have shape (V,H,W,1)")
        if depths.ndim != 3:
            raise ValueError("depths must have shape (V,H,W)")
        if frames.shape[:3] != masks.shape[:3] or frames.shape[:3] != depths.shape:
            raise ValueError("frames, masks, and depths must agree on V,H,W")
        v = frames.shape[0]
        device = frames.device
        dtype = frames.dtype
        radius_t = frames.new_tensor(max(float(radius), 1e-6))
        mask = masks[..., 0].to(dtype=dtype).clamp(0.0, 1.0)
        area = mask.mean(dim=(1, 2), keepdim=False).reshape(v, 1)
        denom = mask.sum(dim=(1, 2), keepdim=False).reshape(v, 1).clamp_min(1e-6)
        rgb_mean = (
            frames.clamp(0.0, 1.0) * mask[..., None]
        ).sum(dim=(1, 2)) / denom
        finite_depth = torch.isfinite(depths) & (depths > 1e-6) & (depths < 1e5)
        depth_weight = mask * finite_depth.to(dtype=dtype)
        depth_den = depth_weight.sum(dim=(1, 2), keepdim=False).reshape(v, 1).clamp_min(1e-6)
        valid_frac = depth_den / denom
        depth_norm = (depths.to(dtype=dtype) / radius_t).clamp(0.0, 8.0)
        depth_mean = (depth_norm * depth_weight).sum(dim=(1, 2)).reshape(v, 1) / depth_den
        depth_var = (
            ((depth_norm - depth_mean.reshape(v, 1, 1)) ** 2) * depth_weight
        ).sum(dim=(1, 2)).reshape(v, 1) / depth_den
        depth_std = torch.sqrt(depth_var.clamp_min(0.0))
        cam = c2w.to(device=device, dtype=dtype)
        origin = (cam[:, :3, 3] / radius_t).clamp(-4.0, 4.0) / 4.0
        forward = F.normalize(-cam[:, :3, 2], dim=-1)
        phase = (
            torch.arange(v, device=device, dtype=dtype).reshape(v, 1)
            * (2.0 * math.pi / max(float(v), 1.0))
        )
        return torch.cat([
            rgb_mean,
            area,
            valid_frac.clamp(0.0, 1.0),
            (depth_mean / 4.0).clamp(0.0, 1.0),
            (depth_std / 2.0).clamp(0.0, 1.0),
            origin,
            forward,
            torch.sin(phase),
            torch.cos(phase),
        ], dim=-1)

    @staticmethod
    def _prior_logits(
        n_views: int,
        device: torch.device,
        dtype: torch.dtype,
        prior_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        eps = 1e-4
        base = -torch.arange(
            n_views, device=device, dtype=dtype
        ) * (eps / max(float(n_views), 1.0)) - eps
        if prior_ids is None or prior_ids.numel() == 0:
            return base
        ids = prior_ids.to(device=device, dtype=torch.long).clamp(0, n_views - 1)
        rank = torch.arange(ids.numel(), device=device, dtype=dtype)
        base = base.clone()
        base[ids] = eps * (ids.numel() - rank)
        return base

    def forward(
        self,
        latent: torch.Tensor,
        frames: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        radius: float,
        prior_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        dtype = frames.dtype
        device = frames.device
        features = self._view_features(frames, masks, depths, K, c2w, radius)
        latent_ctx = self.latent_proj(
            self._latent_summary(latent).to(device=device, dtype=dtype)
        )
        hidden = self.view_proj(features) + latent_ctx.reshape(1, -1)
        raw = self.head(hidden)
        policy = torch.tanh(raw)
        logits = (
            self._prior_logits(frames.shape[0], device, dtype, prior_ids)
            + float(self.score_scale) * policy.reshape(-1)
        )
        if self.gate_scale <= 0:
            gate = torch.ones_like(raw)
        else:
            gate = (1.0 + float(self.gate_scale) * policy).clamp_min(0.0)
        return {"logits": logits, "gate": gate, "raw": raw, "features": features}

    def select(
        self,
        latent: torch.Tensor,
        frames: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        radius: float,
        k: int,
        prior_ids: torch.Tensor | None = None,
        train_noise: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(latent, frames, masks, depths, K, c2w, radius, prior_ids=prior_ids)
        n_views = frames.shape[0]
        k_eff = min(max(int(k), 1), n_views)
        logits = out["logits"]
        if self.training and train_noise > 0:
            logits = logits + torch.randn_like(logits) * float(train_noise)
        ids = torch.topk(logits, k_eff, largest=True, sorted=True).indices
        return {
            "ids": ids,
            "gates": out["gate"].index_select(0, ids),
            "logits": out["logits"],
            "raw": out["raw"],
            "features": out["features"],
        }


class RGBDSurfaceTokenDecoder(nn.Module):
    """Feed-forward learned Gaussian decoder from sampled RGBD surface tokens."""

    FEATURE_DIM = 20

    def __init__(
        self,
        latent_channels: int = 128,
        hidden: int = 256,
        slots: int = 256,
        layers: int = 3,
        heads: int = 8,
        latent_layers: int = 0,
        latent_pool: int = 1,
        latent_gate_init: float = 0.02,
        slot_refine_layers: int = 0,
        slot_refine_mlp_ratio: int = 4,
        slot_refine_gate_init: float = 0.02,
        grid_h: int = 48,
        grid_w: int = 64,
        mean_res_frac: float = 0.03,
        rgb_res_scale: float = 0.20,
        scale_frac: float = 0.003,
        normal_scale_frac: float = 0.0004,
        scale_res_scale: float = 1.0,
        quat_res_scale: float = 0.20,
        opacity_init: float = 0.80,
        checkpoint_blocks: bool = False,
        detail_layer: int = 0,
        detail_mean_res_frac: float = 0.012,
        detail_rgb_res_scale: float = 0.08,
        detail_scale_frac: float = 0.0012,
        detail_normal_scale_frac: float = 0.00012,
        detail_scale_res_scale: float = 1.0,
        detail_quat_res_scale: float = 0.12,
        detail_opacity_init: float = 0.04,
        source_rgb_dropout_prob: float = 0.0,
        learned_scale_base: bool = False,
        learned_scale_head: bool = False,
        learned_scale_min_frac: float = 1e-5,
        learned_scale_max_frac: float = 3e-2,
        learned_opacity_bias: bool = False,
        learned_opacity_prior: bool = False,
        learned_output_scales: bool = False,
        learned_color_affine: bool = False,
        color_affine_scale: float = 0.35,
        learned_policy_head: bool = False,
        policy_depth_res_frac: float = 0.0,
        policy_move_res_frac: float = 0.0,
        policy_scale_res_scale: float = 0.0,
        policy_opacity_res_scale: float = 0.0,
        policy_view_res_scale: float = 0.0,
        policy_confidence_res_scale: float = 0.0,
        policy_keep_res_scale: float = 0.0,
        policy_coverage_scale_res_scale: float = 0.0,
        policy_birth_res_scale: float = 0.0,
        learned_policy_output_scales: bool = False,
        learned_source_depth_confidence_head: bool = False,
        source_depth_res_frac: float = 0.0,
        source_confidence_res_scale: float = 0.0,
        learned_source_depth_confidence_scales: bool = False,
        proposal_count: int = 0,
        proposal_scale_frac: float = 0.0012,
        proposal_normal_scale_frac: float = 0.00012,
        proposal_scale_res_scale: float = 1.0,
        proposal_quat_res_scale: float = 0.25,
        proposal_rgb_res_scale: float = 0.5,
        proposal_extent_frac: float = 1.25,
        proposal_coverage_scale_res_scale: float = 0.75,
        proposal_opacity_init: float = 5e-4,
        proposal_seed_surface: bool = False,
        proposal_seed_pool: int = 1024,
        proposal_surface_res_frac: float = 0.04,
        proposal_anchor_mode: str = "even",
        proposal_anchor_temp: float = 0.25,
        proposal_anchor_local_window: int = 9,
        proposal_anchor_gate_init: float = 1e-4,
        proposal_anchor_mix_res_scale: float = 2.0,
        proposal_anchor_even_prior: float = 0.0,
        learned_proposal_policy_head: bool = False,
        proposal_policy_keep_res_scale: float = 0.0,
        proposal_policy_confidence_res_scale: float = 0.0,
        proposal_policy_coverage_res_scale: float = 0.0,
        learned_proposal_scale_base: bool = False,
        learned_proposal_scale_head: bool = False,
        learned_proposal_scale_min_frac: float = 1e-5,
        learned_proposal_scale_max_frac: float = 3e-2,
        depth_normal_quat: bool = False,
        depth_normal_blend: float = 1.0,
        learned_depth_normal_blend: bool = False,
        learned_depth_normal_blend_head: bool = False,
        depth_normal_blend_head_scale: float = 2.0,
    ):
        super().__init__()
        hidden = max(int(hidden), 32)
        slots = max(int(slots), 1)
        layers = max(int(layers), 1)
        heads = max(int(heads), 1)
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.grid_h = max(int(grid_h), 1)
        self.grid_w = max(int(grid_w), 1)
        self.latent_pool = max(int(latent_pool), 1)
        self.learned_scale_base = bool(learned_scale_base)
        self.learned_scale_head = bool(learned_scale_head)
        self.learned_scale_min_frac = max(float(learned_scale_min_frac), 1e-8)
        self.learned_scale_max_frac = max(
            float(learned_scale_max_frac), self.learned_scale_min_frac * 1.01
        )
        self.learned_opacity_bias = bool(learned_opacity_bias)
        self.learned_opacity_prior = bool(learned_opacity_prior)
        self.learned_output_scales = bool(learned_output_scales)
        self.learned_color_affine = bool(learned_color_affine)
        self.color_affine_scale = float(color_affine_scale)
        self.learned_policy_head = bool(learned_policy_head)
        self.policy_depth_res_frac = float(policy_depth_res_frac)
        self.policy_move_res_frac = float(policy_move_res_frac)
        self.policy_scale_res_scale = float(policy_scale_res_scale)
        self.policy_opacity_res_scale = float(policy_opacity_res_scale)
        self.policy_view_res_scale = float(policy_view_res_scale)
        self.policy_confidence_res_scale = float(policy_confidence_res_scale)
        self.policy_keep_res_scale = float(policy_keep_res_scale)
        self.policy_coverage_scale_res_scale = float(policy_coverage_scale_res_scale)
        self.policy_birth_res_scale = float(policy_birth_res_scale)
        self.learned_policy_output_scales = bool(learned_policy_output_scales)
        self.learned_source_depth_confidence_head = bool(learned_source_depth_confidence_head)
        self.source_depth_res_frac = float(source_depth_res_frac)
        self.source_confidence_res_scale = float(source_confidence_res_scale)
        self.learned_source_depth_confidence_scales = bool(
            learned_source_depth_confidence_scales
        )
        self.proposal_count = max(int(proposal_count), 0)
        self.proposal_scale_frac = float(proposal_scale_frac)
        self.proposal_normal_scale_frac = float(proposal_normal_scale_frac)
        self.proposal_scale_res_scale = float(proposal_scale_res_scale)
        self.proposal_quat_res_scale = float(proposal_quat_res_scale)
        self.proposal_rgb_res_scale = float(proposal_rgb_res_scale)
        self.proposal_extent_frac = float(proposal_extent_frac)
        self.proposal_coverage_scale_res_scale = float(proposal_coverage_scale_res_scale)
        self.proposal_opacity_init = min(
            max(float(proposal_opacity_init), 1e-8), 1.0 - 1e-8
        )
        self.proposal_seed_surface = bool(proposal_seed_surface)
        self.proposal_seed_pool = max(int(proposal_seed_pool), 1)
        self.proposal_surface_res_frac = float(proposal_surface_res_frac)
        self.proposal_anchor_mode = str(proposal_anchor_mode).strip().lower()
        if self.proposal_anchor_mode not in {
            "even",
            "learned_st",
            "learned_local_st",
            "learned_local_unique_st",
        }:
            raise ValueError(
                "proposal_anchor_mode must be 'even', 'learned_st', "
                "'learned_local_st', or 'learned_local_unique_st'"
            )
        self.proposal_anchor_temp = max(float(proposal_anchor_temp), 1e-4)
        self.proposal_anchor_local_window = max(int(proposal_anchor_local_window), 1)
        self.proposal_anchor_gate_init = min(
            max(float(proposal_anchor_gate_init), 1e-6), 1.0 - 1e-6
        )
        self.proposal_anchor_mix_res_scale = max(float(proposal_anchor_mix_res_scale), 0.0)
        self.proposal_anchor_even_prior = max(float(proposal_anchor_even_prior), 0.0)
        self.learned_proposal_policy_head = bool(learned_proposal_policy_head)
        self.proposal_policy_keep_res_scale = float(proposal_policy_keep_res_scale)
        self.proposal_policy_confidence_res_scale = float(
            proposal_policy_confidence_res_scale
        )
        self.proposal_policy_coverage_res_scale = float(
            proposal_policy_coverage_res_scale
        )
        self.learned_proposal_scale_base = bool(learned_proposal_scale_base)
        self.learned_proposal_scale_head = bool(learned_proposal_scale_head)
        self.learned_proposal_scale_min_frac = max(
            float(learned_proposal_scale_min_frac), 1e-8
        )
        self.learned_proposal_scale_max_frac = max(
            float(learned_proposal_scale_max_frac),
            self.learned_proposal_scale_min_frac * 1.01,
        )
        self.depth_normal_quat = bool(depth_normal_quat)
        self.depth_normal_blend = min(max(float(depth_normal_blend), 0.0), 1.0)
        if learned_depth_normal_blend:
            init = min(max(self.depth_normal_blend, 1e-4), 1.0 - 1e-4)
            self.depth_normal_blend_logit = nn.Parameter(
                torch.tensor(math.log(init / (1.0 - init)), dtype=torch.float32)
            )
        else:
            self.depth_normal_blend_logit = None
        self.learned_depth_normal_blend_head = bool(learned_depth_normal_blend_head)
        self.depth_normal_blend_head_scale = max(float(depth_normal_blend_head_scale), 0.0)
        self.mean_res_frac = float(mean_res_frac)
        self.rgb_res_scale = float(rgb_res_scale)
        self.scale_frac = float(scale_frac)
        self.normal_scale_frac = float(normal_scale_frac)
        self.scale_res_scale = float(scale_res_scale)
        self.quat_res_scale = float(quat_res_scale)
        self.opacity_init = float(opacity_init)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.detail_layer = bool(detail_layer)
        self.detail_mean_res_frac = float(detail_mean_res_frac)
        self.detail_rgb_res_scale = float(detail_rgb_res_scale)
        self.detail_scale_frac = float(detail_scale_frac)
        self.detail_normal_scale_frac = float(detail_normal_scale_frac)
        self.detail_scale_res_scale = float(detail_scale_res_scale)
        self.detail_quat_res_scale = float(detail_quat_res_scale)
        self.detail_opacity_init = float(detail_opacity_init)
        self.source_rgb_dropout_prob = float(source_rgb_dropout_prob)

        self.source_proj = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(int(latent_channels), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        latent_layers = max(int(latent_layers), 0)
        if latent_layers > 0:
            self.latent_token_proj = nn.Sequential(
                nn.Linear(int(latent_channels) + 6, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.latent_blocks = nn.ModuleList([
                _LatentSlotBlock(hidden, heads) for _ in range(latent_layers)
            ])
            init = min(max(float(latent_gate_init), 1e-4), 1.0 - 1e-4)
            self.latent_gate_logit = nn.Parameter(
                torch.tensor(math.log(init / (1.0 - init)), dtype=torch.float32)
            )
        else:
            self.latent_token_proj = None
            self.latent_blocks = nn.ModuleList()
            self.latent_gate_logit = None
        slot_refine_layers = max(int(slot_refine_layers), 0)
        if slot_refine_layers > 0:
            self.slot_refine_blocks = nn.ModuleList([
                _SlotSelfBlock(hidden, heads, mlp_ratio=slot_refine_mlp_ratio)
                for _ in range(slot_refine_layers)
            ])
            init = min(max(float(slot_refine_gate_init), 1e-4), 1.0 - 1e-4)
            self.slot_refine_gate_logit = nn.Parameter(
                torch.tensor(math.log(init / (1.0 - init)), dtype=torch.float32)
            )
        else:
            self.slot_refine_blocks = nn.ModuleList()
            self.slot_refine_gate_logit = None
        if self.learned_opacity_prior:
            self.opacity_prior = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.opacity_prior[-1].weight)
            nn.init.zeros_(self.opacity_prior[-1].bias)
        else:
            self.opacity_prior = None
        if self.learned_scale_base:
            init_scale = torch.tensor([
                max(float(self.scale_frac), 1e-8),
                max(float(self.scale_frac), 1e-8),
                max(float(self.normal_scale_frac), 1e-8),
            ], dtype=torch.float32)
            self.log_scale_base = nn.Parameter(init_scale.log())
        else:
            self.log_scale_base = None
        if self.learned_opacity_bias:
            init = min(max(self.opacity_init, 1e-4), 1.0 - 1e-4)
            self.opacity_logit_bias = nn.Parameter(
                torch.tensor(math.log(init / (1.0 - init)), dtype=torch.float32)
            )
        else:
            self.opacity_logit_bias = None
        if self.learned_output_scales:
            init_scales = torch.tensor([
                max(float(self.mean_res_frac), 1e-8),
                max(float(self.rgb_res_scale), 1e-8),
                max(float(self.scale_res_scale), 1e-8),
                max(float(self.quat_res_scale), 1e-8),
            ], dtype=torch.float32)
            self.log_output_scales = nn.Parameter(init_scales.log())
        else:
            self.log_output_scales = None
        if self.learned_color_affine:
            self.color_affine = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 6),
            )
            nn.init.zeros_(self.color_affine[-1].weight)
            nn.init.zeros_(self.color_affine[-1].bias)
        else:
            self.color_affine = None
        if self.learned_policy_head:
            # 13 policy channels:
            # depth, xyz move, xyz scale, opacity, view, confidence, keep,
            # tangent coverage, detail birth. The final layer is zero-init so
            # enabling this head preserves the RGBD scaffold exactly at step 0.
            self.policy_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 13),
            )
            nn.init.zeros_(self.policy_head[-1].weight)
            nn.init.zeros_(self.policy_head[-1].bias)
            if self.learned_policy_output_scales:
                init_policy_scales = torch.tensor([
                    max(abs(float(self.policy_depth_res_frac)), 1e-8),
                    max(abs(float(self.policy_move_res_frac)), 1e-8),
                    max(abs(float(self.policy_scale_res_scale)), 1e-8),
                    max(abs(float(self.policy_opacity_res_scale)), 1e-8),
                    max(abs(float(self.policy_view_res_scale)), 1e-8),
                    max(abs(float(self.policy_confidence_res_scale)), 1e-8),
                    max(abs(float(self.policy_keep_res_scale)), 1e-8),
                    max(abs(float(self.policy_coverage_scale_res_scale)), 1e-8),
                    max(abs(float(self.policy_birth_res_scale)), 1e-8),
                ], dtype=torch.float32)
                self.policy_log_output_scales = nn.Parameter(init_policy_scales.log())
            else:
                self.policy_log_output_scales = None
        else:
            self.policy_head = None
            self.policy_log_output_scales = None
        if self.learned_source_depth_confidence_head:
            self.source_depth_confidence_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2),
            )
            nn.init.zeros_(self.source_depth_confidence_head[-1].weight)
            nn.init.zeros_(self.source_depth_confidence_head[-1].bias)
            if self.learned_source_depth_confidence_scales:
                init_source_scales = torch.tensor([
                    max(abs(float(self.source_depth_res_frac)), 1e-8),
                    max(abs(float(self.source_confidence_res_scale)), 1e-8),
                ], dtype=torch.float32)
                self.source_depth_confidence_log_scales = nn.Parameter(
                    init_source_scales.log()
                )
            else:
                self.source_depth_confidence_log_scales = None
        else:
            self.source_depth_confidence_head = None
            self.source_depth_confidence_log_scales = None
        if self.learned_depth_normal_blend_head:
            self.depth_normal_blend_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.depth_normal_blend_head[-1].weight)
            nn.init.zeros_(self.depth_normal_blend_head[-1].bias)
        else:
            self.depth_normal_blend_head = None
        self.slots = nn.Parameter(torch.randn(1, slots, hidden) * 0.02)
        self.blocks = nn.ModuleList([
            _SlotSurfaceBlock(hidden, heads) for _ in range(layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 14),
        )
        # Identity-ish initialization: start from RGBD points/colors, with the
        # requested opacity prior. Training then moves geometry/color/opacity.
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.detail_head = None
        if self.detail_layer:
            self.detail_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 14),
            )
            last_detail = self.detail_head[-1]
            nn.init.zeros_(last_detail.weight)
            nn.init.zeros_(last_detail.bias)
        if self.proposal_count > 0:
            self.proposal_queries = nn.Parameter(
                torch.randn(1, self.proposal_count, hidden) * 0.02
            )
            self.proposal_base_mean = nn.Parameter(
                _proposal_base_grid(
                    self.proposal_count,
                    float(self.proposal_extent_frac) * 0.75,
                )
            )
            self.proposal_from_slots = nn.MultiheadAttention(hidden, heads, batch_first=True)
            if self.proposal_seed_surface:
                self.proposal_from_surface = nn.MultiheadAttention(hidden, heads, batch_first=True)
                self.proposal_surface_norm_q = nn.LayerNorm(hidden)
                self.proposal_surface_norm_kv = nn.LayerNorm(hidden)
                if self.proposal_anchor_mode in {
                    "learned_st",
                    "learned_local_st",
                    "learned_local_unique_st",
                }:
                    self.proposal_anchor_q = nn.Sequential(
                        nn.LayerNorm(hidden),
                        nn.Linear(hidden, hidden),
                    )
                    self.proposal_anchor_k = nn.Sequential(
                        nn.LayerNorm(hidden),
                        nn.Linear(hidden, hidden),
                    )
                    self.proposal_anchor_mix_logit = nn.Parameter(
                        torch.tensor(
                            math.log(
                                self.proposal_anchor_gate_init
                                / (1.0 - self.proposal_anchor_gate_init)
                            ),
                            dtype=torch.float32,
                        )
                    )
                    self.proposal_anchor_temp_log = nn.Parameter(
                        torch.tensor(math.log(self.proposal_anchor_temp), dtype=torch.float32)
                    )
                    self.proposal_anchor_mix_head = nn.Sequential(
                        nn.LayerNorm(hidden),
                        nn.Linear(hidden, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, 1),
                    )
                    nn.init.zeros_(self.proposal_anchor_mix_head[-1].weight)
                    nn.init.zeros_(self.proposal_anchor_mix_head[-1].bias)
                else:
                    self.proposal_anchor_q = None
                    self.proposal_anchor_k = None
                    self.proposal_anchor_mix_logit = None
                    self.proposal_anchor_temp_log = None
                    self.proposal_anchor_mix_head = None
            else:
                self.proposal_from_surface = None
                self.proposal_surface_norm_q = None
                self.proposal_surface_norm_kv = None
                self.proposal_anchor_q = None
                self.proposal_anchor_k = None
                self.proposal_anchor_mix_logit = None
                self.proposal_anchor_temp_log = None
                self.proposal_anchor_mix_head = None
            self.proposal_norm1 = nn.LayerNorm(hidden)
            self.proposal_norm2 = nn.LayerNorm(hidden)
            self.proposal_ff = nn.Sequential(
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Linear(hidden * 4, hidden),
            )
            self.proposal_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 15),
            )
            if self.learned_proposal_policy_head:
                self.proposal_policy_head = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 3),
                )
                nn.init.zeros_(self.proposal_policy_head[-1].weight)
                nn.init.zeros_(self.proposal_policy_head[-1].bias)
            else:
                self.proposal_policy_head = None
            self.proposal_opacity_logit_bias = nn.Parameter(
                torch.tensor(
                    math.log(self.proposal_opacity_init / (1.0 - self.proposal_opacity_init)),
                    dtype=torch.float32,
                )
            )
            if self.learned_proposal_scale_base:
                init_scale = torch.tensor([
                    max(float(self.proposal_scale_frac), 1e-8),
                    max(float(self.proposal_scale_frac), 1e-8),
                    max(float(self.proposal_normal_scale_frac), 1e-8),
                ], dtype=torch.float32)
                self.proposal_log_scale_base = nn.Parameter(init_scale.log())
            else:
                self.proposal_log_scale_base = None
            nn.init.zeros_(self.proposal_from_slots.out_proj.weight)
            nn.init.zeros_(self.proposal_from_slots.out_proj.bias)
            nn.init.zeros_(self.proposal_ff[-1].weight)
            nn.init.zeros_(self.proposal_ff[-1].bias)
            nn.init.normal_(self.proposal_head[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.proposal_head[-1].bias)
            if self.learned_proposal_scale_head:
                lo = math.log(self.learned_proposal_scale_min_frac)
                hi = math.log(self.learned_proposal_scale_max_frac)
                init = [
                    max(float(self.proposal_scale_frac), 1e-8),
                    max(float(self.proposal_scale_frac), 1e-8),
                    max(float(self.proposal_normal_scale_frac), 1e-8),
                ]
                bias = []
                for frac in init:
                    u = (math.log(frac) - lo) / max(hi - lo, 1e-8)
                    u = min(max(u, 1e-4), 1.0 - 1e-4)
                    bias.append(math.log(u / (1.0 - u)))
                with torch.no_grad():
                    self.proposal_head[-1].bias[6:9] = torch.tensor(
                        bias,
                        dtype=self.proposal_head[-1].bias.dtype,
                    )
        else:
            self.proposal_queries = None
            self.proposal_base_mean = None
            self.proposal_from_slots = None
            self.proposal_from_surface = None
            self.proposal_surface_norm_q = None
            self.proposal_surface_norm_kv = None
            self.proposal_anchor_q = None
            self.proposal_anchor_k = None
            self.proposal_anchor_mix_logit = None
            self.proposal_anchor_temp_log = None
            self.proposal_anchor_mix_head = None
            self.proposal_norm1 = None
            self.proposal_norm2 = None
            self.proposal_ff = None
            self.proposal_head = None
            self.proposal_policy_head = None
            self.proposal_opacity_logit_bias = None
            self.proposal_log_scale_base = None

    def _latent_tokens(
        self,
        latent: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.latent_token_proj is None:
            raise RuntimeError("latent token projection is disabled")
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
        return self.latent_token_proj(torch.cat([vals, pos], dim=-1))[None]

    def _output_scale(
        self,
        idx: int,
        fallback: float,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if self.log_output_scales is None:
            return ref.new_tensor(float(fallback))
        val = torch.exp(self.log_output_scales[idx].to(device=ref.device, dtype=ref.dtype))
        return val.clamp(1e-6, 2.0)

    def _policy_output_scale(
        self,
        idx: int,
        fallback: float,
        ref: torch.Tensor,
        max_value: float,
    ) -> torch.Tensor:
        if self.policy_log_output_scales is None:
            return ref.new_tensor(float(fallback))
        val = torch.exp(
            self.policy_log_output_scales[idx].to(device=ref.device, dtype=ref.dtype)
        )
        return val.clamp(1e-8, max(float(max_value), 1e-8))

    def _source_depth_confidence_scale(
        self,
        idx: int,
        fallback: float,
        ref: torch.Tensor,
        max_value: float,
    ) -> torch.Tensor:
        if self.source_depth_confidence_log_scales is None:
            return ref.new_tensor(float(fallback))
        val = torch.exp(
            self.source_depth_confidence_log_scales[idx].to(
                device=ref.device, dtype=ref.dtype
            )
        )
        return val.clamp(1e-8, max(float(max_value), 1e-8))

    def _maybe_drop_source_rgb(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if not self.training or self.source_rgb_dropout_prob <= 0.0:
            return frames
        p = min(max(self.source_rgb_dropout_prob, 0.0), 1.0)
        drop = torch.rand(
            (frames.shape[0], 1, 1, 1), device=frames.device, dtype=frames.dtype
        ) < p
        if not bool(drop.any()):
            return frames
        weight = masks.clamp(0.0, 1.0)
        denom = weight.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        mean_rgb = (frames * weight).sum(dim=(1, 2), keepdim=True) / denom
        return torch.where(drop, mean_rgb.expand_as(frames), frames)

    @staticmethod
    def _symmetric_gate(raw: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(scale):
            scale_t = scale.to(device=raw.device, dtype=raw.dtype)
        else:
            scale_t = raw.new_tensor(float(scale))
        return (1.0 + scale_t * torch.tanh(raw)).clamp_min(0.0)

    def _proposal_params(
        self,
        slots: torch.Tensor,
        radius_t: torch.Tensor,
        surface_tokens: torch.Tensor | None = None,
        base: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if self.proposal_head is None:
            return None
        q = self.proposal_queries.to(device=slots.device, dtype=slots.dtype)
        msg, _ = self.proposal_from_slots(
            self.proposal_norm1(q), slots, slots, need_weights=False
        )
        prop = q + msg
        anchor_mean = None
        anchor_rgb = None
        anchor_quat = None
        anchor_depth = None
        anchor_mix = None
        anchor_entropy = None
        anchor_entropy_loss = None
        anchor_usage_loss = None
        anchor_usage_perplexity = None
        anchor_unique_frac = None
        anchor_collision_loss = None
        anchor_collision_frac = None
        seed_valid = None
        if (
            self.proposal_from_surface is not None
            and surface_tokens is not None
            and base is not None
        ):
            valid = (base["valid"] * base["mask"]).reshape(-1) > 0.5
            idx = valid.nonzero(as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                idx = torch.arange(
                    surface_tokens.shape[0],
                    device=surface_tokens.device,
                    dtype=torch.long,
                )
            if idx.numel() > self.proposal_seed_pool:
                take = torch.linspace(
                    0,
                    idx.numel() - 1,
                    self.proposal_seed_pool,
                    device=idx.device,
                ).round().to(dtype=torch.long)
                idx = idx.index_select(0, take)
            cand_tokens = surface_tokens.index_select(0, idx)[None]
            seed_msg, _ = self.proposal_from_surface(
                self.proposal_surface_norm_q(q),
                self.proposal_surface_norm_kv(cand_tokens),
                self.proposal_surface_norm_kv(cand_tokens),
                need_weights=False,
            )
            prop = prop + seed_msg
            cand_mean = base["mean"].index_select(0, idx)
            cand_rgb = base["rgb"].index_select(0, idx)
            cand_quat = base["quat"].index_select(0, idx)
            cand_depth = base["raydist"].index_select(0, idx)
            if idx.numel() >= q.shape[1]:
                pick = torch.linspace(
                    0,
                    idx.numel() - 1,
                    q.shape[1],
                    device=idx.device,
                ).round().to(dtype=torch.long)
            else:
                pick = torch.arange(q.shape[1], device=idx.device, dtype=torch.long)
                pick = pick.remainder(idx.numel())
            even_mean = cand_mean.index_select(0, pick)
            even_rgb = cand_rgb.index_select(0, pick).clamp(0.0, 1.0)
            even_quat = F.normalize(cand_quat.index_select(0, pick), dim=-1)
            even_depth = cand_depth.index_select(0, pick)
            if self.proposal_anchor_q is not None and self.proposal_anchor_k is not None:
                q_anchor = self.proposal_anchor_q(prop).to(dtype=prop.dtype)
                k_anchor = self.proposal_anchor_k(cand_tokens).to(dtype=prop.dtype)
                logits = torch.matmul(
                    q_anchor,
                    k_anchor.transpose(-2, -1),
                )[0] / math.sqrt(float(q_anchor.shape[-1]))
                temp = torch.exp(
                    self.proposal_anchor_temp_log.to(device=logits.device, dtype=logits.dtype)
                ).clamp(1e-3, 10.0)
                cand_count = logits.shape[-1]
                if self.proposal_anchor_mode in {
                    "learned_local_st",
                    "learned_local_unique_st",
                }:
                    window = min(max(int(self.proposal_anchor_local_window), 1), cand_count)
                    if self.proposal_anchor_mode == "learned_local_unique_st":
                        offset_vals: list[int] = [0]
                        for delta in range(1, window):
                            if len(offset_vals) >= window:
                                break
                            offset_vals.append(-delta)
                            if len(offset_vals) >= window:
                                break
                            offset_vals.append(delta)
                        offsets = torch.tensor(offset_vals, device=idx.device, dtype=torch.long)
                        center = 0
                    else:
                        offsets = torch.arange(window, device=idx.device, dtype=torch.long)
                        offsets = offsets - (window // 2)
                        center = window // 2
                    local_idx = (pick[:, None] + offsets[None, :]).clamp(0, cand_count - 1)
                    local_logits = logits.gather(1, local_idx)
                    if self.proposal_anchor_even_prior > 0:
                        local_prior = torch.zeros_like(local_logits)
                        local_prior[:, center:center + 1] = float(self.proposal_anchor_even_prior)
                        local_logits = local_logits + local_prior
                    weights = torch.softmax(local_logits / temp, dim=-1)
                    if self.proposal_anchor_mode == "learned_local_unique_st":
                        scores_cpu = local_logits.detach().float().cpu()
                        local_idx_cpu = local_idx.detach().cpu()
                        tie = torch.arange(
                            scores_cpu.shape[1],
                            device=scores_cpu.device,
                            dtype=scores_cpu.dtype,
                        ) * -1e-7
                        used: set[int] = set()
                        hard_cols: list[int] = []
                        for row in range(scores_cpu.shape[0]):
                            order = torch.argsort(scores_cpu[row] + tie, descending=True)
                            fallback = int(order[0])
                            chosen = fallback
                            for col_t in order:
                                col = int(col_t)
                                cand = int(local_idx_cpu[row, col])
                                if cand not in used:
                                    chosen = col
                                    used.add(cand)
                                    break
                            else:
                                used.add(int(local_idx_cpu[row, fallback]))
                            hard_local_idx = chosen
                            hard_cols.append(hard_local_idx)
                        hard_local_idx = torch.tensor(
                            hard_cols,
                            device=weights.device,
                            dtype=torch.long,
                        )
                    else:
                        hard_local_idx = weights.argmax(dim=-1)
                    hard = F.one_hot(
                        hard_local_idx,
                        num_classes=weights.shape[-1],
                    ).to(dtype=weights.dtype)
                    st_weights = hard + weights - weights.detach()
                    cand_mean_local = cand_mean.index_select(0, local_idx.reshape(-1)).reshape(
                        local_idx.shape[0], local_idx.shape[1], -1
                    ).to(device=weights.device, dtype=weights.dtype)
                    cand_rgb_local = cand_rgb.index_select(0, local_idx.reshape(-1)).reshape(
                        local_idx.shape[0], local_idx.shape[1], -1
                    ).to(device=weights.device, dtype=weights.dtype)
                    cand_quat_local = cand_quat.index_select(0, local_idx.reshape(-1)).reshape(
                        local_idx.shape[0], local_idx.shape[1], -1
                    ).to(device=weights.device, dtype=weights.dtype)
                    cand_depth_local = cand_depth.index_select(0, local_idx.reshape(-1)).reshape(
                        local_idx.shape[0], local_idx.shape[1], -1
                    ).to(device=weights.device, dtype=weights.dtype)
                    learned_mean = (st_weights[..., None] * cand_mean_local).sum(dim=1)
                    learned_rgb = (st_weights[..., None] * cand_rgb_local).sum(dim=1).clamp(0.0, 1.0)
                    learned_quat = F.normalize(
                        (st_weights[..., None] * cand_quat_local).sum(dim=1),
                        dim=-1,
                    )
                    learned_depth = (st_weights[..., None] * cand_depth_local).sum(dim=1)
                    hard_anchor_idx = local_idx.gather(1, hard_local_idx[:, None]).reshape(-1)
                    usage = logits.new_zeros((cand_count,), dtype=torch.float32)
                    usage = usage.scatter_add(
                        0,
                        local_idx.reshape(-1),
                        weights.float().reshape(-1),
                    ) / max(float(weights.shape[0]), 1.0)
                    entropy_den = math.log(max(float(weights.shape[-1]), 2.0))
                else:
                    if self.proposal_anchor_even_prior > 0:
                        even_prior = F.one_hot(
                            pick.clamp_max(cand_count - 1),
                            num_classes=cand_count,
                        ).to(device=logits.device, dtype=logits.dtype)
                        logits = logits + even_prior * float(self.proposal_anchor_even_prior)
                    weights = torch.softmax(logits / temp, dim=-1)
                    hard_idx = weights.argmax(dim=-1)
                    hard = F.one_hot(
                        hard_idx,
                        num_classes=weights.shape[-1],
                    ).to(dtype=weights.dtype)
                    st_weights = hard + weights - weights.detach()
                    learned_mean = st_weights @ cand_mean.to(device=weights.device, dtype=weights.dtype)
                    learned_rgb = (
                        st_weights @ cand_rgb.to(device=weights.device, dtype=weights.dtype)
                    ).clamp(0.0, 1.0)
                    learned_quat = F.normalize(
                        st_weights @ cand_quat.to(device=weights.device, dtype=weights.dtype),
                        dim=-1,
                    )
                    learned_depth = st_weights @ cand_depth.to(device=weights.device, dtype=weights.dtype)
                    hard_anchor_idx = hard_idx
                    usage = weights.float().mean(dim=0)
                    entropy_den = math.log(max(float(weights.shape[-1]), 2.0))
                mix_logit = self.proposal_anchor_mix_logit.to(
                    device=weights.device, dtype=weights.dtype
                )
                if self.proposal_anchor_mix_head is not None:
                    mix_logit = mix_logit + (
                        self.proposal_anchor_mix_head(prop[0])
                        * float(self.proposal_anchor_mix_res_scale)
                    )
                mix = torch.sigmoid(mix_logit)
                anchor_mean = even_mean.to(dtype=weights.dtype) + mix * (
                    learned_mean - even_mean.to(dtype=weights.dtype)
                )
                anchor_rgb = even_rgb.to(dtype=weights.dtype) + mix * (
                    learned_rgb - even_rgb.to(dtype=weights.dtype)
                )
                anchor_quat = F.normalize(
                    even_quat.to(dtype=weights.dtype) + mix * (
                        learned_quat - even_quat.to(dtype=weights.dtype)
                    ),
                    dim=-1,
                )
                anchor_depth = even_depth.to(dtype=weights.dtype) + mix * (
                    learned_depth - even_depth.to(dtype=weights.dtype)
                )
                anchor_entropy = (
                    -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
                    / entropy_den
                )
                usage_loss = (
                    usage * (usage * float(cand_count)).clamp_min(1e-8).log()
                ).sum()
                expected_count = usage.to(dtype=weights.dtype) * float(weights.shape[0])
                anchor_collision_loss = (
                    F.relu(expected_count - 1.0).square().sum()
                    / max(float(weights.shape[0]), 1.0)
                )
                usage_entropy = -(usage * usage.clamp_min(1e-8).log()).sum()
                anchor_usage_perplexity = (
                    torch.exp(usage_entropy) / float(cand_count)
                ).reshape(())
                unique = hard_anchor_idx.unique().numel()
                denom_unique = max(1.0, float(min(hard_anchor_idx.shape[0], cand_count)))
                anchor_unique_frac = unique / denom_unique
                anchor_collision_frac = 1.0 - anchor_unique_frac
                anchor_mix = mix.reshape(q.shape[1], 1)
                anchor_entropy_loss = anchor_entropy.mean()
                anchor_usage_loss = usage_loss.to(device=weights.device, dtype=weights.dtype)
            else:
                anchor_mean = even_mean
                anchor_rgb = even_rgb
                anchor_quat = even_quat
                anchor_depth = even_depth
            seed_valid = torch.ones((q.shape[1], 1), device=q.device, dtype=q.dtype)
        prop = prop + self.proposal_ff(self.proposal_norm2(prop))
        raw = self.proposal_head(prop[0])
        proposal_keep_gate = raw.new_ones((raw.shape[0], 1))
        proposal_confidence_gate = raw.new_ones((raw.shape[0], 1))
        proposal_policy_coverage_mult = raw.new_ones((raw.shape[0], 1))
        if self.proposal_policy_head is not None:
            policy = self.proposal_policy_head(prop[0]).to(dtype=raw.dtype)
            proposal_keep_gate = self._symmetric_gate(
                policy[:, 0:1], self.proposal_policy_keep_res_scale
            )
            proposal_confidence_gate = self._symmetric_gate(
                policy[:, 1:2], self.proposal_policy_confidence_res_scale
            )
            proposal_policy_coverage_mult = torch.exp(
                torch.tanh(policy[:, 2:3])
                * float(self.proposal_policy_coverage_res_scale)
            )
        if anchor_mean is None:
            base_mean = self.proposal_base_mean.to(device=raw.device, dtype=raw.dtype) * radius_t
            move_scale = float(self.proposal_extent_frac) * radius_t
            rgb = torch.sigmoid(raw[:, 3:6] * float(self.proposal_rgb_res_scale))
            quat_base = raw.new_zeros((raw.shape[0], 4))
            quat_base[:, 0] = 1.0
            depth = raw.new_zeros((raw.shape[0], 1))
            seed_flag = raw.new_zeros((raw.shape[0], 1))
        else:
            base_mean = anchor_mean.to(device=raw.device, dtype=raw.dtype)
            move_scale = float(self.proposal_surface_res_frac) * radius_t
            rgb = (
                anchor_rgb.to(device=raw.device, dtype=raw.dtype)
                + torch.tanh(raw[:, 3:6]) * float(self.proposal_rgb_res_scale)
            ).clamp(0.0, 1.0)
            quat_base = anchor_quat.to(device=raw.device, dtype=raw.dtype)
            depth = anchor_depth.to(device=raw.device, dtype=raw.dtype)
        seed_flag = (
            seed_valid.to(device=raw.device, dtype=raw.dtype)
            if seed_valid is not None
            else torch.zeros((raw.shape[0], 1), device=raw.device, dtype=raw.dtype)
        )
        mean_offset = torch.tanh(raw[:, 0:3]) * move_scale
        mean = base_mean + mean_offset
        coverage_mult = torch.exp(
            torch.tanh(raw[:, 14:15]) * float(self.proposal_coverage_scale_res_scale)
        ) * proposal_policy_coverage_mult
        coverage_vec = torch.cat([
            coverage_mult,
            coverage_mult,
            torch.ones_like(coverage_mult),
        ], dim=-1)
        if self.learned_proposal_scale_head:
            lo = math.log(self.learned_proposal_scale_min_frac)
            hi = math.log(self.learned_proposal_scale_max_frac)
            scale_frac = torch.exp(raw.new_tensor(lo) + torch.sigmoid(raw[:, 6:9]) * (hi - lo))
            scale = (radius_t * scale_frac * coverage_vec).clamp_min(
                self.learned_proposal_scale_min_frac * radius_t
            )
        else:
            if self.proposal_log_scale_base is not None:
                base_scale = radius_t * torch.exp(
                    self.proposal_log_scale_base.to(device=raw.device, dtype=raw.dtype)
                ).reshape(1, 3)
            else:
                tangent = max(float(self.proposal_scale_frac), 1e-8) * radius_t
                normal = max(float(self.proposal_normal_scale_frac), 1e-8) * radius_t
                base_scale = torch.stack([tangent, tangent, normal]).reshape(1, 3)
            scale = (
                base_scale
                * torch.exp(torch.tanh(raw[:, 6:9]) * float(self.proposal_scale_res_scale))
                * coverage_vec
            ).clamp_min(1e-5 * radius_t)
        quat = F.normalize(
            quat_base + float(self.proposal_quat_res_scale) * torch.tanh(raw[:, 9:13]),
            dim=-1,
        )
        # Start nearly invisible, but keep a real derivative into proposal
        # opacity. Exact zero opacity severs the render-loss path for free
        # proposals because the lower clamp has zero gradient at the boundary.
        opacity = torch.sigmoid(
            raw[:, 13:14]
            + self.proposal_opacity_logit_bias.to(device=raw.device, dtype=raw.dtype)
        )
        opacity = (opacity * proposal_keep_gate * proposal_confidence_gate).clamp(0.0, 1.0)
        valid = (
            seed_valid.to(device=raw.device, dtype=raw.dtype)
            if seed_valid is not None
            else torch.ones_like(opacity)
        )
        return {
            "mean": mean,
            "quat": quat,
            "scale": scale,
            "opacity": opacity,
            "rgb": rgb,
            "depth": depth,
            "mean_anchor": base_mean,
            "mean_offset": mean_offset,
            "scale_raw": scale,
            "_surface_token_mask": valid,
            "_surface_token_valid": valid,
            "_surface_token_proposal": valid,
            "_surface_token_proposal_coverage_mult": coverage_mult,
            "_surface_token_proposal_policy_keep_gate": proposal_keep_gate,
            "_surface_token_proposal_policy_confidence_gate": proposal_confidence_gate,
            "_surface_token_proposal_policy_coverage_mult": proposal_policy_coverage_mult,
            "_surface_token_proposal_surface_seed": seed_flag,
            "_surface_token_proposal_anchor_mix": (
                anchor_mix.to(device=raw.device, dtype=raw.dtype)
                if anchor_mix is not None
                else raw.new_zeros((raw.shape[0], 1))
            ),
            "_surface_token_proposal_anchor_entropy": (
                anchor_entropy.to(device=raw.device, dtype=raw.dtype)
                if anchor_entropy is not None
                else raw.new_zeros((raw.shape[0], 1))
            ),
            "_surface_token_proposal_anchor_entropy_loss": (
                anchor_entropy_loss.to(device=raw.device, dtype=raw.dtype)
                if anchor_entropy_loss is not None
                else raw.new_zeros(())
            ),
            "_surface_token_proposal_anchor_usage_loss": (
                anchor_usage_loss.to(device=raw.device, dtype=raw.dtype)
                if anchor_usage_loss is not None
                else raw.new_zeros(())
            ),
            "_surface_token_proposal_anchor_usage_perplexity": (
                anchor_usage_perplexity.to(device=raw.device, dtype=raw.dtype)
                if anchor_usage_perplexity is not None
                else raw.new_zeros(())
            ),
            "_surface_token_proposal_anchor_unique_frac": (
                raw.new_tensor(float(anchor_unique_frac))
                if anchor_unique_frac is not None
                else raw.new_zeros(())
            ),
            "_surface_token_proposal_anchor_collision_loss": (
                anchor_collision_loss.to(device=raw.device, dtype=raw.dtype)
                if anchor_collision_loss is not None
                else raw.new_zeros(())
            ),
            "_surface_token_proposal_anchor_collision_frac": (
                raw.new_tensor(float(anchor_collision_frac))
                if anchor_collision_frac is not None
                else raw.new_zeros(())
            ),
        }

    def forward(
        self,
        latent: torch.Tensor,
        frames: torch.Tensor,
        masks: torch.Tensor,
        depths: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        radius: float,
        disable_new_capacity: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not disable_new_capacity:
            frames = self._maybe_drop_source_rgb(frames, masks)
        if self.depth_normal_blend_logit is not None and not disable_new_capacity:
            depth_normal_blend = torch.sigmoid(
                self.depth_normal_blend_logit.to(device=frames.device, dtype=frames.dtype)
            )
        else:
            depth_normal_blend = self.depth_normal_blend
        features, base = build_rgbd_surface_tokens(
            frames,
            masks,
            depths,
            K,
            c2w,
            radius,
            self.grid_h,
            self.grid_w,
            use_depth_normals=self.depth_normal_quat,
            depth_normal_blend=depth_normal_blend,
        )
        src = self.source_proj(features)[None]
        if latent.ndim == 4:
            latent_summary = latent.mean(dim=(1, 2, 3))
        elif latent.ndim == 5:
            latent_summary = latent.mean(dim=(2, 3, 4))[0]
        else:
            raise ValueError("latent must have shape (C,T,H,W) or (B,C,T,H,W)")
        latent_ctx = self.latent_proj(latent_summary.to(device=frames.device, dtype=frames.dtype))
        src = src + latent_ctx.reshape(1, 1, -1)
        slots = self.slots.to(device=frames.device, dtype=frames.dtype) + latent_ctx.reshape(1, 1, -1)
        if self.latent_blocks and not disable_new_capacity:
            latent_tokens = self._latent_tokens(latent, frames.dtype, frames.device)
            gate = torch.sigmoid(self.latent_gate_logit).to(device=frames.device, dtype=frames.dtype)
            for block in self.latent_blocks:
                if self.training and self.checkpoint_blocks:
                    from torch.utils.checkpoint import checkpoint
                    slots_next = checkpoint(block, slots, latent_tokens, use_reentrant=False)
                else:
                    slots_next = block(slots, latent_tokens)
                slots = slots + gate * (slots_next - slots)
        if self.slot_refine_blocks and not disable_new_capacity:
            gate = torch.sigmoid(self.slot_refine_gate_logit).to(
                device=frames.device, dtype=frames.dtype
            )
            for block in self.slot_refine_blocks:
                if self.training and self.checkpoint_blocks:
                    from torch.utils.checkpoint import checkpoint
                    slots_next = checkpoint(block, slots, use_reentrant=False)
                else:
                    slots_next = block(slots)
                slots = slots + gate * (slots_next - slots)
        for block in self.blocks:
            if self.training and self.checkpoint_blocks:
                from torch.utils.checkpoint import checkpoint
                src, slots = checkpoint(block, src, slots, use_reentrant=False)
            else:
                src, slots = block(src, slots)
        raw = self.head(src[0])
        policy_raw = None
        if self.policy_head is not None and not disable_new_capacity:
            policy_raw = self.policy_head(src[0])
        source_base_mean = base["mean"]
        source_direction = base["direction"]
        source_depth_res = raw.new_zeros((raw.shape[0], 1))
        source_confidence_gate = raw.new_ones((raw.shape[0], 1))
        if self.source_depth_confidence_head is not None and not disable_new_capacity:
            source_raw = self.source_depth_confidence_head(src[0]).to(dtype=raw.dtype)
            source_depth_scale = self._source_depth_confidence_scale(
                0, self.source_depth_res_frac, raw, max_value=0.25
            )
            source_confidence_scale = self._source_depth_confidence_scale(
                1, self.source_confidence_res_scale, raw, max_value=4.0
            )
            source_depth_res = torch.tanh(source_raw[:, 0:1]) * (
                source_depth_scale * frames.new_tensor(max(float(radius), 1e-6))
            )
            source_confidence_gate = self._symmetric_gate(
                source_raw[:, 1:2], source_confidence_scale
            )
            base = dict(base)
            base["mean"] = base["mean"] + base["direction"] * source_depth_res
            base["raydist"] = (base["raydist"] + source_depth_res).clamp_min(1e-6)

        radius_t = frames.new_tensor(max(float(radius), 1e-6))
        if (
            self.depth_normal_quat
            and self.depth_normal_blend_head is not None
            and not disable_new_capacity
        ):
            if self.depth_normal_blend_logit is not None:
                base_blend_logit = self.depth_normal_blend_logit.to(
                    device=raw.device, dtype=raw.dtype
                )
            else:
                init = min(max(float(self.depth_normal_blend), 1e-4), 1.0 - 1e-4)
                base_blend_logit = raw.new_tensor(math.log(init / (1.0 - init)))
            blend_raw = self.depth_normal_blend_head(src[0]).to(dtype=raw.dtype)
            depth_normal_blend_tok = torch.sigmoid(
                base_blend_logit
                + blend_raw * float(self.depth_normal_blend_head_scale)
            )
            ray_normal = base["ray_normal"].to(device=raw.device, dtype=raw.dtype)
            depth_normal = base["depth_normal"].to(device=raw.device, dtype=raw.dtype)
            normal = F.normalize(
                ray_normal + depth_normal_blend_tok * (depth_normal - ray_normal),
                dim=-1,
            )
            base["quat"] = _quat_from_normals(normal)
        else:
            if torch.is_tensor(depth_normal_blend):
                depth_normal_blend_tok = depth_normal_blend.reshape(1, 1).expand(
                    raw.shape[0], 1
                ).to(device=raw.device, dtype=raw.dtype)
            else:
                depth_normal_blend_tok = raw.new_full(
                    (raw.shape[0], 1),
                    float(depth_normal_blend) if self.depth_normal_quat else 0.0,
                )
        if disable_new_capacity:
            mean_res_frac = raw.new_tensor(float(self.mean_res_frac))
            rgb_res_scale = raw.new_tensor(float(self.rgb_res_scale))
            scale_res_scale = raw.new_tensor(float(self.scale_res_scale))
            quat_res_scale = raw.new_tensor(float(self.quat_res_scale))
        else:
            mean_res_frac = self._output_scale(0, self.mean_res_frac, raw)
            rgb_res_scale = self._output_scale(1, self.rgb_res_scale, raw)
            scale_res_scale = self._output_scale(2, self.scale_res_scale, raw)
            quat_res_scale = self._output_scale(3, self.quat_res_scale, raw)
        valid = base["valid"]
        mask = base["mask"]
        mean_res = torch.tanh(raw[:, 0:3]) * (mean_res_frac * radius_t)
        policy_depth_res = raw.new_zeros((raw.shape[0], 1))
        policy_move_res = raw.new_zeros((raw.shape[0], 3))
        policy_scale_mult = raw.new_ones((raw.shape[0], 3))
        policy_opacity_mult = raw.new_ones((raw.shape[0], 1))
        policy_view_gate = raw.new_ones((raw.shape[0], 1))
        policy_confidence_gate = raw.new_ones((raw.shape[0], 1))
        policy_keep_gate = raw.new_ones((raw.shape[0], 1))
        policy_coverage_mult = raw.new_ones((raw.shape[0], 1))
        policy_birth_gate = raw.new_ones((raw.shape[0], 1))
        if policy_raw is not None:
            policy_depth_scale = self._policy_output_scale(
                0, self.policy_depth_res_frac, raw, max_value=0.25
            )
            policy_move_scale = self._policy_output_scale(
                1, self.policy_move_res_frac, raw, max_value=0.25
            )
            policy_scale_scale = self._policy_output_scale(
                2, self.policy_scale_res_scale, raw, max_value=4.0
            )
            policy_opacity_scale = self._policy_output_scale(
                3, self.policy_opacity_res_scale, raw, max_value=4.0
            )
            policy_view_scale = self._policy_output_scale(
                4, self.policy_view_res_scale, raw, max_value=4.0
            )
            policy_confidence_scale = self._policy_output_scale(
                5, self.policy_confidence_res_scale, raw, max_value=4.0
            )
            policy_keep_scale = self._policy_output_scale(
                6, self.policy_keep_res_scale, raw, max_value=4.0
            )
            policy_coverage_scale = self._policy_output_scale(
                7, self.policy_coverage_scale_res_scale, raw, max_value=4.0
            )
            policy_birth_scale = self._policy_output_scale(
                8, self.policy_birth_res_scale, raw, max_value=4.0
            )
            policy_depth_res = (
                torch.tanh(policy_raw[:, 0:1])
                * (policy_depth_scale * radius_t)
            )
            policy_move_res = (
                torch.tanh(policy_raw[:, 1:4])
                * (policy_move_scale * radius_t)
            )
            policy_scale_mult = torch.exp(
                torch.tanh(policy_raw[:, 4:7]) * policy_scale_scale
            )
            policy_opacity_mult = torch.exp(
                torch.tanh(policy_raw[:, 7:8]) * policy_opacity_scale
            )
            policy_view_gate = self._symmetric_gate(
                policy_raw[:, 8:9], policy_view_scale
            )
            policy_confidence_gate = self._symmetric_gate(
                policy_raw[:, 9:10], policy_confidence_scale
            )
            policy_keep_gate = self._symmetric_gate(
                policy_raw[:, 10:11], policy_keep_scale
            )
            policy_coverage_mult = torch.exp(
                torch.tanh(policy_raw[:, 11:12])
                * policy_coverage_scale
            )
            policy_birth_gate = self._symmetric_gate(
                policy_raw[:, 12:13], policy_birth_scale
            )
        mean = base["mean"] + mean_res + base["direction"] * policy_depth_res + policy_move_res
        base_rgb = base["rgb"]
        if self.color_affine is not None and not disable_new_capacity:
            affine = self.color_affine(slots.mean(dim=1))[0].to(dtype=raw.dtype)
            gain = 1.0 + torch.tanh(affine[:3]) * self.color_affine_scale
            bias = torch.tanh(affine[3:6]) * self.color_affine_scale
            base_rgb = (base_rgb * gain.reshape(1, 3) + bias.reshape(1, 3)).clamp(0.0, 1.0)
        rgb = (base_rgb + torch.tanh(raw[:, 3:6]) * rgb_res_scale).clamp(0.0, 1.0)
        if self.learned_scale_head and not disable_new_capacity:
            lo = math.log(self.learned_scale_min_frac)
            hi = math.log(self.learned_scale_max_frac)
            scale_frac = torch.exp(raw.new_tensor(lo) + torch.sigmoid(raw[:, 6:9]) * (hi - lo))
            scale = (radius_t * scale_frac).clamp_min(self.learned_scale_min_frac * radius_t)
        else:
            scale_mult = torch.exp(torch.tanh(raw[:, 6:9]) * scale_res_scale)
            if self.log_scale_base is not None and not disable_new_capacity:
                base_scale = radius_t * torch.exp(
                    self.log_scale_base.to(device=raw.device, dtype=raw.dtype)
                ).reshape(1, 3)
            else:
                tangent = max(float(self.scale_frac), 1e-6) * radius_t
                normal = max(float(self.normal_scale_frac), 1e-6) * radius_t
                base_scale = torch.stack([tangent, tangent, normal]).reshape(1, 3)
            scale = (base_scale * scale_mult).clamp_min(1e-5 * radius_t)
        coverage_vec = torch.cat([
            policy_coverage_mult,
            policy_coverage_mult,
            torch.ones_like(policy_coverage_mult),
        ], dim=-1)
        scale = (scale * policy_scale_mult * coverage_vec).clamp_min(1e-5 * radius_t)
        quat = F.normalize(base["quat"] + quat_res_scale * torch.tanh(raw[:, 9:13]), dim=-1)
        if ((self.opacity_prior is not None or self.opacity_logit_bias is not None)
                and not disable_new_capacity):
            bias = raw.new_zeros((1, 1))
            if self.opacity_logit_bias is not None:
                bias = bias + self.opacity_logit_bias.to(device=raw.device, dtype=raw.dtype).reshape(1, 1)
            if self.opacity_prior is not None:
                scene_bias = self.opacity_prior(slots.mean(dim=1))[0].to(dtype=raw.dtype)
                bias = bias + scene_bias.reshape(1, 1)
            opacity = torch.sigmoid(raw[:, 13:14] + bias) * mask * valid
        else:
            init = min(max(self.opacity_init, 1e-4), 1.0 - 1e-4)
            prior = math.log(init / (1.0 - init))
            opacity = torch.sigmoid(raw[:, 13:14] + raw.new_tensor(prior)) * mask * valid
        opacity = (
            opacity
            * source_confidence_gate
            * policy_opacity_mult
            * policy_view_gate
            * policy_confidence_gate
            * policy_keep_gate
        ).clamp(0.0, 1.0)
        out = {
            "mean": mean,
            "quat": quat,
            "scale": scale,
            "opacity": opacity,
            "rgb": rgb,
            "depth": base["raydist"],
            "mean_anchor": base["mean"],
            "mean_offset": mean_res,
            "scale_raw": scale,
            "_surface_token_mask": mask,
            "_surface_token_valid": valid,
            "_surface_token_depth_normal_blend": depth_normal_blend_tok,
            "_surface_token_source_depth_res": source_depth_res,
            "_surface_token_source_confidence_gate": source_confidence_gate,
            "_surface_token_source_base_mean": source_base_mean,
            "_surface_token_source_direction": source_direction,
        }
        if policy_raw is not None:
            out.update({
                "_surface_token_policy_depth_res": policy_depth_res,
                "_surface_token_policy_move_res": policy_move_res,
                "_surface_token_policy_scale_mult": policy_scale_mult,
                "_surface_token_policy_opacity_mult": policy_opacity_mult,
                "_surface_token_policy_view_gate": policy_view_gate,
                "_surface_token_policy_confidence_gate": policy_confidence_gate,
                "_surface_token_policy_keep_gate": policy_keep_gate,
                "_surface_token_policy_coverage_mult": policy_coverage_mult,
                "_surface_token_policy_birth_gate": policy_birth_gate,
            })
        if self.detail_head is not None and not disable_new_capacity:
            raw_d = self.detail_head(src[0])
            mean_res_d = torch.tanh(raw_d[:, 0:3]) * (self.detail_mean_res_frac * radius_t)
            mean_d = base["mean"] + mean_res_d
            rgb_d = (
                base["rgb"] + torch.tanh(raw_d[:, 3:6]) * self.detail_rgb_res_scale
            ).clamp(0.0, 1.0)
            scale_mult_d = torch.exp(torch.tanh(raw_d[:, 6:9]) * self.detail_scale_res_scale)
            tangent_d = max(float(self.detail_scale_frac), 1e-6) * radius_t
            normal_d = max(float(self.detail_normal_scale_frac), 1e-6) * radius_t
            base_scale_d = torch.stack([tangent_d, tangent_d, normal_d]).reshape(1, 3)
            scale_d = (base_scale_d * scale_mult_d).clamp_min(1e-5 * radius_t)
            quat_d = F.normalize(
                base["quat"] + self.detail_quat_res_scale * torch.tanh(raw_d[:, 9:13]),
                dim=-1,
            )
            init_d = min(max(self.detail_opacity_init, 1e-4), 1.0 - 1e-4)
            prior_d = math.log(init_d / (1.0 - init_d))
            opacity_d = torch.sigmoid(raw_d[:, 13:14] + raw_d.new_tensor(prior_d)) * mask * valid
            opacity_d = (
                opacity_d
                * source_confidence_gate
                * policy_view_gate
                * policy_confidence_gate
                * policy_birth_gate
            ).clamp(0.0, 1.0)
            out.update({
                "mean": torch.cat([out["mean"], mean_d], dim=0),
                "quat": torch.cat([out["quat"], quat_d], dim=0),
                "scale": torch.cat([out["scale"], scale_d], dim=0),
                "opacity": torch.cat([out["opacity"], opacity_d], dim=0),
                "rgb": torch.cat([out["rgb"], rgb_d], dim=0),
                "depth": torch.cat([out["depth"], base["raydist"]], dim=0),
                "mean_anchor": torch.cat([out["mean_anchor"], base["mean"]], dim=0),
                "mean_offset": torch.cat([out["mean_offset"], mean_res_d], dim=0),
                "scale_raw": torch.cat([out["scale_raw"], scale_d], dim=0),
                "_surface_token_mask": torch.cat([out["_surface_token_mask"], mask], dim=0),
                "_surface_token_valid": torch.cat([out["_surface_token_valid"], valid], dim=0),
                "_surface_token_depth_normal_blend": torch.cat([
                    out["_surface_token_depth_normal_blend"],
                    depth_normal_blend_tok,
                ], dim=0),
                "_surface_token_source_depth_res": torch.cat([
                    out["_surface_token_source_depth_res"],
                    source_depth_res,
                ], dim=0),
                "_surface_token_source_confidence_gate": torch.cat([
                    out["_surface_token_source_confidence_gate"],
                    source_confidence_gate,
                ], dim=0),
                "_surface_token_source_base_mean": torch.cat([
                    out["_surface_token_source_base_mean"],
                    source_base_mean,
                ], dim=0),
                "_surface_token_source_direction": torch.cat([
                    out["_surface_token_source_direction"],
                    source_direction,
                ], dim=0),
                "_surface_token_detail": torch.cat([
                    torch.zeros_like(valid),
                    torch.ones_like(valid),
                ], dim=0),
            })
        proposals = None
        if self.proposal_head is not None and not disable_new_capacity:
            proposals = self._proposal_params(slots, radius_t, surface_tokens=src[0], base=base)
        if proposals is not None:
            base_rows = out["mean"].shape[0]
            out.update({
                "mean": torch.cat([out["mean"], proposals["mean"]], dim=0),
                "quat": torch.cat([out["quat"], proposals["quat"]], dim=0),
                "scale": torch.cat([out["scale"], proposals["scale"]], dim=0),
                "opacity": torch.cat([out["opacity"], proposals["opacity"]], dim=0),
                "rgb": torch.cat([out["rgb"], proposals["rgb"]], dim=0),
                "depth": torch.cat([out["depth"], proposals["depth"]], dim=0),
                "mean_anchor": torch.cat([out["mean_anchor"], proposals["mean_anchor"]], dim=0),
                "mean_offset": torch.cat([out["mean_offset"], proposals["mean_offset"]], dim=0),
                "scale_raw": torch.cat([out["scale_raw"], proposals["scale_raw"]], dim=0),
                "_surface_token_mask": torch.cat([
                    out["_surface_token_mask"], proposals["_surface_token_mask"],
                ], dim=0),
                "_surface_token_valid": torch.cat([
                    out["_surface_token_valid"], proposals["_surface_token_valid"],
                ], dim=0),
                "_surface_token_depth_normal_blend": torch.cat([
                    out["_surface_token_depth_normal_blend"],
                    torch.zeros_like(proposals["_surface_token_valid"]),
                ], dim=0),
                "_surface_token_source_depth_res": torch.cat([
                    out["_surface_token_source_depth_res"],
                    torch.zeros_like(proposals["_surface_token_valid"]),
                ], dim=0),
                "_surface_token_source_confidence_gate": torch.cat([
                    out["_surface_token_source_confidence_gate"],
                    torch.ones_like(proposals["_surface_token_valid"]),
                ], dim=0),
                "_surface_token_source_base_mean": torch.cat([
                    out["_surface_token_source_base_mean"],
                    proposals["mean_anchor"].detach(),
                ], dim=0),
                "_surface_token_source_direction": torch.cat([
                    out["_surface_token_source_direction"],
                    torch.zeros_like(proposals["mean_anchor"]),
                ], dim=0),
                "_surface_token_proposal": torch.cat([
                    torch.zeros((base_rows, 1), device=out["mean"].device, dtype=out["mean"].dtype),
                    proposals["_surface_token_proposal"],
                ], dim=0),
                "_surface_token_proposal_coverage_mult": torch.cat([
                    torch.zeros((base_rows, 1), device=out["mean"].device, dtype=out["mean"].dtype),
                    proposals["_surface_token_proposal_coverage_mult"],
                ], dim=0),
                "_surface_token_proposal_policy_keep_gate": torch.cat([
                    torch.zeros((base_rows, 1), device=out["mean"].device, dtype=out["mean"].dtype),
                    proposals["_surface_token_proposal_policy_keep_gate"],
                ], dim=0),
                "_surface_token_proposal_policy_confidence_gate": torch.cat([
                    torch.zeros((base_rows, 1), device=out["mean"].device, dtype=out["mean"].dtype),
                    proposals["_surface_token_proposal_policy_confidence_gate"],
                ], dim=0),
                "_surface_token_proposal_policy_coverage_mult": torch.cat([
                    torch.zeros((base_rows, 1), device=out["mean"].device, dtype=out["mean"].dtype),
                    proposals["_surface_token_proposal_policy_coverage_mult"],
                ], dim=0),
                "_surface_token_proposal_surface_seed": torch.cat([
                    torch.zeros((base_rows, 1), device=out["mean"].device, dtype=out["mean"].dtype),
                    proposals["_surface_token_proposal_surface_seed"],
                ], dim=0),
                "_surface_token_proposal_anchor_mix": torch.cat([
                    torch.zeros((base_rows, 1), device=out["mean"].device, dtype=out["mean"].dtype),
                    proposals["_surface_token_proposal_anchor_mix"],
                ], dim=0),
                "_surface_token_proposal_anchor_entropy": torch.cat([
                    torch.zeros((base_rows, 1), device=out["mean"].device, dtype=out["mean"].dtype),
                    proposals["_surface_token_proposal_anchor_entropy"],
                ], dim=0),
                "_surface_token_proposal_anchor_entropy_loss": proposals[
                    "_surface_token_proposal_anchor_entropy_loss"
                ],
                "_surface_token_proposal_anchor_usage_loss": proposals[
                    "_surface_token_proposal_anchor_usage_loss"
                ],
                "_surface_token_proposal_anchor_usage_perplexity": proposals[
                    "_surface_token_proposal_anchor_usage_perplexity"
                ],
                "_surface_token_proposal_anchor_unique_frac": proposals[
                    "_surface_token_proposal_anchor_unique_frac"
                ],
                "_surface_token_proposal_anchor_collision_loss": proposals[
                    "_surface_token_proposal_anchor_collision_loss"
                ],
                "_surface_token_proposal_anchor_collision_frac": proposals[
                    "_surface_token_proposal_anchor_collision_frac"
                ],
            })
            if "_surface_token_detail" in out:
                out["_surface_token_detail"] = torch.cat([
                    out["_surface_token_detail"],
                    torch.zeros_like(proposals["_surface_token_valid"]),
                ], dim=0)
        return out
