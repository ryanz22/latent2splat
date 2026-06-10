"""Phase-2: train CleanGSDecoder across many objects (batch=1, one object/step) +
periodic train/held-out eval + W&B logging. Tests whether one shared-weight decoder
can fit many objects (primary) and generalize to held-out ones (secondary).

  wandb login                       # one-time for online; else --wandb_mode offline
  python -m decoder.clean.train_phase2 --steps 40000 --eval_every 1000
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

V4 = os.environ.get("PHASE2_DATA_ROOT", "/home/rrzhang/projects/data/Animals v4 Final")
MANIFEST = os.environ.get("PHASE2_MANIFEST", V4 + "/animals_v4_approved_encoded.json")
LEARNED_FILL_FEATURE_CHANNELS = 19
FUSION_CANDIDATE_FEATURE_CHANNELS = 12
FUSION_CANDIDATE_COORD_FEATURE_CHANNELS = 6
FUSION_CANDIDATE_RICH_FEATURE_CHANNELS = 9
FUSION_CANDIDATE_VOXEL_FEATURE_CHANNELS = 9
FUSION_CANDIDATE_NEIGHBOR_SIGNAL_CHANNELS = 10
FUSION_CANDIDATE_NEIGHBOR_FEATURE_CHANNELS = FUSION_CANDIDATE_NEIGHBOR_SIGNAL_CHANNELS * 2 + 1


class AdaptiveLossBalancer(torch.nn.Module):
    """Learn uncertainty-style weights for selected scalar losses."""

    def __init__(self, names: list[str], logvar_min: float = -4.0, logvar_max: float = 4.0):
        super().__init__()
        self.log_vars = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.zeros(())) for name in names
        })
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def has(self, name: str) -> bool:
        return name in self.log_vars

    def forward(
        self,
        name: str,
        raw: torch.Tensor,
        base_weight: float = 1.0,
    ) -> torch.Tensor:
        log_var = self.log_vars[name].clamp(self.logvar_min, self.logvar_max)
        base = float(base_weight)
        if base <= 0.0:
            base = 1.0
        return base * torch.exp(-log_var) * raw + log_var

    def values(self) -> dict[str, float]:
        return {
            f"lossw_{name}": float(torch.exp(-p.detach().clamp(self.logvar_min, self.logvar_max)))
            for name, p in self.log_vars.items()
        }


def _alpha_anti_lattice_loss(
    alpha: torch.Tensor,
    target_rgb: torch.Tensor,
    target_fg: torch.Tensor,
    *,
    blur_px: int = 2,
    edge_band_px: int = 4,
    detail_edge_thresh: float = 0.025,
) -> torch.Tensor:
    """Suppress dotted proposal coverage inside smooth foreground regions.

    This is deliberately a render-space loss: it does not prescribe which
    proposal should move or widen. It only penalizes high-frequency alpha
    texture where the target foreground is smooth and away from silhouette
    edges, so gradients can train the learned coverage/scale/opacity policy.
    """
    if alpha.ndim != 4 or alpha.shape[-1] != 1:
        raise ValueError("alpha must have shape (N,H,W,1)")
    if target_rgb.ndim != 4 or target_rgb.shape[-1] != 3:
        raise ValueError("target_rgb must have shape (N,H,W,3)")
    if target_fg.ndim != 4 or target_fg.shape[-1] != 1:
        raise ValueError("target_fg must have shape (N,H,W,1)")
    if alpha.shape[:3] != target_rgb.shape[:3] or alpha.shape[:3] != target_fg.shape[:3]:
        raise ValueError("alpha, target_rgb, and target_fg must agree on N,H,W")

    fg = (target_fg.detach() > 0.5).to(dtype=alpha.dtype, device=alpha.device)
    rgb = target_rgb.detach().to(device=alpha.device, dtype=alpha.dtype)
    detail_edge_thresh = max(float(detail_edge_thresh), 1e-6)

    edge = alpha.new_zeros(alpha.shape)
    gx_fg = target_fg[:, :, 1:, :] - target_fg[:, :, :-1, :]
    gy_fg = target_fg[:, 1:, :, :] - target_fg[:, :-1, :, :]
    ex = gx_fg.detach().abs().to(device=alpha.device, dtype=alpha.dtype)
    ey = gy_fg.detach().abs().to(device=alpha.device, dtype=alpha.dtype)
    edge[:, :, 1:, :] = torch.maximum(edge[:, :, 1:, :], ex)
    edge[:, :, :-1, :] = torch.maximum(edge[:, :, :-1, :], ex)
    edge[:, 1:, :, :] = torch.maximum(edge[:, 1:, :, :], ey)
    edge[:, :-1, :, :] = torch.maximum(edge[:, :-1, :, :], ey)

    detail = alpha.new_zeros(alpha.shape)
    gx_rgb = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean(dim=-1, keepdim=True)
    gy_rgb = (rgb[:, 1:, :, :] - rgb[:, :-1, :, :]).abs().mean(dim=-1, keepdim=True)
    detail[:, :, 1:, :] = torch.maximum(detail[:, :, 1:, :], gx_rgb)
    detail[:, :, :-1, :] = torch.maximum(detail[:, :, :-1, :], gx_rgb)
    detail[:, 1:, :, :] = torch.maximum(detail[:, 1:, :, :], gy_rgb)
    detail[:, :-1, :, :] = torch.maximum(detail[:, :-1, :, :], gy_rgb)

    band = max(int(edge_band_px), 0)
    if band > 0:
        edge_nchw = edge.permute(0, 3, 1, 2)
        edge_nchw = F.max_pool2d(
            edge_nchw,
            kernel_size=2 * band + 1,
            stride=1,
            padding=band,
        )
        edge = edge_nchw.permute(0, 2, 3, 1)
        detail_nchw = detail.permute(0, 3, 1, 2)
        detail_nchw = F.max_pool2d(
            detail_nchw,
            kernel_size=2 * band + 1,
            stride=1,
            padding=band,
        )
        detail = detail_nchw.permute(0, 2, 3, 1)

    smooth_w = (1.0 - (detail / detail_edge_thresh).clamp(0.0, 1.0))
    smooth_w = smooth_w * fg * (edge <= 0).to(dtype=alpha.dtype)
    if not bool((smooth_w > 0).any()):
        return alpha.new_zeros(())

    blur_px = max(int(blur_px), 1)
    kernel = 2 * blur_px + 1
    alpha_nchw = alpha.permute(0, 3, 1, 2)
    alpha_pad = F.pad(
        alpha_nchw,
        (blur_px, blur_px, blur_px, blur_px),
        mode="replicate",
    )
    blur = F.avg_pool2d(alpha_pad, kernel_size=kernel, stride=1, padding=0)
    highpass = (alpha_nchw - blur).permute(0, 2, 3, 1)
    return (highpass.abs() * smooth_w).sum() / smooth_w.sum().clamp_min(1.0)


def _normalize_confidence_map(confidence: torch.Tensor,
                              valid: torch.Tensor | None = None,
                              q_low: float = 0.05,
                              q_high: float = 0.95,
                              eps: float = 1e-6) -> torch.Tensor:
    """Robustly normalize a confidence map without breaking constant priors.

    Learned confidence heads are zero-initialized, so their first prediction is
    intentionally near-constant.  Quantile normalization should preserve that
    identity prior; if the quantile span degenerates, fall back to the raw
    clamped confidence instead of mapping the whole image to zero.
    """
    c = torch.where(torch.isfinite(confidence), confidence, confidence.new_zeros(()))
    c = c.clamp(0.0, 1.0)
    if valid is None:
        vals = c.reshape(-1)
    else:
        vals = c[valid]
    if vals.numel() == 0:
        return c
    lo = torch.quantile(vals, q_low)
    hi = torch.quantile(vals, q_high)
    span = hi - lo
    if bool(span <= max(float(eps), 0.0)):
        return c
    return ((c - lo) / span.clamp_min(float(eps))).clamp(0.0, 1.0)


def _depth_confidence_targets(
    prior_frac: torch.Tensor,
    target_frac: torch.Tensor,
    valid: torch.Tensor,
    positive_tol_frac: float,
    negative_tol_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build conservative labels for depth-confidence supervision.

    Small DA3-vs-GT depth differences should preserve surface confidence.
    Very large differences should lower confidence.  The band between those
    thresholds is intentionally ignored so the head does not learn to delete
    plausible but slightly misregistered foreground surface.
    """
    pos_tol = max(float(positive_tol_frac), 1e-6)
    neg_tol = float(negative_tol_frac)
    if neg_tol <= 0:
        neg_tol = pos_tol
    neg_tol = max(neg_tol, pos_tol)
    err = (prior_frac - target_frac).abs()
    valid_bool = valid.to(dtype=torch.bool)
    positive = valid_bool & (err <= pos_tol)
    negative = valid_bool & (err >= neg_tol)
    target = positive.to(dtype=prior_frac.dtype)
    target_valid = positive | negative
    return target, target_valid


def _sample_rgbd_surface_points(
    frames: torch.Tensor,
    fg: torch.Tensor,
    depths: torch.Tensor,
    K_all: torch.Tensor,
    c2w_all: torch.Tensor,
    max_points: int,
    fg_threshold: float = 0.5,
    return_detail: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Randomly sample observed foreground RGBD surface points in world space."""
    max_points = int(max_points)
    if max_points <= 0 or depths is None:
        return None
    if depths.ndim != 3:
        raise ValueError(f"expected depths shaped (V,H,W), got {tuple(depths.shape)}")
    v_count, h, w = depths.shape
    if v_count <= 0:
        return None
    device = depths.device
    dtype = depths.dtype if depths.dtype.is_floating_point else torch.float32
    fg_thr = min(max(float(fg_threshold), 0.0), 1.0)
    per_view = max(1, math.ceil(max_points / max(v_count, 1)))
    attempts = min(int(h * w), max(per_view * 16, 512))
    pts_world: list[torch.Tensor] = []
    pts_rgb: list[torch.Tensor] = []
    pts_detail: list[torch.Tensor] = []
    fg_map = fg[..., 0] if fg.ndim == 4 else fg
    detail_map = None
    if return_detail:
        frames_f = frames.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        detail = torch.zeros((v_count, h, w), device=device, dtype=dtype)
        if w > 1:
            dx = (frames_f[:, :, 1:, :] - frames_f[:, :, :-1, :]).abs().mean(dim=-1)
            detail[:, :, 1:] = torch.maximum(detail[:, :, 1:], dx)
            detail[:, :, :-1] = torch.maximum(detail[:, :, :-1], dx)
        if h > 1:
            dy = (frames_f[:, 1:, :, :] - frames_f[:, :-1, :, :]).abs().mean(dim=-1)
            detail[:, 1:, :] = torch.maximum(detail[:, 1:, :], dy)
            detail[:, :-1, :] = torch.maximum(detail[:, :-1, :], dy)
        detail_map = detail

    for view_i in range(v_count):
        yy = torch.randint(0, h, (attempts,), device=device)
        xx = torch.randint(0, w, (attempts,), device=device)
        z = depths[view_i, yy, xx].to(dtype=dtype)
        m = fg_map[view_i, yy, xx].to(dtype=dtype)
        valid = torch.isfinite(z) & (z > 1e-6) & (z < 1e5) & (m > fg_thr)
        if not bool(valid.any()):
            # Fallback for very thin objects where random sampling misses the mask.
            full_valid = (
                torch.isfinite(depths[view_i])
                & (depths[view_i] > 1e-6)
                & (depths[view_i] < 1e5)
                & (fg_map[view_i] > fg_thr)
            )
            idx = full_valid.reshape(-1).nonzero(as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                continue
            if idx.numel() > per_view:
                idx = idx.index_select(
                    0, torch.randperm(idx.numel(), device=device)[:per_view]
                )
            yy_sel = torch.div(idx, w, rounding_mode="floor")
            xx_sel = idx.remainder(w)
            z_sel = depths[view_i, yy_sel, xx_sel].to(dtype=dtype)
        else:
            idx = valid.nonzero(as_tuple=False).reshape(-1)
            if idx.numel() > per_view:
                idx = idx.index_select(
                    0, torch.randperm(idx.numel(), device=device)[:per_view]
                )
            yy_sel = yy.index_select(0, idx)
            xx_sel = xx.index_select(0, idx)
            z_sel = z.index_select(0, idx)

        k = K_all[view_i].to(device=device, dtype=dtype)
        x = (xx_sel.to(dtype=dtype) - k[0, 2]) / k[0, 0].clamp_min(1e-6)
        y = (yy_sel.to(dtype=dtype) - k[1, 2]) / k[1, 1].clamp_min(1e-6)
        d_cam = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)
        factor = torch.sqrt(x.square() + y.square() + 1.0)
        ray_t = z_sel * factor
        c2w = c2w_all[view_i].to(device=device, dtype=dtype)
        d_world = d_cam @ c2w[:3, :3].T
        d_world = d_world / d_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pts_world.append(c2w[:3, 3].reshape(1, 3) + ray_t.reshape(-1, 1) * d_world)
        pts_rgb.append(frames[view_i, yy_sel, xx_sel].to(device=device, dtype=dtype))
        if detail_map is not None:
            pts_detail.append(detail_map[view_i, yy_sel, xx_sel].reshape(-1, 1))

    if not pts_world:
        return None
    world = torch.cat(pts_world, dim=0)
    rgb = torch.cat(pts_rgb, dim=0)
    detail = torch.cat(pts_detail, dim=0) if pts_detail else None
    if world.shape[0] > max_points:
        idx = torch.randperm(world.shape[0], device=device)[:max_points]
        world = world.index_select(0, idx)
        rgb = rgb.index_select(0, idx)
        if detail is not None:
            detail = detail.index_select(0, idx)
    if return_detail:
        if detail is None:
            detail = world.new_zeros((world.shape[0], 1))
        return world, rgb, detail
    return world, rgb


def _chunked_nearest(
    query: torch.Tensor,
    support: torch.Tensor,
    chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbor squared distance and index without materializing huge matrices."""
    if query.numel() == 0 or support.numel() == 0:
        raise ValueError("_chunked_nearest requires non-empty query and support tensors")
    mins: list[torch.Tensor] = []
    argmins: list[torch.Tensor] = []
    chunk = max(int(chunk), 1)
    for start in range(0, query.shape[0], chunk):
        q = query[start:start + chunk]
        d2 = torch.cdist(q, support).square()
        val, idx = d2.min(dim=1)
        mins.append(val)
        argmins.append(idx)
    return torch.cat(mins, dim=0), torch.cat(argmins, dim=0)


def _surface_token_proposal_losses(
    params: dict[str, torch.Tensor],
    frames: torch.Tensor,
    fg: torch.Tensor,
    depths: torch.Tensor | None,
    K_all: torch.Tensor,
    c2w_all: torch.Tensor,
    radius: float,
    cover_points: int,
    depth_tol_frac: float,
    fg_threshold: float,
    opacity_positive_weight: float = 2.0,
    opacity_negative_weight: float = 0.25,
    policy_target_mode: str = "support",
    detail_edge_thresh: float = 0.025,
) -> dict[str, torch.Tensor]:
    """Allocation losses for free proposal splats against observed RGBD surface."""
    mean = params["mean"]
    zero = mean.new_zeros(())
    out = {
        "cover": zero,
        "surface": zero,
        "opacity": zero,
        "rgb": zero,
        "detail_cover": zero,
        "detail_mean": zero,
        "policy_keep": zero,
        "policy_confidence": zero,
        "policy_coverage": zero,
        "support_mean": zero,
        "opacity_mean": zero,
        "policy_keep_target_mean": zero,
        "policy_confidence_target_mean": zero,
        "policy_coverage_target_mean": zero,
    }
    proposal_mask = params.get("_surface_token_proposal")
    if proposal_mask is None or depths is None:
        return out
    prop_idx = (proposal_mask.reshape(-1) > 0.5).nonzero(as_tuple=False).reshape(-1)
    if prop_idx.numel() == 0:
        return out
    sampled = _sample_rgbd_surface_points(
        frames,
        fg,
        depths,
        K_all,
        c2w_all,
        max_points=cover_points,
        fg_threshold=fg_threshold,
        return_detail=True,
    )
    if sampled is None:
        return out
    target_xyz, target_rgb, target_detail = sampled
    if target_xyz.numel() == 0:
        return out
    target_xyz = target_xyz.to(device=mean.device, dtype=mean.dtype)
    target_rgb = target_rgb.to(device=mean.device, dtype=params["rgb"].dtype)
    target_detail = target_detail.to(device=mean.device, dtype=mean.dtype).reshape(-1)
    prop_mean = mean.index_select(0, prop_idx)
    prop_opacity = params["opacity"].index_select(0, prop_idx).reshape(-1).clamp(1e-5, 1.0 - 1e-5)
    prop_rgb = params["rgb"].index_select(0, prop_idx)
    radius_t = mean.new_tensor(max(float(radius), 1e-6))
    tol = radius_t * max(float(depth_tol_frac), 1e-6)

    target_to_prop_d2, _ = _chunked_nearest(target_xyz, prop_mean)
    prop_to_target_d2, prop_nn = _chunked_nearest(prop_mean, target_xyz)
    target_to_prop_dist = torch.sqrt(target_to_prop_d2 + 1e-12)
    prop_to_target_dist = torch.sqrt(prop_to_target_d2 + 1e-12)
    cover = (target_to_prop_dist / radius_t).clamp(max=2.0).mean()
    surface = (prop_to_target_dist / radius_t).clamp(max=2.0).mean()
    detail_edge_thresh = max(float(detail_edge_thresh), 1e-6)
    detail_w = (target_detail / detail_edge_thresh).clamp(0.0, 1.0).detach()
    detail_cover = (
        ((target_to_prop_dist / radius_t).clamp(max=2.0) * detail_w).sum()
        / detail_w.sum().clamp_min(1e-6)
    )
    support = torch.exp(-0.5 * (prop_to_target_dist / tol.clamp_min(1e-6)).square()).detach()
    bce = F.binary_cross_entropy(prop_opacity, support.clamp(1e-5, 1.0 - 1e-5), reduction="none")
    op_w = (
        max(float(opacity_positive_weight), 0.0) * support
        + max(float(opacity_negative_weight), 0.0) * (1.0 - support)
    )
    opacity = (bce * op_w).sum() / op_w.sum().clamp_min(1e-6)
    rgb_w = support.clamp_min(0.05).unsqueeze(-1)
    rgb_target = target_rgb.index_select(0, prop_nn)
    rgb = ((prop_rgb - rgb_target).abs() * rgb_w).sum() / (rgb_w.sum() * prop_rgb.shape[-1]).clamp_min(1e-6)
    support_target = support.detach().clamp(0.0, 1.0)
    policy_target_mode = str(policy_target_mode).strip().lower()
    if policy_target_mode not in {"support", "identity", "none"}:
        raise ValueError("policy_target_mode must be 'support', 'identity', or 'none'")
    if policy_target_mode == "support":
        keep_target = 0.25 + 1.50 * support_target
        confidence_target = 0.25 + 1.50 * support_target
        coverage_target = torch.exp((1.0 - 2.0 * support_target) * 0.75)
    else:
        keep_target = torch.ones_like(support_target)
        confidence_target = torch.ones_like(support_target)
        coverage_target = torch.ones_like(support_target)
    policy_keep = zero
    policy_confidence = zero
    policy_coverage = zero
    keep_gate = params.get("_surface_token_proposal_policy_keep_gate")
    confidence_gate = params.get("_surface_token_proposal_policy_confidence_gate")
    coverage_gate = params.get("_surface_token_proposal_policy_coverage_mult")
    if keep_gate is not None and policy_target_mode != "none":
        keep_q = keep_gate.index_select(0, prop_idx).reshape(-1)
        policy_keep = (keep_q - keep_target.to(dtype=keep_q.dtype)).square().mean()
    if confidence_gate is not None and policy_target_mode != "none":
        conf_q = confidence_gate.index_select(0, prop_idx).reshape(-1)
        policy_confidence = (
            conf_q - confidence_target.to(dtype=conf_q.dtype)
        ).square().mean()
    if coverage_gate is not None and policy_target_mode != "none":
        cov_q = coverage_gate.index_select(0, prop_idx).reshape(-1).clamp_min(1e-6)
        target = coverage_target.to(dtype=cov_q.dtype).clamp_min(1e-6)
        policy_coverage = (cov_q.log() - target.log()).square().mean()
    out.update({
        "cover": cover,
        "surface": surface,
        "opacity": opacity,
        "rgb": rgb,
        "detail_cover": detail_cover,
        "detail_mean": target_detail.detach().mean(),
        "policy_keep": policy_keep,
        "policy_confidence": policy_confidence,
        "policy_coverage": policy_coverage,
        "support_mean": support.mean(),
        "opacity_mean": prop_opacity.detach().mean(),
        "policy_keep_target_mean": keep_target.mean(),
        "policy_confidence_target_mean": confidence_target.mean(),
        "policy_coverage_target_mean": coverage_target.mean(),
    })
    return out


def _surface_token_source_policy_losses(
    params: dict[str, torch.Tensor],
    frames: torch.Tensor,
    fg: torch.Tensor,
    depths: torch.Tensor | None,
    K_all: torch.Tensor,
    c2w_all: torch.Tensor,
    radius: float,
    source_points: int,
    target_points: int,
    depth_tol_frac: float,
    fg_threshold: float,
    confidence_positive_weight: float = 2.0,
    confidence_negative_weight: float = 0.5,
    confidence_target_scale: float = 0.35,
    support_mode: str = "nearest",
    target_mode: str = "support",
    include_detail: bool = False,
) -> dict[str, torch.Tensor]:
    """Direct support targets for learned source depth/confidence policy.

    Proposal losses train the free splats. This trains the dense RGBD-derived
    source tokens, so the source confidence/depth head has a non-render signal
    for suppressing unsupported tokens and nudging supported ones along their
    own input rays.
    """
    mean = params["mean"]
    zero = mean.new_zeros(())
    out = {
        "confidence": zero,
        "depth": zero,
        "support_mean": zero,
        "confidence_mean": zero,
        "confidence_target_mean": zero,
        "depth_target_abs_frac": zero,
    }
    gate = params.get("_surface_token_source_confidence_gate")
    depth_res = params.get("_surface_token_source_depth_res")
    base_mean = params.get("_surface_token_source_base_mean")
    direction = params.get("_surface_token_source_direction")
    valid = params.get("_surface_token_valid")
    mask = params.get("_surface_token_mask")
    if gate is None or depth_res is None or base_mean is None or direction is None:
        return out
    if depths is None:
        return out
    source_mask = torch.ones(mean.shape[0], dtype=torch.bool, device=mean.device)
    if valid is not None:
        source_mask &= valid.reshape(-1).to(device=mean.device) > 0.5
    if mask is not None:
        source_mask &= mask.reshape(-1).to(device=mean.device) > 0.5
    proposal_mask = params.get("_surface_token_proposal")
    if proposal_mask is not None:
        source_mask &= ~(proposal_mask.reshape(-1).to(device=mean.device) > 0.5)
    detail_mask = params.get("_surface_token_detail")
    if detail_mask is not None and not include_detail:
        source_mask &= ~(detail_mask.reshape(-1).to(device=mean.device) > 0.5)
    src_idx = source_mask.nonzero(as_tuple=False).reshape(-1)
    if src_idx.numel() == 0:
        return out
    max_source = max(int(source_points), 0)
    if max_source > 0 and src_idx.numel() > max_source:
        pick = torch.randperm(src_idx.numel(), device=src_idx.device)[:max_source]
        src_idx = src_idx.index_select(0, pick)
    sampled = _sample_rgbd_surface_points(
        frames,
        fg,
        depths,
        K_all,
        c2w_all,
        max_points=max(int(target_points), 1),
        fg_threshold=fg_threshold,
        return_detail=False,
    )
    if sampled is None:
        return out
    target_xyz, _ = sampled
    if target_xyz.numel() == 0:
        return out
    target_xyz = target_xyz.to(device=mean.device, dtype=mean.dtype)
    radius_t = mean.new_tensor(max(float(radius), 1e-6))
    tol = radius_t * max(float(depth_tol_frac), 1e-6)

    src_mean = mean.index_select(0, src_idx).detach()
    target_d2, nn = _chunked_nearest(src_mean, target_xyz)
    target_dist = torch.sqrt(target_d2 + 1e-12)
    nearest_support = torch.exp(
        -0.5 * (target_dist / tol.clamp_min(1e-6)).square()
    ).detach().clamp(0.0, 1.0)
    support_mode = str(support_mode).strip().lower()
    if support_mode not in {"nearest", "projective"}:
        raise ValueError("support_mode must be 'nearest' or 'projective'")
    support = nearest_support
    support_valid = torch.ones_like(support, dtype=torch.bool)
    confidence_driver = 2.0 * support - 1.0
    if support_mode == "projective":
        supports = torch.zeros_like(support)
        conflicts = torch.zeros_like(support)
        coverage = torch.zeros_like(support)
        h, w = depths.shape[-2:]
        gl_to_cv = torch.diag(
            mean.new_tensor([1.0, -1.0, -1.0, 1.0])
        ).reshape(1, 4, 4)
        c2w_cv = c2w_all.to(device=mean.device, dtype=mean.dtype) @ gl_to_cv
        w2c_all = torch.linalg.inv(c2w_cv)
        fg_map = fg[..., 0] if fg.ndim == 4 else fg
        margin = 0
        for view_i in range(depths.shape[0]):
            cam = (
                src_mean @ w2c_all[view_i, :3, :3].T
                + w2c_all[view_i, :3, 3]
            )
            z = cam[:, 2]
            K_i = K_all[view_i].to(device=mean.device, dtype=mean.dtype)
            u = K_i[0, 0] * (cam[:, 0] / z.clamp_min(1e-6)) + K_i[0, 2]
            v = K_i[1, 1] * (cam[:, 1] / z.clamp_min(1e-6)) + K_i[1, 2]
            inb = (
                (z > 1e-6)
                & (u >= 0)
                & (u <= w - 1)
                & (v >= 0)
                & (v <= h - 1)
            )
            if not bool(inb.any()):
                continue
            idx = torch.nonzero(inb, as_tuple=False).reshape(-1)
            coverage[idx] += 1.0
            fg_view = fg_map[view_i].to(device=mean.device) > fg_threshold
            if margin > 0:
                fg_view = F.max_pool2d(
                    fg_view.float()[None, None],
                    kernel_size=2 * margin + 1,
                    stride=1,
                    padding=margin,
                )[0, 0] > 0.5
            z_view = depths[view_i].to(device=mean.device, dtype=mean.dtype)
            sampled_fg, depth_match, front_conflict, _ = _sample_depth_support_window(
                fg_view,
                z_view,
                u[idx],
                v[idx],
                z[idx],
                float(tol.detach()),
                radius_px=0,
            )
            if depth_match.any():
                supports[idx[depth_match]] += 1.0
            conflict = (~sampled_fg) | front_conflict
            if conflict.any():
                conflicts[idx[conflict]] += 1.0
        support_valid = coverage > 0
        support = torch.where(
            support_valid,
            (supports / coverage.clamp_min(1.0)).clamp(0.0, 1.0),
            nearest_support,
        ).detach()
        conflict_frac = torch.where(
            support_valid,
            (conflicts / coverage.clamp_min(1.0)).clamp(0.0, 1.0),
            torch.zeros_like(conflicts),
        )
        confidence_driver = torch.where(
            support_valid,
            (support - conflict_frac).clamp(-1.0, 1.0),
            2.0 * nearest_support - 1.0,
        ).detach()

    target_mode = str(target_mode).strip().lower()
    if target_mode not in {"support", "identity", "none"}:
        raise ValueError("target_mode must be 'support', 'identity', or 'none'")
    if target_mode == "support":
        target_scale = max(float(confidence_target_scale), 0.0)
        conf_target = 1.0 + target_scale * confidence_driver
        conf_target = conf_target.clamp_min(1e-4)
    else:
        conf_target = torch.ones_like(support)

    confidence = zero
    gate_q = gate.index_select(0, src_idx).reshape(-1)
    if target_mode != "none":
        valid_w = support_valid.to(dtype=gate_q.dtype)
        weights = (
            max(float(confidence_positive_weight), 0.0) * support
            + max(float(confidence_negative_weight), 0.0) * (1.0 - support)
        ).to(dtype=gate_q.dtype) * valid_w
        confidence = (
            (gate_q - conf_target.to(dtype=gate_q.dtype)).square() * weights
        ).sum() / weights.sum().clamp_min(1e-6)

    depth = zero
    depth_target_abs_frac = zero
    if target_mode != "none":
        base_q = base_mean.index_select(0, src_idx).detach().to(dtype=mean.dtype)
        dir_q = F.normalize(
            direction.index_select(0, src_idx).detach().to(dtype=mean.dtype),
            dim=-1,
        )
        target_nn = target_xyz.index_select(0, nn).detach()
        target_res = ((target_nn - base_q) * dir_q).sum(dim=-1, keepdim=True)
        max_res = 0.25 * radius_t
        target_res = target_res.clamp(-max_res, max_res)
        pred_res = depth_res.index_select(0, src_idx)
        # Depth residuals are useful only where there is nearby observed
        # surface support. Unsupported tokens should be handled by confidence.
        d_w = (support * support_valid.to(dtype=support.dtype)).reshape(-1, 1).to(
            dtype=pred_res.dtype
        )
        depth_raw = F.smooth_l1_loss(
            pred_res / radius_t,
            target_res.to(dtype=pred_res.dtype) / radius_t,
            reduction="none",
        )
        depth = (depth_raw * d_w).sum() / d_w.sum().clamp_min(1e-6)
        depth_target_abs_frac = (target_res.abs() / radius_t).mean()

    out.update({
        "confidence": confidence,
        "depth": depth,
        "support_mean": support.mean(),
        "confidence_mean": gate_q.detach().mean(),
        "confidence_target_mean": (
            (conf_target * support_valid.to(dtype=conf_target.dtype)).sum()
            / support_valid.to(dtype=conf_target.dtype).sum().clamp_min(1.0)
        ),
        "depth_target_abs_frac": depth_target_abs_frac,
    })
    return out


def _erode_mask_2d(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Binary erosion for masks shaped (N,H,W), returned as a float mask."""
    radius_px = max(int(radius_px), 0)
    if radius_px <= 0:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"expected (N,H,W) mask, got {tuple(mask.shape)}")
    x = mask[:, None].to(dtype=torch.float32)
    eroded = 1.0 - F.max_pool2d(
        1.0 - x,
        kernel_size=2 * radius_px + 1,
        stride=1,
        padding=radius_px,
    )
    return eroded[:, 0].to(dtype=mask.dtype)


def _select_iblend_object_color(obj_stack: torch.Tensor,
                                weight_stack: torch.Tensor,
                                mode: str) -> torch.Tensor:
    """Choose object RGB for an iblend candidate stack.

    ``average`` is the original behavior. ``nearest`` keeps color from the
    nearest selected anchor while still allowing blended alpha/coverage.
    ``maxweight`` picks the per-pixel candidate with the highest deterministic
    blend weight, which is sharper but can be more fragmented. ``maxweight_st``
    has the same forward value but straight-through gradients through the
    weighted average, so a learned blend head can train candidate weights from
    photometric losses without blurring the visible render.
    """
    if mode == "average":
        denom = weight_stack.sum(dim=0).clamp_min(1e-6)
        return (obj_stack * weight_stack).sum(dim=0) / denom
    if mode == "nearest":
        return obj_stack[0]
    if mode == "maxweight":
        pick = weight_stack[..., 0].argmax(dim=0)
        gather = pick[None, ..., None].expand(1, *pick.shape, obj_stack.shape[-1])
        return obj_stack.gather(0, gather)[0]
    if mode == "maxweight_st":
        pick = weight_stack[..., 0].argmax(dim=0)
        gather = pick[None, ..., None].expand(1, *pick.shape, obj_stack.shape[-1])
        hard = obj_stack.gather(0, gather)[0]
        soft = _select_iblend_object_color(obj_stack, weight_stack, "average")
        return hard.detach() + soft - soft.detach()
    raise ValueError(f"unsupported iblend color mode: {mode}")


def _select_iblend_alpha(alpha_stack: torch.Tensor,
                         weight_stack: torch.Tensor,
                         mode: str) -> torch.Tensor:
    """Choose/blend alpha for an iblend candidate stack."""
    denom = weight_stack.sum(dim=0).clamp_min(1e-6)
    soft = (alpha_stack * weight_stack).sum(dim=0) / denom
    if mode == "average":
        return soft
    if mode == "nearest":
        return alpha_stack[0]
    if mode == "nearest_average":
        return 0.5 * (alpha_stack[0] + soft)
    if mode in {"maxweight", "maxweight_st"}:
        pick = weight_stack[..., 0].argmax(dim=0)
        gather = pick[None, ..., None].expand(1, *pick.shape, alpha_stack.shape[-1])
        hard = alpha_stack.gather(0, gather)[0]
    elif mode == "maxalpha":
        pick = alpha_stack[..., 0].argmax(dim=0)
        gather = pick[None, ..., None].expand(1, *pick.shape, alpha_stack.shape[-1])
        hard = alpha_stack.gather(0, gather)[0]
    else:
        raise ValueError(f"unsupported iblend alpha mode: {mode}")
    if mode.endswith("_st"):
        return hard.detach() + soft - soft.detach()
    return hard


def _compose_iblend_anchor(obj_stack: torch.Tensor,
                           alpha_stack: torch.Tensor,
                           weight_stack: torch.Tensor,
                           color_mode: str,
                           alpha_mode: str,
                           bg: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose an object-color/alpha candidate stack into one anchor render."""
    obj = _select_iblend_object_color(obj_stack, weight_stack, color_mode)
    alpha = _select_iblend_alpha(alpha_stack, weight_stack, alpha_mode)
    rgb = obj.clamp(0.0, 1.0) * alpha + (1.0 - alpha) * bg
    return obj, alpha, rgb


def _static_fill_confidence(static_alpha: torch.Tensor,
                            min_alpha: float,
                            softness: float) -> torch.Tensor:
    """Continuous confidence for using the static fill render."""
    if min_alpha <= 0:
        return torch.ones_like(static_alpha)
    if softness <= 0:
        return (static_alpha >= min_alpha).to(dtype=static_alpha.dtype)
    t = ((static_alpha - min_alpha) / max(float(softness), 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blend_static_fill(primary_rgb: torch.Tensor,
                       primary_alpha: torch.Tensor,
                       static_rgb: torch.Tensor,
                       static_alpha: torch.Tensor,
                       fill_alpha_power: float,
                       static_alpha_min: float = 0.0,
                       static_alpha_softness: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend low-alpha primary pixels with static fill, optionally gated by static confidence."""
    gate = primary_alpha.clamp(0.0, 1.0).pow(max(float(fill_alpha_power), 1e-6))
    fill_w = 1.0 - gate
    if static_alpha_min > 0:
        fill_w = fill_w * _static_fill_confidence(
            static_alpha.clamp(0.0, 1.0),
            float(static_alpha_min),
            float(static_alpha_softness),
        )
    rgb = primary_rgb * (1.0 - fill_w) + static_rgb * fill_w
    alpha = primary_alpha * (1.0 - fill_w) + static_alpha * fill_w
    return rgb, alpha


def _sample_depth_support_window(
    fg_view: torch.Tensor,
    z_view: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    tol: float,
    radius_px: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample local silhouette/depth support around projected query points.

    ``radius_px=0`` preserves nearest-pixel behavior. Larger radii use a local
    depth min/max window so tiny projection/depth errors on thin structures do
    not immediately become hard support conflicts.
    """
    h, w = z_view.shape[-2:]
    ui = u.round().long().clamp(0, w - 1)
    vi = v.round().long().clamp(0, h - 1)
    fg_bool = fg_view.to(device=z_view.device).bool()
    valid_view = (
        torch.isfinite(z_view)
        & (z_view > 1e-6)
        & (z_view < 1e5)
        & fg_bool
    )
    if radius_px <= 0:
        sampled_fg = fg_bool[vi, ui]
        sampled_valid = valid_view[vi, ui]
        sampled_z = z_view[vi, ui]
        depth_match = sampled_valid & ((z - sampled_z).abs() <= tol)
        front_conflict = sampled_valid & (z < sampled_z - tol)
        bidir_conflict = sampled_valid & ((z - sampled_z).abs() > tol)
        return sampled_fg, depth_match, front_conflict, bidir_conflict

    radius = int(radius_px)
    k = 2 * radius + 1
    valid_f = valid_view.to(dtype=z_view.dtype)[None, None]
    fg_f = fg_bool.to(dtype=z_view.dtype)[None, None]
    local_fg = F.max_pool2d(fg_f, kernel_size=k, stride=1, padding=radius)[0, 0] > 0.5
    local_valid = F.max_pool2d(valid_f, kernel_size=k, stride=1, padding=radius)[0, 0] > 0.5
    neg_big = z_view.new_full(z_view.shape, -1e6)
    local_min = -F.max_pool2d(
        torch.where(valid_view, -z_view, neg_big)[None, None],
        kernel_size=k,
        stride=1,
        padding=radius,
    )[0, 0]
    local_max = F.max_pool2d(
        torch.where(valid_view, z_view, neg_big)[None, None],
        kernel_size=k,
        stride=1,
        padding=radius,
    )[0, 0]
    sampled_fg = local_fg[vi, ui]
    sampled_valid = local_valid[vi, ui]
    min_z = local_min[vi, ui]
    max_z = local_max[vi, ui]
    depth_match = sampled_valid & (z >= min_z - tol) & (z <= max_z + tol)
    front_conflict = sampled_valid & (z < min_z - tol)
    bidir_conflict = sampled_valid & ((z < min_z - tol) | (z > max_z + tol))
    return sampled_fg, depth_match, front_conflict, bidir_conflict


def _local_valid_median_map(
    values: torch.Tensor,
    valid: torch.Tensor,
    radius_px: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local median over valid pixels for ``(V,H,W)`` tensors."""
    if radius_px <= 0:
        return values, valid.bool()
    radius = int(radius_px)
    k = 2 * radius + 1
    v, h, w = values.shape
    patches = F.unfold(
        values[:, None], kernel_size=k, stride=1, padding=radius
    ).view(v, k * k, h * w)
    mask = F.unfold(
        valid[:, None].to(dtype=values.dtype), kernel_size=k, stride=1, padding=radius
    ).view(v, k * k, h * w) > 0.5
    sentinel = values.new_full((), 1e6)
    sorted_vals = torch.where(mask, patches, sentinel).sort(dim=1).values
    counts = mask.sum(dim=1)
    gather_idx = ((counts.clamp_min(1) - 1) // 2).unsqueeze(1)
    med = sorted_vals.gather(1, gather_idx).squeeze(1).view(v, h, w)
    return med, counts.view(v, h, w) > 0


def _depth_multiview_support_maps(
    depths: torch.Tensor,
    fg: torch.Tensor,
    K_all: torch.Tensor,
    c2w_all: torch.Tensor,
    radius: float,
    tol_frac: float,
    max_refs: int,
    radius_px: int = 0,
) -> torch.Tensor:
    """Per-view geometric support features from the other conditioning views.

    Returns ``(V,4,H,W)`` channels: depth support, conflict, front-conflict, and
    projected-reference coverage. These are deterministic inputs for learned
    depth/visibility heads, not a post-render edit.
    """
    v_count, h, w = depths.shape
    dtype, device = depths.dtype, depths.device
    fg_bool = fg[..., 0] > 0.5 if fg.ndim == 4 else fg > 0.5
    valid = torch.isfinite(depths) & (depths > 1e-6) & (depths < 1e5) & fg_bool
    out = depths.new_zeros((v_count, 4, h, w))
    if v_count <= 1:
        return out

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)
    w2c_all = torch.linalg.inv(c2w_all)
    centers = c2w_all[:, :3, 3]
    dists = torch.cdist(centers, centers)
    ref_limit = v_count - 1 if max_refs <= 0 else min(max(int(max_refs), 1), v_count - 1)
    tol = max(float(tol_frac) * float(radius), 1e-6)

    for src_i in range(v_count):
        valid_flat = valid[src_i].reshape(-1)
        if not valid_flat.any():
            continue
        flat_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
        z_src = depths[src_i].reshape(-1)[flat_idx]
        k_src = K_all[src_i].to(device=device, dtype=dtype)
        cam_x = (x_flat[flat_idx] - k_src[0, 2]) / k_src[0, 0].clamp_min(1e-6) * z_src
        cam_y = (y_flat[flat_idx] - k_src[1, 2]) / k_src[1, 1].clamp_min(1e-6) * z_src
        pts_cam = torch.stack([cam_x, cam_y, z_src], dim=-1)
        c2w_src = c2w_all[src_i].to(device=device, dtype=dtype)
        pts_world = pts_cam @ c2w_src[:3, :3].T + c2w_src[:3, 3]

        support = z_src.new_zeros(z_src.shape)
        conflict = z_src.new_zeros(z_src.shape)
        front = z_src.new_zeros(z_src.shape)
        coverage = z_src.new_zeros(z_src.shape)
        refs = torch.argsort(dists[src_i])
        refs = refs[refs != src_i][:ref_limit]
        for ref_t in refs:
            ref_i = int(ref_t.item())
            w2c_ref = w2c_all[ref_i].to(device=device, dtype=dtype)
            cam = pts_world @ w2c_ref[:3, :3].T + w2c_ref[:3, 3]
            z_ref = cam[:, 2]
            k_ref = K_all[ref_i].to(device=device, dtype=dtype)
            u = k_ref[0, 0] * (cam[:, 0] / z_ref.clamp_min(1e-6)) + k_ref[0, 2]
            vv = k_ref[1, 1] * (cam[:, 1] / z_ref.clamp_min(1e-6)) + k_ref[1, 2]
            inb = (z_ref > 1e-6) & (u >= 0) & (u <= w - 1) & (vv >= 0) & (vv <= h - 1)
            if not inb.any():
                continue
            local_fg, depth_match, front_conflict, bidir_conflict = _sample_depth_support_window(
                fg_bool[ref_i],
                depths[ref_i],
                u[inb],
                vv[inb],
                z_ref[inb],
                tol,
                radius_px,
            )
            idx = torch.nonzero(inb, as_tuple=False).squeeze(1)
            coverage[idx] += 1.0
            support[idx] += depth_match.to(dtype=dtype)
            conflict[idx] += ((~local_fg) | bidir_conflict).to(dtype=dtype)
            front[idx] += front_conflict.to(dtype=dtype)

        denom = coverage.clamp_min(1.0)
        out_flat = out[src_i].reshape(4, -1)
        out_flat[0, flat_idx] = support / denom
        out_flat[1, flat_idx] = conflict / denom
        out_flat[2, flat_idx] = front / denom
        out_flat[3, flat_idx] = coverage / float(max(ref_limit, 1))
    return out.clamp(0.0, 1.0)


def _surface_confidence_protect_mask(
    mv_features: torch.Tensor | None,
    support_min: float = 0.0,
    conflict_max: float = 1.0,
    coverage_min: float = 0.0,
) -> torch.Tensor | None:
    """Mask source pixels whose splats should not be opacity-gated.

    ``mv_features`` follows ``_depth_multiview_support_maps`` layout:
    support, conflict, front-conflict, coverage. The mask is intentionally
    conservative: when enabled, all active thresholds must pass before a splat
    is protected. This lets learned cleanup act on weak/contradictory fringe
    splats while preserving high-confidence surface splats.
    """
    if mv_features is None:
        return None
    if mv_features.shape[0] < 4:
        raise ValueError("surface confidence protect mask expects 4 support channels")

    use_support = support_min > 0.0
    use_conflict = conflict_max < 1.0
    use_coverage = coverage_min > 0.0
    if not (use_support or use_conflict or use_coverage):
        return None

    support = mv_features[0]
    conflict = mv_features[1]
    coverage = mv_features[3]
    protect = torch.ones_like(support, dtype=torch.bool)
    if use_support:
        protect = protect & (support >= float(support_min))
    if use_conflict:
        protect = protect & (conflict <= float(conflict_max))
    if use_coverage:
        protect = protect & (coverage >= float(coverage_min))
    return protect


def _alpha_mask_stats(alpha: torch.Tensor, gt_mask: torch.Tensor) -> dict[str, float]:
    """Silhouette diagnostics for fog/fringe failures.

    PSNR is computed only on foreground pixels, so it can miss background alpha
    leakage and silhouette debris. These metrics are intentionally simple and
    thresholded so probe runs can be compared from `eval_metrics.jsonl`.
    """
    a = alpha.detach().clamp(0.0, 1.0)
    gt = gt_mask.detach().to(device=a.device)
    gt_bool = gt > 0.5
    gt_f = gt_bool.to(dtype=a.dtype)
    bg_f = 1.0 - gt_f
    fg_area = gt_f.sum().clamp_min(1.0)
    bg_area = bg_f.sum().clamp_min(1.0)
    pred01 = a > 0.1
    pred05 = a > 0.5
    inter = (pred05 & gt_bool).to(dtype=a.dtype).sum()
    union = (pred05 | gt_bool).to(dtype=a.dtype).sum().clamp_min(1.0)
    return {
        "alpha_l1": float((a - gt_f).abs().mean()),
        "alpha_bg_mean": float((a * bg_f).sum() / bg_area),
        "alpha_fg_miss": float(((1.0 - a) * gt_f).sum() / fg_area),
        "alpha_fp_gt_0_1": float((pred01 & ~gt_bool).to(dtype=a.dtype).sum() / bg_area),
        "alpha_fp_gt_0_5": float((pred05 & ~gt_bool).to(dtype=a.dtype).sum() / bg_area),
        "alpha_fn_le_0_5": float(((~pred05) & gt_bool).to(dtype=a.dtype).sum() / fg_area),
        "alpha_iou_0_5": float(inter / union),
    }


def _learned_iblend_feature_channels(topk: int) -> int:
    # Per candidate: object RGB, alpha, deterministic normalized weight,
    # absolute source-support score, expected depth, and scalar view weight.
    # Shared: aggregate/static render, their color difference, fill prior,
    # xy coordinates, and background value.
    return 8 * max(int(topk), 1) + 15


class LearnedFillBlend(torch.nn.Module):
    """Small shared image-space blend head for view-conditioned fill.

    The head predicts a residual logit on top of the deterministic nearest-fill
    alpha prior, plus an optional RGB correction. Zero init makes step 0 exactly
    match the hand-coded fill rule.
    """

    def __init__(self, in_channels: int = LEARNED_FILL_FEATURE_CHANNELS,
                 hidden: int = 32, layers: int = 3,
                 rgb_residual_scale: float = 0.0):
        super().__init__()
        hidden = max(int(hidden), 8)
        layers = max(int(layers), 1)
        blocks = []
        cin = in_channels
        for _ in range(max(layers - 1, 0)):
            blocks.extend([
                torch.nn.Conv2d(cin, hidden, kernel_size=3, padding=1),
                torch.nn.GroupNorm(min(8, hidden), hidden),
                torch.nn.GELU(),
            ])
            cin = hidden
        out_channels = 1 + (3 if rgb_residual_scale > 0 else 0)
        head = torch.nn.Conv2d(cin, out_channels, kernel_size=3, padding=1)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        blocks.append(head)
        self.net = torch.nn.Sequential(*blocks)
        self.rgb_residual_scale = max(float(rgb_residual_scale), 0.0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.net(features)
        rgb_delta = None
        if self.rgb_residual_scale > 0:
            rgb_delta = torch.tanh(out[:, 1:4]) * self.rgb_residual_scale
        return out[:, 0:1], rgb_delta


class LearnedIblendFillBlend(torch.nn.Module):
    """Shared residual visibility head for multi-anchor iblend fill.

    The deterministic `iblend_fill` path already gives a strong prior. This head
    only predicts residual logits on top of that prior: K candidate-anchor
    residuals plus one fill-gate residual. The final layer is zero-initialized,
    so step 0 exactly matches the deterministic renderer.
    """

    def __init__(self, topk: int, hidden: int = 32, layers: int = 3,
                 rgb_residual_scale: float = 0.0):
        super().__init__()
        self.topk = max(int(topk), 1)
        self.in_channels = _learned_iblend_feature_channels(self.topk)
        hidden = max(int(hidden), 8)
        layers = max(int(layers), 1)
        blocks = []
        cin = self.in_channels
        for _ in range(max(layers - 1, 0)):
            blocks.extend([
                torch.nn.Conv2d(cin, hidden, kernel_size=3, padding=1),
                torch.nn.GroupNorm(min(8, hidden), hidden),
                torch.nn.GELU(),
            ])
            cin = hidden
        out_channels = self.topk + 1 + (3 if rgb_residual_scale > 0 else 0)
        head = torch.nn.Conv2d(cin, out_channels, kernel_size=3, padding=1)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        blocks.append(head)
        self.net = torch.nn.Sequential(*blocks)
        self.rgb_residual_scale = max(float(rgb_residual_scale), 0.0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"learned-iblend feature bug: got {features.shape[1]} channels, "
                f"expected {self.in_channels}"
            )
        out = self.net(features)
        cand_delta = out[:, :self.topk]
        fill_delta = out[:, self.topk:self.topk + 1]
        rgb_delta = None
        if self.rgb_residual_scale > 0:
            rgb_delta = torch.tanh(out[:, self.topk + 1:self.topk + 4]) * self.rgb_residual_scale
        return cand_delta, fill_delta, rgb_delta


class LearnedIblendFillUNet(torch.nn.Module):
    """U-Net visibility/fill head over the multi-anchor candidate stack.

    The small CNN head is intentionally local. This variant adds a modest
    encoder/decoder with skip connections so the shared head can use object
    silhouette context when deciding whether a candidate shell or static fill is
    trustworthy. The final layer is still zero-initialized, so step 0 exactly
    matches the deterministic renderer.
    """

    def __init__(self, topk: int, hidden: int = 32,
                 rgb_residual_scale: float = 0.0):
        super().__init__()
        self.topk = max(int(topk), 1)
        self.in_channels = _learned_iblend_feature_channels(self.topk)
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = self._block(self.in_channels, hidden)
        self.down1 = self._down(hidden, h2)
        self.down2 = self._down(h2, h4)
        self.mid = self._block(h4, h4)
        self.up1 = self._block(h4 + h2, h2)
        self.up2 = self._block(h2 + hidden, hidden)
        out_channels = self.topk + 1 + (3 if rgb_residual_scale > 0 else 0)
        self.head = torch.nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.rgb_residual_scale = max(float(rgb_residual_scale), 0.0)

    @staticmethod
    def _block(cin: int, cout: int) -> torch.nn.Sequential:
        groups = min(8, cout)
        return torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            torch.nn.GroupNorm(groups, cout),
            torch.nn.GELU(),
            torch.nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            torch.nn.GroupNorm(groups, cout),
            torch.nn.GELU(),
        )

    @staticmethod
    def _down(cin: int, cout: int) -> torch.nn.Sequential:
        groups = min(8, cout)
        return torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
            torch.nn.GroupNorm(groups, cout),
            torch.nn.GELU(),
            torch.nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            torch.nn.GroupNorm(groups, cout),
            torch.nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"learned-iblend-unet feature bug: got {features.shape[1]} channels, "
                f"expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        e2 = self.down1(e1)
        x = self.down2(e2)
        x = self.mid(x)
        x = torch.nn.functional.interpolate(
            x, size=e2.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up1(torch.cat([x, e2], dim=1))
        x = torch.nn.functional.interpolate(
            x, size=e1.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up2(torch.cat([x, e1], dim=1))
        out = self.head(x)
        cand_delta = out[:, :self.topk]
        fill_delta = out[:, self.topk:self.topk + 1]
        rgb_delta = None
        if self.rgb_residual_scale > 0:
            rgb_delta = torch.tanh(out[:, self.topk + 1:self.topk + 4]) * self.rgb_residual_scale
        return cand_delta, fill_delta, rgb_delta


class DepthRefineUNet(torch.nn.Module):
    """Shared feed-forward depth-prior correction head.

    Inputs are per-conditioning-view RGB*mask, mask, normalized prior ray-depth,
    and prior validity. The head predicts a residual logit for the normalized
    depth fraction. Its final layer is zero-initialized, so enabling the module
    reproduces the input depth prior at step 0.
    """

    in_channels = 6

    def __init__(self, hidden: int = 32, in_channels: int | None = None):
        super().__init__()
        self.in_channels = int(in_channels or self.in_channels)
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = self._block(self.in_channels, hidden)
        self.down1 = self._down(hidden, h2)
        self.down2 = self._down(h2, h4)
        self.mid = self._block(h4, h4)
        self.up1 = self._block(h4 + h2, h2)
        self.up2 = self._block(h2 + hidden, hidden)
        self.head = torch.nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    @staticmethod
    def _block(cin: int, cout: int) -> torch.nn.Sequential:
        groups = min(8, cout)
        return torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            torch.nn.GroupNorm(groups, cout),
            torch.nn.GELU(),
            torch.nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            torch.nn.GroupNorm(groups, cout),
            torch.nn.GELU(),
        )

    @staticmethod
    def _down(cin: int, cout: int) -> torch.nn.Sequential:
        groups = min(8, cout)
        return torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
            torch.nn.GroupNorm(groups, cout),
            torch.nn.GELU(),
            torch.nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            torch.nn.GroupNorm(groups, cout),
            torch.nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"depth-refine feature bug: got {features.shape[1]} channels, "
                f"expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        e2 = self.down1(e1)
        x = self.down2(e2)
        x = self.mid(x)
        x = torch.nn.functional.interpolate(
            x, size=e2.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up1(torch.cat([x, e2], dim=1))
        x = torch.nn.functional.interpolate(
            x, size=e1.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up2(torch.cat([x, e1], dim=1))
        return self.head(x)


class SupportGateUNet(torch.nn.Module):
    """Shared feed-forward opacity gate for RGBD anchor splats.

    Inputs match the depth-refine head: RGB*mask, mask, normalized prior depth,
    and prior-depth validity. The output is a residual logit over a near-one
    opacity gate, so enabling the head preserves the deterministic renderer at
    step 0 while giving training a direct way to suppress bad depth support.
    """

    in_channels = 6

    def __init__(self, hidden: int = 24):
        super().__init__()
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = DepthRefineUNet._block(self.in_channels, hidden)
        self.down1 = DepthRefineUNet._down(hidden, h2)
        self.down2 = DepthRefineUNet._down(h2, h4)
        self.mid = DepthRefineUNet._block(h4, h4)
        self.up1 = DepthRefineUNet._block(h4 + h2, h2)
        self.up2 = DepthRefineUNet._block(h2 + hidden, hidden)
        self.head = torch.nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"support-gate feature bug: got {features.shape[1]} channels, "
                f"expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        e2 = self.down1(e1)
        x = self.down2(e2)
        x = self.mid(x)
        x = torch.nn.functional.interpolate(
            x, size=e2.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up1(torch.cat([x, e2], dim=1))
        x = torch.nn.functional.interpolate(
            x, size=e1.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up2(torch.cat([x, e1], dim=1))
        return self.head(x)


class SurfaceConfidenceUNet(torch.nn.Module):
    """Source-splat confidence head with explicit multi-view support inputs.

    This differs from the image-space fill heads: it gates the RGBD source splats
    before static voxel fusion, so the learned signal changes the emitted 3DGS
    instead of only compositing a target-view render. The final layer is
    zero-initialized and the downstream gate is normalized by its init
    probability, so enabling the module preserves the deterministic baseline at
    step 0.
    """

    # RGB*mask, mask, normalized depth, depth-valid, and 4 geometric support maps.
    in_channels = 10

    def __init__(self, hidden: int = 24, in_channels: int | None = None):
        super().__init__()
        self.in_channels = int(in_channels or self.in_channels)
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = DepthRefineUNet._block(self.in_channels, hidden)
        self.down1 = DepthRefineUNet._down(hidden, h2)
        self.down2 = DepthRefineUNet._down(h2, h4)
        self.mid = DepthRefineUNet._block(h4, h4)
        self.up1 = DepthRefineUNet._block(h4 + h2, h2)
        self.up2 = DepthRefineUNet._block(h2 + hidden, hidden)
        self.head = torch.nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"surface-confidence feature bug: got {features.shape[1]} channels, "
                f"expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        e2 = self.down1(e1)
        x = self.down2(e2)
        x = self.mid(x)
        x = torch.nn.functional.interpolate(
            x, size=e2.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up1(torch.cat([x, e2], dim=1))
        x = torch.nn.functional.interpolate(
            x, size=e1.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up2(torch.cat([x, e1], dim=1))
        return self.head(x)


class SurfaceRefineUNet(torch.nn.Module):
    """Shared high-resolution source-surface refiner for RGBD splats.

    This head edits the emitted 3DGS candidates before voxel fusion instead of
    post-processing a target-view render. It sees the same image-resolution RGBD
    and multi-view support features as the surface-confidence gate, then emits
    bounded residuals for opacity, scale, and RGB. The final layer is
    zero-initialized, so enabling it exactly preserves the deterministic prior
    until learned.
    """

    # RGB*mask, mask, normalized depth, depth-valid, and 4 support maps.
    in_channels = 10

    def __init__(self, hidden: int = 32, in_channels: int | None = None):
        super().__init__()
        self.in_channels = int(in_channels or self.in_channels)
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = DepthRefineUNet._block(self.in_channels, hidden)
        self.down1 = DepthRefineUNet._down(hidden, h2)
        self.down2 = DepthRefineUNet._down(h2, h4)
        self.mid = DepthRefineUNet._block(h4, h4)
        self.up1 = DepthRefineUNet._block(h4 + h2, h2)
        self.up2 = DepthRefineUNet._block(h2 + hidden, hidden)
        self.head = torch.nn.Conv2d(hidden, 5, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"surface-refine feature bug: got {features.shape[1]} channels, "
                f"expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        e2 = self.down1(e1)
        x = self.down2(e2)
        x = self.mid(x)
        x = torch.nn.functional.interpolate(
            x, size=e2.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up1(torch.cat([x, e2], dim=1))
        x = torch.nn.functional.interpolate(
            x, size=e1.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up2(torch.cat([x, e1], dim=1))
        return self.head(x)


def _fusion_candidate_features(params: dict,
                               radius: float | None,
                               ref_count: int,
                               include_coords: bool = False,
                               include_rich: bool = False,
                               include_voxel: bool = False,
                               include_neighbor: bool = False,
                               neighbor_radius: int = 1,
                               voxel_size: float | None = None) -> torch.Tensor:
    """Low-dimensional per-splat features for learned pre-fusion selection.

    The features are deliberately built from deterministic RGBD-fusion signals,
    not target-view render errors. This keeps the head usable at inference and
    makes it a bounded correction to the hand-coded prior.
    """
    opacity = params["opacity"]
    n = opacity.shape[0]
    dtype, device = opacity.dtype, opacity.device
    radius_norm = max(float(radius) if radius is not None else 1.0, 1e-6)
    ref_norm = max(float(ref_count), 1.0)

    def col(key: str, default: float = 0.0) -> torch.Tensor:
        value = params.get(key)
        if value is None:
            return opacity.new_full((n, 1), float(default))
        value = value.reshape(n, -1)[:, :1]
        return value.to(device=device, dtype=dtype)

    rgb = params["rgb"].reshape(n, -1)[:, :3].to(device=device, dtype=dtype).clamp(0.0, 1.0)
    depth = col("depth") / radius_norm
    if "scale" in params:
        scale = params["scale"].reshape(n, -1).to(device=device, dtype=dtype).mean(dim=-1, keepdim=True)
        scale = (scale / radius_norm) * 1000.0
    else:
        scale = opacity.new_zeros((n, 1))
    score_raw = col("_fusion_score")
    support_raw = col("_fusion_support")
    conflict_raw = col("_fusion_conflict")
    coverage_raw = col("_fusion_coverage")
    color_support_raw = col("_fusion_color_support")
    score = score_raw / ref_norm
    support = support_raw / ref_norm
    conflict = conflict_raw / ref_norm
    coverage = coverage_raw / ref_norm
    color_support = color_support_raw / ref_norm
    detail = col("_fusion_detail")

    features = [
        rgb,
        opacity.reshape(n, -1)[:, :1].to(device=device, dtype=dtype).clamp(0.0, 1.0),
        depth.clamp(0.0, 8.0) / 8.0,
        scale.clamp(0.0, 4.0) / 4.0,
        score.clamp(-2.0, 2.0) / 2.0,
        support.clamp(0.0, 2.0) / 2.0,
        conflict.clamp(0.0, 2.0) / 2.0,
        coverage.clamp(0.0, 2.0) / 2.0,
        color_support.clamp(0.0, 2.0) / 2.0,
        detail.clamp(0.0, 1.0),
    ]
    if include_rich:
        cov_den = coverage_raw.clamp_min(1.0)
        support_ratio = (support_raw / cov_den).clamp(0.0, 1.0)
        conflict_ratio = (conflict_raw / cov_den).clamp(0.0, 1.0)
        color_support_ratio = (color_support_raw / support_raw.clamp_min(1.0)).clamp(0.0, 1.0)
        net_support = ((support_raw - conflict_raw) / cov_den).clamp(-1.0, 1.0)
        score_per_coverage = (score_raw / cov_den).clamp(-2.0, 2.0) / 2.0
        depth_error = (col("_fusion_depth_error") / cov_den).clamp(0.0, 4.0) / 4.0
        color_error = (col("_fusion_color_error") / cov_den).clamp(0.0, 1.0)
        front_ratio = (col("_fusion_front_conflict") / cov_den).clamp(0.0, 1.0)
        silhouette_ratio = (col("_fusion_silhouette_conflict") / cov_den).clamp(0.0, 1.0)
        features.extend([
            support_ratio,
            conflict_ratio,
            color_support_ratio,
            net_support,
            score_per_coverage,
            depth_error,
            color_error,
            front_ratio,
            silhouette_ratio,
        ])
    if include_coords:
        mean = params["mean"].reshape(n, -1)[:, :3].to(device=device, dtype=dtype)
        mean_norm = (mean / radius_norm).clamp(-2.0, 2.0) / 2.0
        radial = (mean / radius_norm).norm(dim=-1, keepdim=True).clamp(0.0, 2.0) / 2.0
        source = params.get("_fusion_source")
        if source is None:
            phase = opacity.new_zeros((n, 1))
        else:
            phase = source.reshape(n, -1)[:, :1].to(device=device, dtype=dtype)
            phase = phase / ref_norm
        angle = phase * (2.0 * math.pi)
        features.extend([mean_norm, radial, torch.sin(angle), torch.cos(angle)])
    if include_voxel:
        mean = params["mean"].reshape(n, -1)[:, :3].to(device=device, dtype=dtype)
        voxel = max(
            float(voxel_size) if voxel_size is not None else radius_norm * 0.01,
            1e-6,
        )
        if n == 0:
            features.append(opacity.new_zeros((0, FUSION_CANDIDATE_VOXEL_FEATURE_CHANNELS)))
            return torch.cat(features, dim=-1)

        with torch.no_grad():
            q = torch.floor(mean.detach() / voxel).to(torch.int64)
            q = q - q.amin(dim=0, keepdim=True)
            dims = q.amax(dim=0) + 1
            key = q[:, 0] + dims[0] * (q[:, 1] + dims[1] * q[:, 2])
            _, inverse = torch.unique(key, sorted=False, return_inverse=True)
            n_vox = int(inverse.max().item()) + 1 if inverse.numel() else 0

        if n_vox <= 0:
            voxel_features = opacity.new_zeros((n, FUSION_CANDIDATE_VOXEL_FEATURE_CHANNELS))
        else:
            inverse_col = inverse[:, None].expand(n, 1)
            count = torch.bincount(inverse, minlength=n_vox).to(device=device, dtype=dtype)[:, None]
            count_safe = count.clamp_min(1.0)

            def voxel_mean(values: torch.Tensor) -> torch.Tensor:
                out = values.new_zeros((n_vox, 1))
                out.scatter_add_(0, inverse_col, values)
                return out / count_safe

            def voxel_max(values: torch.Tensor) -> torch.Tensor:
                out = values.new_full((n_vox, 1), -1e9)
                out.scatter_reduce_(0, inverse_col, values, reduce="amax", include_self=True)
                return out

            score_mean = voxel_mean(score_raw)
            score_max = voxel_max(score_raw)
            support_mean = voxel_mean(support_raw)
            conflict_mean = voxel_mean(conflict_raw)
            opacity_col = opacity.reshape(n, -1)[:, :1].to(device=device, dtype=dtype)
            opacity_mean = voxel_mean(opacity_col)

            voxel_count = (
                torch.log1p(count[inverse].clamp_min(0.0))
                / math.log1p(32.0)
            ).clamp(0.0, 1.0)
            score_gap = ((score_raw - score_max[inverse]) / ref_norm).clamp(-2.0, 0.0) / 2.0
            score_center = ((score_raw - score_mean[inverse]) / ref_norm).clamp(-2.0, 2.0) / 2.0
            support_center = ((support_raw - support_mean[inverse]) / ref_norm).clamp(-2.0, 2.0) / 2.0
            conflict_center = ((conflict_raw - conflict_mean[inverse]) / ref_norm).clamp(-2.0, 2.0) / 2.0
            opacity_center = (opacity_col - opacity_mean[inverse]).clamp(-1.0, 1.0)
            voxel_center = (torch.floor(mean.detach() / voxel).to(dtype=dtype) + 0.5) * voxel
            local_offset = ((mean - voxel_center) / voxel).clamp(-1.0, 1.0)
            voxel_features = torch.cat([
                voxel_count,
                score_gap,
                score_center,
                support_center,
                conflict_center,
                opacity_center,
                local_offset,
            ], dim=-1)
        features.append(voxel_features)
    if include_neighbor:
        mean = params["mean"].reshape(n, -1)[:, :3].to(device=device, dtype=dtype)
        voxel = max(
            float(voxel_size) if voxel_size is not None else radius_norm * 0.01,
            1e-6,
        )
        r = max(int(neighbor_radius), 0)
        if n == 0:
            features.append(opacity.new_zeros((0, FUSION_CANDIDATE_NEIGHBOR_FEATURE_CHANNELS)))
            return torch.cat(features, dim=-1)

        cov_den = coverage_raw.clamp_min(1.0)
        opacity_col = opacity.reshape(n, -1)[:, :1].to(device=device, dtype=dtype).clamp(0.0, 1.0)
        depth_error = (col("_fusion_depth_error") / cov_den).clamp(0.0, 4.0) / 4.0
        color_error = (col("_fusion_color_error") / cov_den).clamp(0.0, 1.0)
        signal = torch.cat([
            score.clamp(-2.0, 2.0) / 2.0,
            support.clamp(0.0, 2.0) / 2.0,
            conflict.clamp(0.0, 2.0) / 2.0,
            coverage.clamp(0.0, 2.0) / 2.0,
            color_support.clamp(0.0, 2.0) / 2.0,
            depth_error,
            color_error,
            opacity_col,
            depth.clamp(0.0, 8.0) / 8.0,
            scale.clamp(0.0, 4.0) / 4.0,
        ], dim=-1)
        if signal.shape[-1] != FUSION_CANDIDATE_NEIGHBOR_SIGNAL_CHANNELS:
            raise RuntimeError(
                "fusion-candidate neighbor feature bug: "
                f"got {signal.shape[-1]} signal channels"
            )
        with torch.no_grad():
            q = torch.floor(mean.detach() / voxel).to(torch.int64)
            q = q - q.amin(dim=0, keepdim=True)
            dims = q.amax(dim=0) + 1
            key = q[:, 0] + dims[0] * (q[:, 1] + dims[1] * q[:, 2])
            unique_keys, inverse = torch.unique(key, sorted=True, return_inverse=True)
            n_vox = int(unique_keys.numel())
        if n_vox <= 0:
            neighbor_features = opacity.new_zeros((n, FUSION_CANDIDATE_NEIGHBOR_FEATURE_CHANNELS))
        else:
            signal_vox = signal.new_zeros((n_vox, signal.shape[-1]))
            signal_vox.scatter_add_(0, inverse[:, None].expand(-1, signal.shape[-1]), signal)
            count = torch.bincount(inverse, minlength=n_vox).to(device=device, dtype=dtype)[:, None]
            signal_vox = signal_vox / count.clamp_min(1.0)

            stride01 = dims[0] * dims[1]
            q2 = unique_keys // stride01
            rem = unique_keys - q2 * stride01
            q1 = rem // dims[0]
            q0 = rem - q1 * dims[0]
            q_vox = torch.stack([q0, q1, q2], dim=-1)

            neigh_sum = signal_vox.new_zeros(signal_vox.shape)
            neigh_count = signal_vox.new_zeros((n_vox, 1))
            offsets = range(-r, r + 1)
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        qn = q_vox + q_vox.new_tensor([dx, dy, dz])
                        valid = (
                            (qn[:, 0] >= 0) & (qn[:, 0] < dims[0])
                            & (qn[:, 1] >= 0) & (qn[:, 1] < dims[1])
                            & (qn[:, 2] >= 0) & (qn[:, 2] < dims[2])
                        )
                        if not bool(valid.any()):
                            continue
                        dst = torch.nonzero(valid, as_tuple=False).squeeze(1)
                        qv = qn[dst]
                        nk = qv[:, 0] + dims[0] * (qv[:, 1] + dims[1] * qv[:, 2])
                        pos = torch.searchsorted(unique_keys, nk)
                        pos_safe = pos.clamp_max(max(int(unique_keys.numel()) - 1, 0))
                        hit = (pos < unique_keys.numel()) & (unique_keys[pos_safe] == nk)
                        if not bool(hit.any()):
                            continue
                        dst_h = dst[hit]
                        src_h = pos[hit]
                        neigh_sum.index_add_(0, dst_h, signal_vox[src_h])
                        neigh_count.index_add_(
                            0,
                            dst_h,
                            torch.ones(dst_h.shape[0], 1, device=device, dtype=dtype),
                        )
            neigh_mean = neigh_sum / neigh_count.clamp_min(1.0)
            max_neighbors = max((2 * r + 1) ** 3, 1)
            count_norm = (neigh_count[inverse] / float(max_neighbors)).clamp(0.0, 1.0)
            neigh_per_splat = neigh_mean[inverse]
            neighbor_features = torch.cat([
                neigh_per_splat,
                signal - neigh_per_splat,
                count_norm,
            ], dim=-1)
        features.append(neighbor_features)
    return torch.cat(features, dim=-1)


class FusionCandidateGate(torch.nn.Module):
    """Tiny per-splat head that learns pre-voxel visibility/score corrections.

    `voxel_fuse_params` uses detached score ranking, so a learned score alone is
    not render-loss trainable. This head emits both a score residual and a
    differentiable opacity-gate residual. The last layer is zero initialized, so
    step 0 exactly matches the deterministic RGBD/DA3 fusion prior.
    """

    def __init__(self, in_channels: int = FUSION_CANDIDATE_FEATURE_CHANNELS,
                 hidden: int = 0, layers: int = 2):
        super().__init__()
        self.in_channels = int(in_channels)
        hidden = max(int(hidden), 0)
        layers = max(int(layers), 1)
        if hidden <= 0:
            self.body = torch.nn.Identity()
            self.head = torch.nn.Linear(self.in_channels, 2)
        else:
            blocks = []
            cin = self.in_channels
            for _ in range(max(layers - 1, 0)):
                blocks.extend([
                    torch.nn.Linear(cin, hidden),
                    torch.nn.LayerNorm(hidden),
                    torch.nn.GELU(),
                ])
                cin = hidden
            self.body = torch.nn.Sequential(*blocks)
            self.head = torch.nn.Linear(cin, 2)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.in_channels:
            raise RuntimeError(
                f"fusion-candidate feature bug: got {features.shape[-1]} channels, "
                f"expected {self.in_channels}"
            )
        out = self.head(self.body(features))
        return out[:, 0:1], out[:, 1:2]


def _collate_first(batch):
    return batch[0]          # batch_size=1 -> hand back the single sample dict, un-batched


def _load_safe(ds, n):
    """Load ds[0..n-1], skipping any object that fails (e.g. a broken obj missing a frame —
    animals_v4 test has one such, the chicken missing frame_024)."""
    out = []
    for i in range(n):
        try:
            out.append(ds[i])
        except Exception as ex:
            entry = ds.entries[i]
            uid = entry if isinstance(entry, str) else entry.get("uid", "<unknown>")
            print(f"[phase2] skip {uid[:10]} (load failed: {type(ex).__name__})",
                  flush=True)
    return out


def _have_wandb_creds() -> bool:
    import os
    if os.environ.get("WANDB_API_KEY"):
        return True
    nr = os.path.expanduser("~/.netrc")
    return os.path.exists(nr) and "api.wandb.ai" in open(nr).read()


def _init_wandb(args, n_params, n_objects, n_gauss):
    """Returns a wandb run, or None if disabled. Online falls back to offline if no creds
    (so a background run never hangs on a login prompt)."""
    if args.wandb_mode == "disabled":
        return None
    import wandb
    mode = args.wandb_mode
    if mode == "online" and not _have_wandb_creds():
        print("[phase2] no W&B creds -> offline (sync later with `wandb sync`)", flush=True)
        mode = "offline"
    gk = f"{n_gauss // 1000}k"   # actual Gaussian count, e.g. 98k (ups4) / 393k (ups5)
    return wandb.init(project="latent2splat", entity=args.wandb_entity,
                      name=f"phase2_{gk}_cap{args.scale_cap_frac:g}_lr{args.lr:g}_{int(time.time())}",
                      dir=args.out_dir, mode=mode, tags=["phase2", gk, "generalization"],
                      config={**vars(args), "n_params": n_params, "n_objects": n_objects})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)        # OPTIMIZER steps (each = --accum objects)
    ap.add_argument("--accum", type=int, default=8)           # objects accumulated per optimizer step (eff. batch)
    ap.add_argument("--eval_every", type=int, default=250)    # in optimizer steps
    ap.add_argument("--log_every", type=int, default=20)      # in optimizer steps
    ap.add_argument("--k_views", type=int, default=8)
    ap.add_argument("--anchor_views", type=int, default=1)          # feed-forward multi-view anchors to concatenate
    ap.add_argument("--anchor_render_mode", default="concat",
                    choices=["concat", "nearest", "blend", "zselect", "tsdf",
                             "nearest_fill", "learned_fill", "zselect_fill",
                             "maxalpha", "maxalpha_fill", "iblend", "iblend_fill",
                             "learned_iblend_fill", "target_rgbd", "target_rgbd_fill",
                             "target_rgbd_splat", "target_rgbd_splat_fill",
                             "iblend_surf_gate", "iblend_surf_gate_fill",
                             "iblend_tsurf_fill"])
    ap.add_argument("--anchor_blend_topk", type=int, default=2)
    ap.add_argument("--anchor_blend_temp", type=float, default=0.25)
    ap.add_argument("--anchor_iblend_alpha_power", type=float, default=1.0)
    ap.add_argument("--anchor_iblend_view_weight", type=int, default=0)
    ap.add_argument("--anchor_iblend_depth_weight", type=float, default=0.0)
    ap.add_argument("--anchor_iblend_depth_tol_frac", type=float, default=0.03)
    ap.add_argument("--anchor_iblend_color_mode", default="average",
                    choices=["average", "nearest", "maxweight", "maxweight_st"])
    ap.add_argument("--anchor_iblend_alpha_mode", default="average",
                    choices=[
                        "average", "nearest", "nearest_average",
                        "maxweight", "maxweight_st", "maxalpha",
                    ])
    ap.add_argument("--anchor_iblend_support_weight", type=float, default=0.0)
    ap.add_argument("--anchor_iblend_support_refs", type=int, default=2)
    ap.add_argument("--anchor_iblend_support_decay", type=float, default=0.7)
    ap.add_argument("--anchor_iblend_support_floor", type=float, default=0.2)
    ap.add_argument("--anchor_iblend_support_tol_frac", type=float, default=-1.0)
    ap.add_argument("--support_sample_radius_px", type=int, default=0)
    ap.add_argument("--anchor_iblend_agree_weight", type=float, default=0.0)
    ap.add_argument("--anchor_iblend_agree_sigma", type=float, default=0.08)
    ap.add_argument("--anchor_zselect_alpha_min", type=float, default=0.02)
    ap.add_argument("--anchor_fill_alpha_power", type=float, default=1.0)
    ap.add_argument("--anchor_fill_mask_alpha_min", type=float, default=0.0)
    ap.add_argument("--anchor_fill_mask_dilate_px", type=int, default=0)
    ap.add_argument("--anchor_fill_static_alpha_min", type=float, default=0.0)
    ap.add_argument("--anchor_fill_static_alpha_softness", type=float, default=0.0)
    ap.add_argument("--anchor_fill_hull_mask", type=int, default=0)
    ap.add_argument("--anchor_fill_target_surface_mask", type=int, default=0)
    ap.add_argument("--anchor_output_hull_mask", type=int, default=0)
    ap.add_argument("--output_alpha_cleanup_min", type=float, default=0.0)
    ap.add_argument("--output_alpha_cleanup_softness", type=float, default=0.0)
    ap.add_argument("--output_alpha_cleanup_erode_px", type=int, default=0)
    ap.add_argument("--output_alpha_cleanup_dilate_px", type=int, default=0)
    ap.add_argument("--output_alpha_refine_unet", type=int, default=0)
    ap.add_argument("--output_alpha_refine_hidden", type=int, default=16)
    ap.add_argument("--output_alpha_refine_init", type=float, default=0.995)
    ap.add_argument("--output_alpha_refine_floor", type=float, default=0.0)
    ap.add_argument("--output_alpha_refine_delta_scale", type=float, default=10.0)
    ap.add_argument("--output_alpha_refine_prior_weight", type=float, default=0.0)
    ap.add_argument("--output_alpha_refine_tv_weight", type=float, default=0.0)
    ap.add_argument("--target_surface_depth_tol_frac", type=float, default=0.015)
    ap.add_argument("--target_surface_scale_frac", type=float, default=-1.0)
    ap.add_argument("--target_surface_normal_scale_frac", type=float, default=-1.0)
    ap.add_argument("--target_surface_opacity", type=float, default=-1.0)
    ap.add_argument("--target_surface_view_weight_temp_frac", type=float, default=0.0)
    ap.add_argument("--target_surface_min_support", type=int, default=1)
    ap.add_argument("--target_surface_support_tol_frac", type=float, default=-1.0)
    ap.add_argument("--target_surface_gate_power", type=float, default=1.0)
    ap.add_argument("--target_surface_gate_dilate_px", type=int, default=0)
    ap.add_argument("--anchor_learned_fill_hidden", type=int, default=32)
    ap.add_argument("--anchor_learned_fill_layers", type=int, default=3)
    ap.add_argument("--anchor_learned_fill_arch", default="cnn", choices=["cnn", "unet"])
    ap.add_argument("--anchor_learned_fill_rgb_residual_scale", type=float, default=0.0)
    ap.add_argument("--anchor_learned_fill_detach_inputs", type=int, default=1)
    ap.add_argument("--anchor_learned_fill_delta_scale", type=float, default=0.0)
    ap.add_argument("--anchor_learned_fill_candidate_delta_scale", type=float, default=-1.0)
    ap.add_argument("--anchor_learned_fill_prior_weight", type=float, default=0.0)
    ap.add_argument("--anchor_learned_fill_tv_weight", type=float, default=0.0)
    ap.add_argument("--anchor_learned_fill_oracle_weight", type=float, default=0.0)
    ap.add_argument("--anchor_learned_fill_oracle_temp", type=float, default=0.05)
    ap.add_argument("--anchor_learned_fill_oracle_mask_weight", type=float, default=0.25)
    ap.add_argument("--anchor_learned_fill_oracle_alpha_min", type=float, default=0.03)
    ap.add_argument("--depth_refine_unet", type=int, default=0)
    ap.add_argument("--depth_refine_hidden", type=int, default=32)
    ap.add_argument("--depth_refine_delta_scale", type=float, default=0.10)
    ap.add_argument("--depth_refine_detach_inputs", type=int, default=1)
    ap.add_argument("--depth_refine_gt_weight", type=float, default=0.0)
    ap.add_argument("--depth_refine_metric_gt_weight", type=float, default=0.0)
    ap.add_argument("--depth_refine_metric_delta_frac", type=float, default=0.01)
    ap.add_argument("--depth_refine_conflict_weight", type=float, default=0.0)
    ap.add_argument("--depth_refine_gt_outlier_weight", type=float, default=0.0)
    ap.add_argument("--depth_refine_gt_outlier_power", type=float, default=1.0)
    ap.add_argument("--depth_refine_prior_weight", type=float, default=0.0)
    ap.add_argument("--depth_refine_tv_weight", type=float, default=0.0)
    ap.add_argument("--depth_refine_multiview_features", type=int, default=0)
    ap.add_argument("--depth_refine_multiview_refs", type=int, default=4)
    ap.add_argument("--depth_refine_multiview_tol_frac", type=float, default=0.02)
    ap.add_argument("--depth_refine_multiview_radius_px", type=int, default=0)
    ap.add_argument("--depth_refine_chunk_views", type=int, default=0,
                    help="Run the full-resolution depth-refine U-Net in view chunks. 0 disables chunking.")
    ap.add_argument("--depth_refine_checkpoint", type=int, default=0,
                    help="Activation-checkpoint the depth-refine U-Net to reduce VRAM.")
    ap.add_argument("--depth_refine_apply_mv_conflict_min", type=float, default=0.0,
                    help="If >0, apply depth-refine deltas only where multi-view conflict/front-conflict reaches this threshold.")
    ap.add_argument("--depth_refine_apply_mv_support_max", type=float, default=1.0,
                    help="If <1, apply depth-refine deltas only where multi-view support is at most this value.")
    ap.add_argument("--depth_refine_apply_mv_coverage_min", type=float, default=0.0,
                    help="If >0, apply depth-refine deltas only where projected-reference coverage reaches this threshold.")
    ap.add_argument("--depth_refine_apply_erode_px", type=int, default=0)
    ap.add_argument("--freeze_depth_refine_head", type=int, default=0)
    ap.add_argument("--support_gate_unet", type=int, default=0)
    ap.add_argument("--support_gate_hidden", type=int, default=24)
    ap.add_argument("--support_gate_init", type=float, default=0.99)
    ap.add_argument("--support_gate_floor", type=float, default=0.0)
    ap.add_argument("--support_gate_delta_scale", type=float, default=8.0)
    ap.add_argument("--support_gate_detach_inputs", type=int, default=1)
    ap.add_argument("--support_gate_gt_weight", type=float, default=0.0)
    ap.add_argument("--support_gate_prior_weight", type=float, default=0.0)
    ap.add_argument("--support_gate_tv_weight", type=float, default=0.0)
    ap.add_argument("--support_gate_depth_tol_frac", type=float, default=0.02)
    ap.add_argument("--support_gate_multiview_target", type=int, default=0)
    ap.add_argument("--support_gate_multiview_refs", type=int, default=4)
    ap.add_argument("--surface_confidence_unet", type=int, default=0)
    ap.add_argument("--surface_confidence_hidden", type=int, default=24)
    ap.add_argument("--surface_confidence_init", type=float, default=0.995)
    ap.add_argument("--surface_confidence_floor", type=float, default=0.0)
    ap.add_argument("--surface_confidence_opacity_max", type=float, default=0.0)
    ap.add_argument("--surface_confidence_gate_strength", type=float, default=1.0)
    ap.add_argument("--surface_confidence_scale_strength", type=float, default=0.0)
    ap.add_argument("--surface_confidence_scale_floor", type=float, default=0.25)
    ap.add_argument("--surface_confidence_delta_scale", type=float, default=8.0)
    ap.add_argument("--surface_confidence_detach_inputs", type=int, default=1)
    ap.add_argument("--surface_confidence_gt_weight", type=float, default=0.0)
    ap.add_argument("--surface_confidence_prior_weight", type=float, default=0.0)
    ap.add_argument("--surface_confidence_tv_weight", type=float, default=0.0)
    ap.add_argument("--surface_confidence_positive_weight", type=float, default=1.0)
    ap.add_argument("--surface_confidence_negative_weight", type=float, default=1.0)
    ap.add_argument("--surface_confidence_target_pos_min", type=float, default=0.0)
    ap.add_argument("--surface_confidence_target_neg_max", type=float, default=1.0)
    ap.add_argument("--surface_confidence_target_min_pos_support", type=float, default=0.0)
    ap.add_argument("--surface_confidence_target_min_neg_conflicts", type=float, default=0.0)
    ap.add_argument("--surface_confidence_depth_tol_frac", type=float, default=0.02)
    ap.add_argument("--surface_confidence_multiview_refs", type=int, default=4)
    ap.add_argument("--surface_confidence_multiview_tol_frac", type=float, default=0.02)
    ap.add_argument("--surface_confidence_multiview_radius_px", type=int, default=0)
    ap.add_argument("--surface_confidence_score_weight", type=float, default=0.0)
    ap.add_argument("--surface_confidence_protect_support_min", type=float, default=0.0)
    ap.add_argument("--surface_confidence_protect_conflict_max", type=float, default=1.0)
    ap.add_argument("--surface_confidence_protect_coverage_min", type=float, default=0.0)
    ap.add_argument("--surface_refine_unet", type=int, default=0)
    ap.add_argument("--surface_refine_hidden", type=int, default=32)
    ap.add_argument("--surface_refine_init", type=float, default=0.995)
    ap.add_argument("--surface_refine_opacity_floor", type=float, default=0.25)
    ap.add_argument("--surface_refine_opacity_delta_scale", type=float, default=4.0)
    ap.add_argument("--surface_refine_scale_delta_scale", type=float, default=0.25)
    ap.add_argument("--surface_refine_scale_floor", type=float, default=0.50)
    ap.add_argument("--surface_refine_rgb_delta_scale", type=float, default=0.06)
    ap.add_argument("--surface_refine_detach_inputs", type=int, default=1)
    ap.add_argument("--surface_refine_checkpoint", type=int, default=0,
                    help="Activation-checkpoint the full-resolution surface-refine U-Net.")
    ap.add_argument("--surface_refine_prior_weight", type=float, default=0.0)
    ap.add_argument("--surface_refine_tv_weight", type=float, default=0.0)
    ap.add_argument("--surface_refine_rgb_gt_weight", type=float, default=0.0)
    ap.add_argument("--surface_refine_rgb_grad_gt_weight", type=float, default=0.0)
    ap.add_argument("--surface_refine_gt_alpha_min", type=float, default=0.5)
    ap.add_argument("--condition_source", default="target", choices=["target", "fixed"])
    ap.add_argument("--cond_subdir", default=None)                  # e.g. ltx_decoded
    ap.add_argument("--cond_depth_subdir", default=None)            # e.g. depth_anything_ltx
    ap.add_argument("--cond_visibility_depth_subdir", default=None) # optional separate consistency/support depth
    ap.add_argument("--cond_conf_subdir", default=None)             # e.g. da3_ltx
    ap.add_argument("--cond_view_indices", default="")              # available | spread:N | comma-list
    ap.add_argument("--strict_condition_depth", type=int, default=1,
                    help="Fail fast when an explicitly requested conditioning depth sidecar is missing.")
    ap.add_argument("--filter_missing_condition", type=int, default=0,
                    help="Drop objects missing fixed conditioning frames/depth before training/eval.")
    ap.add_argument("--filter_missing_condition_min_views", type=int, default=1)
    ap.add_argument("--condition_mask_source", default="gt",
                    choices=["gt", "rgb_white", "rgb_border"])
    ap.add_argument("--condition_mask_rgb_threshold", type=float, default=0.08)
    ap.add_argument("--condition_mask_rgb_softness", type=float, default=0.02)
    ap.add_argument("--condition_mask_rgb_erode_px", type=int, default=0)
    ap.add_argument("--condition_mask_rgb_dilate_px", type=int, default=0)
    ap.add_argument("--condition_mask_refine_unet", type=int, default=0)
    ap.add_argument("--condition_mask_refine_hidden", type=int, default=32)
    ap.add_argument("--condition_mask_refine_scale", type=float, default=8.0)
    ap.add_argument("--fusion_depth_filter", type=int, default=0)   # prune static concat splats inconsistent with anchor depths
    ap.add_argument("--fusion_filter_all_views", type=int, default=0)
    ap.add_argument("--fusion_filter_nearest_refs", type=int, default=0)
    ap.add_argument("--fusion_filter_mode", default="hard", choices=["hard", "opacity"])
    ap.add_argument("--fusion_depth_bidirectional", type=int, default=0)
    ap.add_argument("--fusion_filter_silhouette_weight", type=float, default=1.0)
    ap.add_argument("--fusion_filter_front_weight", type=float, default=1.0)
    ap.add_argument("--fusion_min_support", type=int, default=0)
    ap.add_argument("--fusion_voxel_size_frac", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_min_count", type=int, default=1)
    ap.add_argument("--fusion_voxel_max_per_cell", type=int, default=1)
    ap.add_argument("--fusion_voxel_mode", default="select", choices=["select", "average"])
    ap.add_argument("--fusion_voxel_color_mode", default="average",
                    choices=["average", "select", "score_select", "score_soft"])
    ap.add_argument("--fusion_voxel_color_select_mix", type=float, default=1.0)
    ap.add_argument("--fusion_voxel_representative", default="opacity", choices=["opacity", "medoid", "score"])
    ap.add_argument("--fusion_voxel_score_softmax_temp", type=float, default=1.0)
    ap.add_argument("--fusion_voxel_score_soft_opacity_mix", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_score_soft_geometry_mix", type=float, default=0.0,
                    help="Blend fused geometry toward score-soft source geometry for render-connected candidate scoring.")
    ap.add_argument("--fusion_voxel_scale_mult", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_scale_floor_z_mult", type=float, default=1.0)
    ap.add_argument("--fusion_voxel_low_support_scale_mult", type=float, default=1.0)
    ap.add_argument("--fusion_voxel_detail_scale_min", type=float, default=1.0)
    ap.add_argument("--fusion_detail_power", type=float, default=1.0)
    ap.add_argument("--fusion_detail_quantile", type=float, default=0.95)
    ap.add_argument("--fusion_voxel_average_dist_decay", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_neighbor_radius", type=int, default=1)
    ap.add_argument("--fusion_voxel_neighbor_min", type=int, default=0)
    ap.add_argument("--fusion_voxel_neighbor_opacity_decay", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_support_propagation_steps", type=int, default=0)
    ap.add_argument("--fusion_voxel_support_propagation_radius", type=int, default=1)
    ap.add_argument("--fusion_voxel_support_propagation_opacity_decay", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_low_support_opacity_decay", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_coverage_opacity_mult", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_coverage_scale_mult", type=float, default=1.0)
    ap.add_argument("--fusion_voxel_pca_quat", type=int, default=0)
    ap.add_argument("--fusion_voxel_score_depth", type=int, default=0)
    ap.add_argument("--fusion_voxel_score_depth_tol_frac", type=float, default=-1.0)
    ap.add_argument("--fusion_voxel_score_exclude_source_view", type=int, default=0)
    ap.add_argument("--fusion_voxel_score_conflict_weight", type=float, default=1.0)
    ap.add_argument("--fusion_voxel_score_color", type=int, default=0)
    ap.add_argument("--fusion_voxel_score_color_weight", type=float, default=1.0)
    ap.add_argument("--fusion_voxel_score_color_tol", type=float, default=0.08)
    ap.add_argument("--fusion_voxel_score_confidence", type=int, default=0)
    ap.add_argument("--fusion_voxel_score_confidence_normalize", type=int, default=1)
    ap.add_argument("--fusion_voxel_score_confidence_floor", type=float, default=0.25)
    ap.add_argument("--fusion_voxel_score_confidence_power", type=float, default=1.0)
    ap.add_argument("--fusion_voxel_score_confidence_supports", type=int, default=1,
                    help="When confidence maps are enabled, use them to weight positive depth support.")
    ap.add_argument("--fusion_voxel_score_confidence_conflicts", type=int, default=1,
                    help="When confidence maps are enabled, use them to downweight depth/front conflicts too.")
    ap.add_argument("--fusion_voxel_score_opacity_norm", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_score_opacity_floor", type=float, default=0.0)
    ap.add_argument("--fusion_voxel_score_opacity_power", type=float, default=1.0)
    # Learned pre-fusion candidate selection. This sees the deterministic
    # support/conflict/color score for every source splat before voxel fusion and
    # emits bounded score + opacity-gate residuals. It is zero-initialized, so
    # enabling it preserves the deterministic prior at step 0.
    ap.add_argument("--fusion_candidate_gate", type=int, default=0)
    ap.add_argument("--fusion_candidate_hidden", type=int, default=0)
    ap.add_argument("--fusion_candidate_layers", type=int, default=2)
    ap.add_argument("--fusion_candidate_coord_features", type=int, default=0,
                    help="Add normalized xyz/radius/source-phase features to the pre-voxel gate.")
    ap.add_argument("--fusion_candidate_rich_features", type=int, default=0,
                    help="Add support ratios and depth/color error features to the pre-voxel gate.")
    ap.add_argument("--fusion_candidate_voxel_features", type=int, default=0,
                    help="Add same-voxel competition features to the pre-voxel gate.")
    ap.add_argument("--fusion_candidate_neighbor_features", type=int, default=0,
                    help="Add occupied-neighborhood support/conflict features to the pre-voxel gate.")
    ap.add_argument("--fusion_candidate_neighbor_radius", type=int, default=1)
    ap.add_argument("--fusion_candidate_detach_inputs", type=int, default=1)
    ap.add_argument("--fusion_candidate_checkpoint", type=int, default=0,
                    help="Activation-checkpoint the per-splat candidate MLP to reduce VRAM.")
    ap.add_argument("--fusion_candidate_chunk_size", type=int, default=0,
                    help="Run the per-splat candidate MLP in chunks. 0 disables chunking.")
    ap.add_argument("--fusion_candidate_score_delta_scale", type=float, default=0.0)
    ap.add_argument("--fusion_candidate_opacity_delta_scale", type=float, default=0.0)
    ap.add_argument("--fusion_candidate_opacity_init", type=float, default=0.95)
    ap.add_argument("--fusion_candidate_opacity_floor", type=float, default=0.25)
    ap.add_argument("--fusion_candidate_prior_weight", type=float, default=0.0)
    ap.add_argument("--fusion_candidate_gt_weight", type=float, default=0.0)
    ap.add_argument("--fusion_candidate_gt_source", default="support",
                    choices=["support", "target_depth"],
                    help="Supervision source for candidate gate: DA3 support counts or GT source depth.")
    ap.add_argument("--fusion_candidate_positive_weight", type=float, default=1.0)
    ap.add_argument("--fusion_candidate_negative_weight", type=float, default=1.0)
    ap.add_argument("--fusion_candidate_target_pos_min", type=float, default=0.75)
    ap.add_argument("--fusion_candidate_target_neg_max", type=float, default=0.25)
    # Learned multi-view fusion via sparse 3D conv on the voxel-fused dict
    # (Option B: replaces the hand-tuned per-voxel residual heads with a 3D-context-aware net).
    ap.add_argument("--use_sparse_voxel_fusion", type=int, default=0)
    ap.add_argument("--use_mlp_voxel_fusion", type=int, default=0,
                    help="Use dense MLP fused-splat residual head instead of spconv sparse head.")
    ap.add_argument("--use_message_voxel_fusion", type=int, default=0,
                    help="Use dependency-free learned occupied-voxel message passing before the residual head.")
    ap.add_argument("--sparse_voxel_hidden", type=int, default=32)
    ap.add_argument("--mlp_voxel_layers", type=int, default=3)
    ap.add_argument("--mlp_voxel_neighbor_radius", type=int, default=0,
                    help="Append occupied-neighborhood aggregate features for dense MLP fallback.")
    ap.add_argument("--mlp_voxel_message_radius", type=int, default=1,
                    help="Occupied-voxel message radius for --use_message_voxel_fusion.")
    ap.add_argument("--sparse_voxel_depth_res_frac", type=float, default=0.05)
    ap.add_argument("--sparse_voxel_rgb_res_scale", type=float, default=0.1)
    ap.add_argument("--sparse_voxel_opacity_res_scale", type=float, default=0.1)
    # Symmetric vis range: vis = 1 + tanh(.)*vis_delta ∈ [1-δ, 1+δ].
    # δ=0.5 → [0.5, 1.5] lets the net both suppress (kill shell) AND boost
    # (rescue under-supported real surface), fixing the deletion bias.
    ap.add_argument("--sparse_voxel_vis_delta", type=float, default=0.5)
    # Identity regularization: penalize ((vis-1)^2).mean().  Pushes the
    # default toward "preserve prior" so the net only moves vis when there's
    # strong photometric gradient.  Critical for avoiding sharpness-via-
    # deletion on a strong-prior + small-head regime.
    ap.add_argument("--sparse_voxel_identity_reg_weight", type=float, default=0.0)
    ap.add_argument("--sparse_voxel_support_reg_weight", type=float, default=0.0,
                    help="Penalize opacity boosts on weak/conflicting fused splats and deletion on strongly supported splats.")
    ap.add_argument("--sparse_voxel_target_vis_weight", type=float, default=0.0,
                    help="Train fused-splat opacity gates from GT target-depth support/conflict labels.")
    ap.add_argument("--sparse_voxel_target_vis_pos_min", type=float, default=0.75)
    ap.add_argument("--sparse_voxel_target_vis_neg_max", type=float, default=0.25)
    ap.add_argument("--sparse_voxel_target_vis_positive_weight", type=float, default=2.0)
    ap.add_argument("--sparse_voxel_target_vis_negative_weight", type=float, default=1.0)
    # Enhance-only mode: vis ∈ [1, 1+2δ], op_res ∈ [0, 2·op_max].  Architecturally
    # impossible to delete splats.  Direct counter to the failure mode that all
    # soft-regularization variants v1.0-v1.5 couldn't fix.
    ap.add_argument("--sparse_voxel_enhance_only", type=int, default=0)
    # Early learned decoder: bypass the ray-grid/voxel-fusion scaffold and emit
    # Gaussians directly from sampled conditioning RGBD surface tokens.
    ap.add_argument("--use_surface_token_decoder", type=int, default=0)
    ap.add_argument("--surface_token_grid_h", type=int, default=48)
    ap.add_argument("--surface_token_grid_w", type=int, default=64)
    ap.add_argument("--surface_token_hidden", type=int, default=256)
    ap.add_argument("--surface_token_slots", type=int, default=256)
    ap.add_argument("--surface_token_layers", type=int, default=3)
    ap.add_argument("--surface_token_heads", type=int, default=8)
    ap.add_argument("--surface_token_latent_layers", type=int, default=0,
                    help="Extra slot cross-attention layers over the full LTX latent grid.")
    ap.add_argument("--surface_token_latent_pool", type=int, default=1,
                    help="Spatial average-pooling factor for dense latent tokens.")
    ap.add_argument("--surface_token_latent_gate_init", type=float, default=0.02,
                    help="Initial residual gate for dense latent-grid slot attention.")
    ap.add_argument("--surface_token_slot_refine_layers", type=int, default=0,
                    help="Extra slot-only refinement layers; adds learned capacity without "
                         "source-token attention memory growth.")
    ap.add_argument("--surface_token_slot_refine_mlp_ratio", type=int, default=4)
    ap.add_argument("--surface_token_slot_refine_gate_init", type=float, default=0.02)
    ap.add_argument("--surface_token_mean_res_frac", type=float, default=0.03)
    ap.add_argument("--surface_token_rgb_res_scale", type=float, default=0.20)
    ap.add_argument("--surface_token_scale_frac", type=float, default=0.003)
    ap.add_argument("--surface_token_normal_scale_frac", type=float, default=0.0004)
    ap.add_argument("--surface_token_scale_res_scale", type=float, default=1.0)
    ap.add_argument("--surface_token_quat_res_scale", type=float, default=0.20)
    ap.add_argument("--surface_token_depth_normal_quat", type=int, default=0,
                    help="Orient surface-token Gaussians from RGBD depth normals instead of camera rays.")
    ap.add_argument("--surface_token_depth_normal_blend", type=float, default=1.0,
                    help="Blend between ray-facing normals (0) and RGBD depth normals (1).")
    ap.add_argument("--surface_token_learned_depth_normal_blend", type=int, default=0,
                    help="Learn the ray/depth-normal blend as a global trainable scalar.")
    ap.add_argument("--surface_token_learned_depth_normal_blend_head", type=int, default=0,
                    help="Learn per-token ray/depth-normal blending from surface-token features.")
    ap.add_argument("--surface_token_depth_normal_blend_head_scale", type=float, default=2.0,
                    help="Logit residual scale for the per-token learned normal-blend head.")
    ap.add_argument("--surface_token_opacity_init", type=float, default=0.80)
    ap.add_argument("--surface_token_checkpoint_blocks", type=int, default=0,
                    help="Use activation checkpointing inside surface-token attention blocks.")
    ap.add_argument("--surface_token_train_rgb_head_only", type=int, default=0,
                    help="Freeze the surface-token backbone and all non-RGB output rows; "
                         "only train final-head rows 3:6 that control RGB residuals.")
    ap.add_argument("--surface_token_train_detail_only", type=int, default=0,
                    help="Freeze the base surface-token decoder and train only the optional detail head.")
    ap.add_argument("--surface_token_train_new_capacity_only", type=int, default=0,
                    help="Freeze the loaded surface-token scaffold and train only newly "
                         "added latent/refinement capacity plus learned scale/opacity/color priors.")
    ap.add_argument("--surface_token_train_policy_heads_only", type=int, default=0,
                    help="Freeze the heavy surface-token trunk and train only learned "
                         "scale/opacity/color/policy/proposal/view-allocation heads.")
    ap.add_argument("--surface_token_detail_layer", type=int, default=0,
                    help="Emit a second learned small-splat detail layer per sampled surface token.")
    ap.add_argument("--surface_token_detail_mean_res_frac", type=float, default=0.012)
    ap.add_argument("--surface_token_detail_rgb_res_scale", type=float, default=0.08)
    ap.add_argument("--surface_token_detail_scale_frac", type=float, default=0.0012)
    ap.add_argument("--surface_token_detail_normal_scale_frac", type=float, default=0.00012)
    ap.add_argument("--surface_token_detail_scale_res_scale", type=float, default=1.0)
    ap.add_argument("--surface_token_detail_quat_res_scale", type=float, default=0.12)
    ap.add_argument("--surface_token_detail_opacity_init", type=float, default=0.04)
    ap.add_argument("--surface_token_source_rgb_dropout_prob", type=float, default=0.0,
                    help="During training, replace whole conditioning-view RGB maps with "
                         "their foreground mean to force latent/color reasoning.")
    ap.add_argument("--surface_token_learned_scale_base", type=int, default=0,
                    help="Make the multiplicative scale base trainable instead of fixed.")
    ap.add_argument("--surface_token_learned_scale_head", type=int, default=0,
                    help="Predict Gaussian scales directly in a broad log range instead "
                         "of using hand-set tangent/normal scale priors.")
    ap.add_argument("--surface_token_learned_scale_min_frac", type=float, default=1e-5)
    ap.add_argument("--surface_token_learned_scale_max_frac", type=float, default=3e-2)
    ap.add_argument("--surface_token_learned_opacity_bias", type=int, default=0,
                    help="Make the opacity logit bias trainable instead of fixed.")
    ap.add_argument("--surface_token_learned_opacity_prior", type=int, default=0,
                    help="Use a latent/slot-conditioned opacity bias instead of a fixed opacity init.")
    ap.add_argument("--surface_token_learned_output_scales", type=int, default=0,
                    help="Learn mean/color/scale/quaternion residual amplitudes instead "
                         "of keeping those output ranges fixed.")
    ap.add_argument("--surface_token_learned_color_affine", type=int, default=0,
                    help="Use a zero-init scene-conditioned RGB gain/bias before the "
                         "per-token color residual.")
    ap.add_argument("--surface_token_color_affine_scale", type=float, default=0.35)
    ap.add_argument("--surface_token_learned_policy_head", type=int, default=0,
                    help="Enable a zero-init learned per-token policy for source depth/confidence, "
                         "view/keep gates, coverage, scale/opacity, and splat moves.")
    ap.add_argument("--surface_token_policy_depth_res_frac", type=float, default=0.0)
    ap.add_argument("--surface_token_policy_move_res_frac", type=float, default=0.0)
    ap.add_argument("--surface_token_policy_scale_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_policy_opacity_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_policy_view_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_policy_confidence_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_policy_keep_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_policy_coverage_scale_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_policy_birth_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_learned_policy_output_scales", type=int, default=0,
                    help="Learn the main policy depth/move/scale/opacity/gate amplitudes "
                         "instead of keeping those ranges fixed.")
    ap.add_argument("--surface_token_learned_source_depth_confidence_head", type=int, default=0,
                    help="Enable a zero-init learned source-depth/confidence head that "
                         "moves RGBD token anchors along rays and gates their opacity.")
    ap.add_argument("--surface_token_source_depth_res_frac", type=float, default=0.0)
    ap.add_argument("--surface_token_source_confidence_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_learned_source_depth_confidence_scales", type=int, default=0,
                    help="Learn source depth/confidence output amplitudes instead of "
                         "keeping those ranges fixed.")
    ap.add_argument("--surface_token_learned_view_selector", type=int, default=0,
                    help="Learn which source views to feed the surface-token decoder. "
                         "Zero-init preserves the current fixed/spread anchor rule.")
    ap.add_argument("--surface_token_view_selector_hidden", type=int, default=128)
    ap.add_argument("--surface_token_view_selector_score_scale", type=float, default=1.0)
    ap.add_argument("--surface_token_view_selector_gate_scale", type=float, default=0.75)
    ap.add_argument("--surface_token_view_selector_train_noise", type=float, default=0.0,
                    help="Optional train-time logit exploration noise for discrete top-k view selection.")
    ap.add_argument("--surface_token_proposal_count", type=int, default=0,
                    help="Emit this many free learned Gaussian proposals in addition to "
                         "the RGBD surface-token scaffold.")
    ap.add_argument("--surface_token_proposal_scale_frac", type=float, default=0.0012)
    ap.add_argument("--surface_token_proposal_normal_scale_frac", type=float, default=0.00012)
    ap.add_argument("--surface_token_proposal_scale_res_scale", type=float, default=1.0)
    ap.add_argument("--surface_token_proposal_quat_res_scale", type=float, default=0.25)
    ap.add_argument("--surface_token_proposal_rgb_res_scale", type=float, default=0.5)
    ap.add_argument("--surface_token_proposal_extent_frac", type=float, default=1.25)
    ap.add_argument("--surface_token_proposal_coverage_scale_res_scale", type=float, default=0.75)
    ap.add_argument("--surface_token_proposal_opacity_init", type=float, default=5e-4)
    ap.add_argument("--surface_token_proposal_seed_surface", type=int, default=0,
                    help="Initialize free proposals by learned attention over pooled RGBD surface tokens.")
    ap.add_argument("--surface_token_proposal_seed_pool", type=int, default=1024)
    ap.add_argument("--surface_token_proposal_surface_res_frac", type=float, default=0.04)
    ap.add_argument("--surface_token_proposal_anchor_mode", type=str, default="even",
                    choices=["even", "learned_st", "learned_local_st", "learned_local_unique_st"],
                    help="How surface-seeded proposals choose RGBD surface anchors. "
                         "'even' preserves the current deterministic pool ordering; "
                         "'learned_st' uses global hard straight-through learned anchor selection; "
                         "'learned_local_st' learns within a local window around each even anchor; "
                         "'learned_local_unique_st' uses learned local scores but greedily avoids "
                         "duplicate hard anchors.")
    ap.add_argument("--surface_token_proposal_anchor_temp", type=float, default=0.25,
                    help="Initial softmax temperature for learned proposal anchor selection.")
    ap.add_argument("--surface_token_proposal_anchor_local_window", type=int, default=9,
                    help="Candidate window size for learned_local_st proposal anchor selection.")
    ap.add_argument("--surface_token_proposal_anchor_gate_init", type=float, default=1e-4,
                    help="Initial learned-anchor mix. Small values preserve even anchors at resume/init.")
    ap.add_argument("--surface_token_proposal_anchor_mix_res_scale", type=float, default=2.0,
                    help="Scale for per-proposal learned anchor-mix residuals.")
    ap.add_argument("--surface_token_proposal_anchor_even_prior", type=float, default=0.0,
                    help="Optional logit prior on the deterministic even anchor, preserving hard scaffold picks at init.")
    ap.add_argument("--surface_token_proposal_anchor_even_prior_final", type=float, default=-1.0,
                    help="If >=0, anneal the learned_st even-anchor prior to this value.")
    ap.add_argument("--surface_token_proposal_anchor_even_prior_decay_steps", type=int, default=0,
                    help="Optimizer steps, relative to the current resume/start step, for even-prior annealing.")
    ap.add_argument("--surface_token_learned_proposal_policy_head", type=int, default=0,
                    help="Enable a zero-init learned policy head for proposal keep/confidence/coverage gates.")
    ap.add_argument("--surface_token_proposal_policy_keep_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_proposal_policy_confidence_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_proposal_policy_coverage_res_scale", type=float, default=0.0)
    ap.add_argument("--surface_token_learned_proposal_scale_base", type=int, default=0,
                    help="Make proposal tangent/normal scale bases trainable instead of fixed.")
    ap.add_argument("--surface_token_learned_proposal_scale_head", type=int, default=0,
                    help="Predict proposal Gaussian scales directly in a broad log range instead "
                         "of using hand-set proposal tangent/normal scale priors.")
    ap.add_argument("--surface_token_learned_proposal_scale_min_frac", type=float, default=1e-5)
    ap.add_argument("--surface_token_learned_proposal_scale_max_frac", type=float, default=3e-2)
    ap.add_argument("--surface_token_scale_reg_weight", type=float, default=0.0,
                    help="Penalize surface-token Gaussian scales above explicit tangent/normal targets.")
    ap.add_argument("--surface_token_tangent_scale_max_frac", type=float, default=0.004,
                    help="Max preferred tangent scale as a fraction of object radius.")
    ap.add_argument("--surface_token_normal_scale_max_frac", type=float, default=0.00035,
                    help="Max preferred normal thickness as a fraction of object radius.")
    ap.add_argument("--surface_token_mean_reg_weight", type=float, default=0.0,
                    help="Penalize large learned mean offsets for the surface-token decoder.")
    ap.add_argument("--surface_token_projective_rgb_weight", type=float, default=0.0,
                    help="Train-only direct RGB loss by projecting emitted surface-token "
                         "Gaussians into target RGBD views.")
    ap.add_argument("--surface_token_projective_depth_weight", type=float, default=0.0,
                    help="Train-only direct depth/geometry loss on emitted surface-token "
                         "Gaussians projected into target RGBD views.")
    ap.add_argument("--surface_token_projective_opacity_weight", type=float, default=0.0,
                    help="Train-only direct positive opacity loss for emitted splats "
                         "that are depth-consistent in target RGBD views.")
    ap.add_argument("--surface_token_projective_max_points", type=int, default=32768)
    ap.add_argument("--surface_token_projective_depth_tol_frac", type=float, default=0.02)
    ap.add_argument("--surface_token_projective_fg_threshold", type=float, default=0.5)
    ap.add_argument("--surface_token_source_confidence_weight", type=float, default=0.0,
                    help="Train source-token confidence gates from multiview RGBD support.")
    ap.add_argument("--surface_token_source_depth_weight", type=float, default=0.0,
                    help="Train source-token depth residuals toward nearest supported RGBD surface along the source ray.")
    ap.add_argument("--surface_token_source_policy_points", type=int, default=8192,
                    help="Max source tokens sampled for source depth/confidence policy losses.")
    ap.add_argument("--surface_token_source_policy_target_points", type=int, default=4096,
                    help="Max RGBD target surface points sampled for source policy targets.")
    ap.add_argument("--surface_token_source_policy_depth_tol_frac", type=float, default=0.02)
    ap.add_argument("--surface_token_source_policy_fg_threshold", type=float, default=0.5)
    ap.add_argument("--surface_token_source_policy_positive_weight", type=float, default=2.0)
    ap.add_argument("--surface_token_source_policy_negative_weight", type=float, default=0.5)
    ap.add_argument("--surface_token_source_policy_confidence_target_scale", type=float, default=-1.0,
                    help="Source confidence target amplitude. <0 uses --surface_token_source_confidence_res_scale.")
    ap.add_argument("--surface_token_source_policy_support_mode", type=str, default="nearest",
                    choices=["nearest", "projective"],
                    help="How to build source confidence targets: nearest 3D surface or target-view projective support/conflict.")
    ap.add_argument("--surface_token_source_policy_target_mode", type=str, default="support",
                    choices=["support", "identity", "none"],
                    help="Direct source-policy target mode for confidence/depth heads.")
    ap.add_argument("--surface_token_proposal_cover_weight", type=float, default=0.0,
                    help="Train-only target-surface to proposal coverage loss for "
                         "free surface-token proposals.")
    ap.add_argument("--surface_token_proposal_surface_weight", type=float, default=0.0,
                    help="Train-only proposal to observed-surface distance loss.")
    ap.add_argument("--surface_token_proposal_opacity_weight", type=float, default=0.0,
                    help="Train-only opacity target for proposals near observed RGBD surface.")
    ap.add_argument("--surface_token_proposal_rgb_weight", type=float, default=0.0,
                    help="Train-only color target for proposals near observed RGBD surface.")
    ap.add_argument("--surface_token_proposal_detail_weight", type=float, default=0.0,
                    help="Train-only coverage loss for target RGBD points with high local RGB detail.")
    ap.add_argument("--surface_token_proposal_detail_edge_thresh", type=float, default=0.025,
                    help="RGB gradient threshold used to weight proposal detail coverage.")
    ap.add_argument("--surface_token_proposal_cover_points", type=int, default=2048)
    ap.add_argument("--surface_token_proposal_depth_tol_frac", type=float, default=0.02)
    ap.add_argument("--surface_token_proposal_fg_threshold", type=float, default=0.5)
    ap.add_argument("--surface_token_proposal_opacity_positive_weight", type=float, default=2.0)
    ap.add_argument("--surface_token_proposal_opacity_negative_weight", type=float, default=0.25)
    ap.add_argument("--surface_token_proposal_anchor_entropy_weight", type=float, default=0.0,
                    help="Minimize proposal anchor assignment entropy so learned allocation becomes decisive.")
    ap.add_argument("--surface_token_proposal_anchor_usage_weight", type=float, default=0.0,
                    help="KL-to-uniform penalty on mean anchor usage to prevent learned allocation collapse.")
    ap.add_argument("--surface_token_proposal_anchor_collision_weight", type=float, default=0.0,
                    help="Penalize expected duplicate learned-anchor assignments across proposals.")
    ap.add_argument("--surface_token_proposal_policy_keep_weight", type=float, default=0.0,
                    help="Train proposal keep gate from RGBD nearest-surface support.")
    ap.add_argument("--surface_token_proposal_policy_confidence_weight", type=float, default=0.0,
                    help="Train proposal confidence gate from RGBD nearest-surface support.")
    ap.add_argument("--surface_token_proposal_policy_coverage_weight", type=float, default=0.0,
                    help="Train proposal coverage scale from inverse RGBD nearest-surface support.")
    ap.add_argument("--surface_token_proposal_policy_target_mode", type=str, default="support",
                    choices=["support", "identity", "none"],
                    help=("Direct proposal-policy target mode. 'support' uses the old RGBD-support "
                          "formula, 'identity' only regularizes gates toward scaffold-preserving "
                          "ones, and 'none' disables direct gate targets so render/proposal losses "
                          "train the policy."))
    ap.add_argument("--surface_token_scaffold_rgb_weight", type=float, default=0.0,
                    help="Train-only trust-region loss to keep surface-token renders "
                         "near the new-capacity-disabled scaffold where the learned "
                         "render is not better than the scaffold.")
    ap.add_argument("--surface_token_scaffold_alpha_weight", type=float, default=0.0)
    ap.add_argument("--surface_token_scaffold_detail_weight", type=float, default=0.0,
                    help="Train-only foreground RGB-gradient preservation loss from "
                         "the new-capacity-disabled surface-token scaffold.")
    ap.add_argument("--surface_token_scaffold_detail_alpha_min", type=float, default=0.25,
                    help="Only apply surface-token scaffold-detail loss where the "
                         "scaffold alpha is confidently foreground.")
    ap.add_argument("--surface_token_scaffold_margin", type=float, default=0.0,
                    help="Per-pixel target-error margin before scaffold preservation activates.")
    # More learned decoder: consolidate RGBD observations into canonical
    # occupied voxels, then use learned 3D message passing and latent-grid
    # cross-attention to emit Gaussians.
    ap.add_argument("--use_canonical_voxel_decoder", type=int, default=0)
    ap.add_argument("--canonical_voxel_grid_h", type=int, default=72)
    ap.add_argument("--canonical_voxel_grid_w", type=int, default=108)
    ap.add_argument("--canonical_voxel_hidden", type=int, default=384)
    ap.add_argument("--canonical_voxel_layers", type=int, default=5)
    ap.add_argument("--canonical_voxel_heads", type=int, default=8)
    ap.add_argument("--canonical_voxel_latent_layers", type=int, default=0,
                    help="Self-attention layers over LTX latent tokens before voxel cross-attention.")
    ap.add_argument("--canonical_voxel_scene_slots", type=int, default=0,
                    help="Learned global scene-memory slots used by canonical voxel blocks.")
    ap.add_argument("--canonical_voxel_latent_pool", type=int, default=2)
    ap.add_argument("--canonical_voxel_message_radius", type=int, default=1)
    ap.add_argument("--canonical_voxel_size_frac", type=float, default=0.003)
    ap.add_argument("--canonical_voxel_max_voxels", type=int, default=60000)
    ap.add_argument("--canonical_voxel_gaussians_per_voxel", type=int, default=1)
    ap.add_argument("--canonical_voxel_child_offset_mult", type=float, default=0.35)
    ap.add_argument("--canonical_voxel_mean_res_voxels", type=float, default=0.75)
    ap.add_argument("--canonical_voxel_rgb_res_scale", type=float, default=0.25)
    ap.add_argument("--canonical_voxel_tangent_scale_mult", type=float, default=0.45)
    ap.add_argument("--canonical_voxel_normal_scale_mult", type=float, default=0.10)
    ap.add_argument("--canonical_voxel_scale_res_scale", type=float, default=0.75)
    ap.add_argument("--canonical_voxel_quat_res_scale", type=float, default=0.20)
    ap.add_argument("--canonical_voxel_opacity_init", type=float, default=0.82)
    ap.add_argument("--canonical_voxel_opacity_support_floor", type=float, default=0.35)
    ap.add_argument("--canonical_voxel_opacity_support_target", type=float, default=2.0)
    ap.add_argument("--canonical_voxel_detail_sampling", type=int, default=0,
                    help="Project canonical voxels back into conditioning RGBD views and learn a high-res detail attention summary.")
    ap.add_argument("--canonical_voxel_detail_color_mix", type=float, default=0.75)
    ap.add_argument("--canonical_voxel_detail_depth_tol_frac", type=float, default=0.015)
    ap.add_argument("--canonical_voxel_detail_score_temp", type=float, default=0.75)
    ap.add_argument("--canonical_voxel_detail_chunk", type=int, default=16384)
    ap.add_argument("--canonical_voxel_view_feature_channels", type=int, default=0,
                    help="Per-view CNN feature channels sampled by canonical detail attention.")
    ap.add_argument("--canonical_voxel_view_feature_scale", type=float, default=0.5,
                    help="Resolution multiplier for canonical per-view CNN features.")
    ap.add_argument("--canonical_voxel_opacity_prior_weight", type=float, default=1.0,
                    help="How strongly support-count opacity prior biases learned opacity logits.")
    ap.add_argument("--canonical_voxel_zero_init_head", type=int, default=1,
                    help="Zero-init final canonical Gaussian head to preserve RGBD prior at step 0.")
    ap.add_argument("--canonical_target_vis_weight", type=float, default=0.0,
                    help="Train canonical Gaussian opacity from GT-depth support/conflict counts.")
    ap.add_argument("--canonical_target_vis_pos_min", type=float, default=0.70)
    ap.add_argument("--canonical_target_vis_neg_max", type=float, default=0.30)
    ap.add_argument("--canonical_target_vis_positive_weight", type=float, default=2.0)
    ap.add_argument("--canonical_target_vis_negative_weight", type=float, default=1.0)
    ap.add_argument("--canonical_scale_reg_weight", type=float, default=0.0,
                    help="Penalize canonical Gaussian scales above tangent/normal ceilings.")
    ap.add_argument("--canonical_scale_reg_start", type=int, default=0)
    ap.add_argument("--canonical_scale_reg_ramp", type=int, default=0)
    ap.add_argument("--canonical_tangent_scale_max_frac", type=float, default=0.0015)
    ap.add_argument("--canonical_normal_scale_max_frac", type=float, default=0.00020)
    ap.add_argument("--canonical_source_vis_gate", type=int, default=0,
                    help="Opacity-gate canonical Gaussians by source-view depth/mask consistency.")
    ap.add_argument("--canonical_source_vis_min_support", type=float, default=2.0)
    ap.add_argument("--canonical_source_vis_conflict_weight", type=float, default=1.0)
    ap.add_argument("--canonical_source_vis_softness", type=float, default=0.5)
    ap.add_argument("--canonical_source_vis_floor", type=float, default=0.05)
    ap.add_argument("--canonical_source_vis_learned_refine", type=int, default=0,
                    help="Use a trainable canonical-decoder refiner over source-view support/conflict features.")
    ap.add_argument("--canonical_source_vis_refine_hidden", type=int, default=128)
    ap.add_argument("--canonical_source_vis_refine_opacity_strength", type=float, default=1.0)
    ap.add_argument("--canonical_source_vis_refine_rgb_scale", type=float, default=0.10)
    ap.add_argument("--canonical_source_vis_refine_scale_res_scale", type=float, default=0.25)
    ap.add_argument("--canonical_source_vis_refine_zero_init", type=int, default=1)
    ap.add_argument("--canonical_source_vis_distill_weight", type=float, default=0.0,
                    help="Train canonical opacity to imitate the soft source-view support/conflict gate.")
    ap.add_argument("--view_opacity_gate", type=int, default=0)
    ap.add_argument("--view_opacity_floor", type=float, default=0.0)
    ap.add_argument("--view_opacity_power", type=float, default=1.0)
    ap.add_argument("--view_opacity_flip", type=int, default=0)
    ap.add_argument("--fusion_sh_degree", type=int, default=0)
    ap.add_argument("--fusion_sh_depth_tol_frac", type=float, default=0.02)
    ap.add_argument("--fusion_sh_ridge", type=float, default=1e-3)
    ap.add_argument("--fusion_sh_min_obs", type=int, default=2)
    ap.add_argument("--fusion_sh_mix", type=float, default=1.0)
    ap.add_argument("--tsdf_voxel_size_frac", type=float, default=0.004)
    ap.add_argument("--tsdf_trunc_mult", type=float, default=4.0)
    ap.add_argument("--tsdf_min_weight", type=float, default=2.0)
    ap.add_argument("--tsdf_surface_thresh", type=float, default=0.75)
    ap.add_argument("--tsdf_max_voxels", type=int, default=1500000)
    ap.add_argument("--tsdf_max_points", type=int, default=250000)
    ap.add_argument("--tsdf_scale_mult", type=float, default=0.7)
    ap.add_argument("--tsdf_normal_scale_mult", type=float, default=0.25)
    ap.add_argument("--tsdf_opacity", type=float, default=0.95)
    ap.add_argument("--tsdf_color_mode", default="select", choices=["select", "average"])
    ap.add_argument("--tsdf_surface_mode", default="centers", choices=["centers", "edges"])
    ap.add_argument("--fusion_conflict_opacity_decay", type=float, default=0.7)
    ap.add_argument("--fusion_depth_tol_frac", type=float, default=0.02)
    ap.add_argument("--fusion_bg_margin_px", type=int, default=2)
    ap.add_argument("--fusion_tsdf_filter", type=int, default=0)
    ap.add_argument("--fusion_tsdf_band", type=float, default=0.5)
    ap.add_argument("--fusion_tsdf_opacity_decay", type=float, default=4.0)
    ap.add_argument("--fusion_tsdf_invalid_opacity_mult", type=float, default=0.15)
    ap.add_argument("--oracle_anchor_depth", type=int, default=0)   # ablation: place anchor splats on GT depth maps
    ap.add_argument("--latent_t", type=int, default=0)               # 0 = infer from latent.npy
    ap.add_argument("--latent_h", type=int, default=0)
    ap.add_argument("--latent_w", type=int, default=0)
    ap.add_argument("--ups_stages", type=int, default=4)             # 98k Gaussians
    ap.add_argument("--scale_cap_frac", type=float, default=0.05)    # 98k coverage-stable
    ap.add_argument("--perceptual_weight", type=float, default=0.5)
    ap.add_argument("--perceptual_start", type=int, default=0)
    ap.add_argument("--perceptual_ramp", type=int, default=0)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"],
                    help="Adafactor is useful for very large decoder pilots on limited VRAM.")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--surface_token_proposal_lr_mult", type=float, default=1.0,
                    help="LR multiplier for surface-token proposal parameters.")
    ap.add_argument("--surface_token_proposal_policy_lr_mult", type=float, default=1.0,
                    help="LR multiplier for the learned proposal policy head.")
    ap.add_argument("--surface_token_policy_lr_mult", type=float, default=1.0,
                    help="LR multiplier for the learned main surface-token policy head.")
    ap.add_argument("--surface_token_proposal_opacity_lr_mult", type=float, default=1.0,
                    help="LR multiplier for the proposal opacity logit bias.")
    ap.add_argument("--surface_token_depth_normal_blend_lr_mult", type=float, default=1.0,
                    help="LR multiplier for the learned depth-normal blend scalar.")
    ap.add_argument("--warmup", type=int, default=100)        # LR warmup, in optimizer steps
    ap.add_argument("--fg_weight", type=float, default=10.0)
    ap.add_argument("--mask_weight", type=float, default=0.5)
    ap.add_argument("--fg_alpha_weight", type=float, default=0.0,
                    help="Extra foreground-normalized penalty for alpha below the GT mask.")
    # Photometric loss type: l1 (default; has deletion bias at silhouette edges)
    # or l2/MSE (aligns directly with PSNR since PSNR ≡ -10·log10(MSE)).
    ap.add_argument("--photometric_loss_type", default="l1", choices=["l1", "l2"])
    # GREAT-QUALITY mode: substitute GT depth (cond_target_depths) for the
    # DA3-predicted depth at the conditioning views.  Achieves the documented
    # ~20 dB "oracle" ceiling on v7.  Note: requires GT depth available at the
    # 9 conditioning views at inference (a deployment constraint).
    ap.add_argument("--cond_use_target_depth", type=int, default=0)
    ap.add_argument("--bg_alpha_weight", type=float, default=0.0)    # extra anti-floater alpha penalty outside GT mask
    ap.add_argument("--anchor_rgb_weight", type=float, default=0.0)  # direct ref-view color on FG grid cells
    ap.add_argument("--anchor_rgb_target", default="condition", choices=["condition", "gt"],
                    help="Direct anchor RGB target: conditioning RGB or GT RGB at the same views.")
    ap.add_argument("--anchor_opacity_weight", type=float, default=0.0)  # direct ref-mask occupancy on Gaussian grid
    ap.add_argument("--anchor_scale_weight", type=float, default=0.0)    # shrink FG Gaussian footprint directly
    ap.add_argument("--anchor_visibility_weight", type=float, default=0.0) # train-only depth/mask visibility target
    ap.add_argument("--anchor_scale_frac", type=float, default=0.004)    # target scale = frac * radius
    ap.add_argument("--fg_color_weight", type=float, default=0.0)    # alpha-normalized foreground render color
    ap.add_argument("--fg_color_alpha_min", type=float, default=0.03)
    ap.add_argument("--adaptive_loss_weights", type=int, default=0,
                    help="Use learned uncertainty weights for selected render losses.")
    ap.add_argument("--adaptive_loss_names", default=(
        "photo,ssim,mask,hinge,fg_alpha,bg_alpha,fg_color,grad,alpha_grad,depth"
    ))
    ap.add_argument("--adaptive_loss_logvar_min", type=float, default=-4.0)
    ap.add_argument("--adaptive_loss_logvar_max", type=float, default=4.0)
    ap.add_argument("--grad_weight", type=float, default=0.0)        # delayed foreground image-gradient loss
    ap.add_argument("--grad_start", type=int, default=0)
    ap.add_argument("--grad_ramp", type=int, default=0)
    ap.add_argument("--scaffold_detail_weight", type=float, default=0.0,
                    help="Preserve high-frequency foreground detail from the deterministic scaffold render.")
    ap.add_argument("--scaffold_detail_start", type=int, default=0)
    ap.add_argument("--scaffold_detail_ramp", type=int, default=0)
    ap.add_argument("--scaffold_detail_alpha_min", type=float, default=0.25,
                    help="Only apply scaffold-detail loss where the scaffold is confidently foreground.")
    ap.add_argument("--detail_teacher_weight", type=float, default=0.0,
                    help="Targeted high-frequency teacher loss on real foreground image detail.")
    ap.add_argument("--detail_teacher_start", type=int, default=0)
    ap.add_argument("--detail_teacher_ramp", type=int, default=0)
    ap.add_argument("--detail_teacher_alpha_min", type=float, default=0.35,
                    help="Only apply detail-teacher gradients where detached render alpha is confident.")
    ap.add_argument("--detail_teacher_edge_thresh", type=float, default=0.025,
                    help="Target RGB-gradient threshold that defines real image detail.")
    ap.add_argument("--detail_teacher_artifact_weight", type=float, default=0.10,
                    help="Penalty for render gradients in target-smooth foreground regions.")
    ap.add_argument("--alpha_grad_weight", type=float, default=0.0,
                    help="Match rendered alpha gradients to GT silhouette gradients near object edges.")
    ap.add_argument("--alpha_grad_start", type=int, default=0)
    ap.add_argument("--alpha_grad_ramp", type=int, default=0)
    ap.add_argument("--alpha_grad_band_px", type=int, default=2,
                    help="Dilate GT silhouette edges by this many pixels before applying alpha-gradient loss.")
    ap.add_argument("--alpha_interior_smooth_weight", type=float, default=0.0,
                    help="Penalize alpha lattice gradients inside the GT foreground, away from silhouette edges.")
    ap.add_argument("--alpha_interior_smooth_start", type=int, default=0)
    ap.add_argument("--alpha_interior_smooth_ramp", type=int, default=0)
    ap.add_argument("--alpha_interior_smooth_edge_band_px", type=int, default=4,
                    help="Exclude this many pixels around GT silhouette edges from alpha-interior smoothness.")
    ap.add_argument("--alpha_anti_lattice_weight", type=float, default=0.0,
                    help="Penalize high-frequency alpha dots inside target-smooth foreground regions.")
    ap.add_argument("--alpha_anti_lattice_start", type=int, default=0)
    ap.add_argument("--alpha_anti_lattice_ramp", type=int, default=0)
    ap.add_argument("--alpha_anti_lattice_blur_px", type=int, default=2,
                    help="Blur radius used to estimate alpha high-pass lattice residuals.")
    ap.add_argument("--alpha_anti_lattice_edge_band_px", type=int, default=4,
                    help="Exclude this many pixels around silhouette and target-detail edges.")
    ap.add_argument("--alpha_anti_lattice_detail_edge_thresh", type=float, default=0.025,
                    help="RGB-gradient threshold above which target detail is protected.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train_objects", type=int, default=0)      # 0 = full train split
    ap.add_argument("--n_train_eval", type=int, default=8)           # fixed train objs for the gap
    ap.add_argument("--n_heldout_eval", type=int, default=0)         # 0 = all held-out objs
    ap.add_argument("--eval_views_per_object", type=int, default=0)  # 0 = all target views
    ap.add_argument("--eval_render_chunk", type=int, default=4)
    ap.add_argument("--eval_at_step0", type=int, default=1)
    ap.add_argument("--eval_before_train", type=int, default=0,
                    help="Run an eval at step -1 before any optimizer update.")
    ap.add_argument("--eval_only", type=int, default=0)
    ap.add_argument("--train_precision", default="fp32", choices=["fp32", "bf16"],
                    help="Training compute precision. bf16 autocasts the learned "
                         "forward path and casts emitted Gaussians back to fp32 "
                         "before rendering/losses.")
    ap.add_argument("--save_eval_viz", type=int, default=1)
    ap.add_argument("--save_eval_viz_views", type=int, default=1)
    ap.add_argument("--save_eval_viz_heldout_count", type=int, default=1)
    ap.add_argument("--save_eval_viz_train_count", type=int, default=1)
    ap.add_argument("--save_eval_viz_novel_elevations", default="")
    ap.add_argument("--save_eval_viz_novel_azimuths", default="")
    ap.add_argument("--save_eval_viz_novel_radius_scale", type=float, default=1.0)
    ap.add_argument("--save_eval_viz_novel_azimuth_offset", type=float, default=0.0)
    ap.add_argument("--save_eval_jsonl", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=0)             # 0 = only eval/final checkpoints
    ap.add_argument("--save_named_checkpoints", type=int, default=0)
    ap.add_argument("--save_before_eval", type=int, default=0,
                    help="Write phase2.pt before eval so eval interruptions do not lose training.")
    ap.add_argument("--save_checkpoints", type=int, default=1,
                    help="Set to 0 for metric-only probes that should not write large checkpoints.")
    ap.add_argument("--resume_from", default="")
    ap.add_argument("--reset_optimizer_on_resume", type=int, default=0)
    ap.add_argument("--resume_aux_from", default="")                 # comma-list; load auxiliary heads only
    ap.add_argument("--render_eps2d", type=float, default=0.3)
    ap.add_argument("--rasterize_mode", default="classic", choices=["classic", "antialiased"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_entity", default="rzhan269-brown-university")
    ap.add_argument("--out_dir", default="runs/phase2_98k")
    ap.add_argument("--depth_weight", type=float, default=0.0)        # StableGS ref-view depth
    ap.add_argument("--depth_abs_weight", type=float, default=0.05)   # paired metric-depth Huber
    ap.add_argument("--depth_render_mode", default="ED", choices=["D", "ED", "Ed"])
    ap.add_argument("--depth_render_scale", type=float, default=1.0)  # <1 lowers depth raster cost
    ap.add_argument("--anchor_depth_weight", type=float, default=0.0) # direct per-Gaussian ray depth
    ap.add_argument("--depth_anneal_start", type=int, default=-1)     # <0 disables depth annealing
    ap.add_argument("--depth_anneal_end", type=int, default=-1)
    ap.add_argument("--depth_anneal_final_mult", type=float, default=1.0)
    ap.add_argument("--anchor_depth_use_target", type=int, default=0)
    ap.add_argument("--opacity_reg", type=float, default=0.0)         # 3DGS-MCMC L1 opacity
    ap.add_argument("--scale_reg", type=float, default=0.0)           # 3DGS-MCMC L1 scale
    ap.add_argument("--opacity_reg_masked", type=int, default=1)      # 1=penalize BG-anchored only
    ap.add_argument("--opacity_entropy", type=float, default=0.0)     # binarization prior weight
    ap.add_argument("--mean_offset_frac", type=float, default=0.0)    # bounded free-mean offset
    ap.add_argument("--upsample_mode", default="deconv", choices=["deconv", "resize"])
    ap.add_argument("--latent_skip", type=int, default=0)
    ap.add_argument("--coord_inject", type=int, default=0)
    ap.add_argument("--coord_fourier", type=int, default=4)
    ap.add_argument("--image_condition", type=int, default=0)       # ref RGB*mask + mask high-res skip
    ap.add_argument("--image_head_skip", type=int, default=0)       # direct RGB/opacity output prior from ref image
    ap.add_argument("--condition_unsharp_amount", type=float, default=0.0) # sharpen decoded/source RGB before color skip
    ap.add_argument("--condition_unsharp_kernel", type=int, default=5)
    ap.add_argument("--condition_contrast", type=float, default=1.0)
    ap.add_argument("--condition_saturation", type=float, default=1.0)
    ap.add_argument("--condition_color_calibration", default="none",
                    choices=["none", "train_affine"])
    ap.add_argument("--condition_color_calib_max_objects", type=int, default=0)
    ap.add_argument("--condition_color_calib_views", type=int, default=9)
    ap.add_argument("--condition_color_calib_sample_px", type=int, default=20000)
    ap.add_argument("--condition_color_calib_ridge", type=float, default=1e-4)
    ap.add_argument("--condition_depth_calibration", default="none",
                    choices=["none", "train_affine_frac"])
    ap.add_argument("--condition_depth_calib_max_objects", type=int, default=64)
    ap.add_argument("--condition_depth_calib_views", type=int, default=9)
    ap.add_argument("--condition_depth_calib_sample_px", type=int, default=20000)
    ap.add_argument("--condition_depth_calib_ridge", type=float, default=1e-4)
    ap.add_argument("--condition_depth_median_radius_px", type=int, default=0)
    ap.add_argument("--condition_depth_median_thresh_frac", type=float, default=0.0)
    ap.add_argument("--condition_depth_median_mix", type=float, default=1.0)
    ap.add_argument("--condition_rgb_refine_unet", type=int, default=0)
    ap.add_argument("--condition_rgb_refine_hidden", type=int, default=32)
    ap.add_argument("--condition_rgb_refine_scale", type=float, default=0.15)
    ap.add_argument("--condition_rgb_refine_gt_weight", type=float, default=0.0)
    ap.add_argument("--condition_rgb_refine_gt_alpha_min", type=float, default=0.5)
    # Joint conditioning refiner: LTX/RGBD conditioning -> cleaner RGB + depth
    # before 3DGS fusion. This is the deployable replacement for GT-depth
    # oracle mode: same shared weights for every object, no per-object opt.
    ap.add_argument("--condition_rgbd_refine_unet", type=int, default=0)
    ap.add_argument("--condition_rgbd_refine_arch", default="unet",
                    choices=["unet", "view"])
    ap.add_argument("--condition_rgbd_refine_hidden", type=int, default=32)
    ap.add_argument("--condition_rgbd_refine_context_layers", type=int, default=2)
    ap.add_argument("--condition_rgbd_refine_context_heads", type=int, default=4)
    ap.add_argument("--condition_rgbd_refine_max_views", type=int, default=64)
    ap.add_argument("--condition_rgbd_refine_multiview_features", type=int, default=0)
    ap.add_argument("--condition_rgbd_refine_multiview_refs", type=int, default=4)
    ap.add_argument("--condition_rgbd_refine_multiview_tol_frac", type=float, default=0.02)
    ap.add_argument("--condition_rgbd_refine_multiview_radius_px", type=int, default=0)
    ap.add_argument("--condition_rgbd_refine_detach_inputs", type=int, default=0)
    ap.add_argument("--condition_rgbd_refine_rgb_scale", type=float, default=0.15)
    ap.add_argument("--condition_rgbd_refine_depth_scale", type=float, default=0.7)
    ap.add_argument("--condition_rgbd_refine_apply_erode_px", type=int, default=0)
    ap.add_argument("--condition_rgbd_refine_prior_weight", type=float, default=0.0)
    ap.add_argument("--condition_rgbd_refine_tv_weight", type=float, default=0.0)
    ap.add_argument("--condition_rgbd_refine_rgb_gt_weight", type=float, default=0.0)
    ap.add_argument("--condition_rgbd_refine_depth_gt_weight", type=float, default=0.0)
    ap.add_argument("--condition_rgbd_refine_gt_alpha_min", type=float, default=0.5)
    # Pose auxiliary head: learn to recover the conditioning orbit cameras from
    # RGBD features. This is the bridge to generated LTX orbits with no poses.
    ap.add_argument("--condition_pose_head", type=int, default=0)
    ap.add_argument("--condition_pose_hidden", type=int, default=64)
    ap.add_argument("--condition_pose_context_layers", type=int, default=2)
    ap.add_argument("--condition_pose_context_heads", type=int, default=4)
    ap.add_argument("--condition_pose_max_views", type=int, default=64)
    ap.add_argument("--condition_pose_depth_norm", default="ray_frac",
                    choices=["ray_frac", "local"],
                    help="Depth feature for pose head. local avoids using known camera bounds.")
    ap.add_argument("--condition_pose_detach_inputs", type=int, default=1)
    ap.add_argument("--condition_pose_use_predicted", type=int, default=0,
                    help="Replace conditioning c2w/w2c with pose-head predictions.")
    ap.add_argument("--freeze_condition_pose_head", type=int, default=0)
    ap.add_argument("--condition_pose_weight", type=float, default=0.0)
    ap.add_argument("--condition_pose_center_weight", type=float, default=1.0)
    ap.add_argument("--condition_pose_forward_weight", type=float, default=1.0)
    ap.add_argument("--condition_pose_dist_weight", type=float, default=0.25)
    # Per-view global depth affine correction. This captures systematic DA3
    # scale/shift errors without learning a dense deletion/confidence map.
    ap.add_argument("--condition_depth_affine_head", type=int, default=0)
    ap.add_argument("--condition_depth_affine_hidden", type=int, default=64)
    ap.add_argument("--condition_depth_affine_layers", type=int, default=3)
    ap.add_argument("--condition_depth_affine_scale_range", type=float, default=0.15)
    ap.add_argument("--condition_depth_affine_shift_range", type=float, default=0.03)
    ap.add_argument("--condition_depth_affine_gt_weight", type=float, default=0.0)
    ap.add_argument("--condition_depth_affine_prior_weight", type=float, default=0.0)
    ap.add_argument("--condition_depth_affine_detach_inputs", type=int, default=1)
    # Learned depth-confidence maps for estimated conditioning depth. Unlike
    # RGBD refinement, this does not move geometry; it only downweights unreliable
    # depth evidence before voxel fusion.
    ap.add_argument("--condition_depth_confidence_unet", type=int, default=0)
    ap.add_argument("--condition_depth_confidence_hidden", type=int, default=24)
    ap.add_argument("--condition_depth_confidence_chunk_views", type=int, default=0,
                    help="Apply the full-res depth-confidence U-Net in view chunks. 0 disables chunking.")
    ap.add_argument("--condition_depth_confidence_multiview_features", type=int, default=0)
    ap.add_argument("--condition_depth_confidence_multiview_refs", type=int, default=4)
    ap.add_argument("--condition_depth_confidence_multiview_tol_frac", type=float, default=0.02)
    ap.add_argument("--condition_depth_confidence_multiview_radius_px", type=int, default=0)
    ap.add_argument("--condition_depth_confidence_detach_inputs", type=int, default=1)
    ap.add_argument("--condition_depth_confidence_init", type=float, default=0.95)
    ap.add_argument("--condition_depth_confidence_floor", type=float, default=0.25)
    ap.add_argument("--condition_depth_confidence_delta_scale", type=float, default=6.0)
    ap.add_argument("--condition_depth_confidence_gt_weight", type=float, default=0.0)
    ap.add_argument("--condition_depth_confidence_prior_weight", type=float, default=0.0)
    ap.add_argument("--condition_depth_confidence_tv_weight", type=float, default=0.0)
    ap.add_argument("--condition_depth_confidence_tol_frac", type=float, default=0.02)
    ap.add_argument("--condition_depth_confidence_neg_tol_frac", type=float, default=-1.0,
                    help="Depth error fraction for negative confidence labels; <=0 uses tol_frac.")
    ap.add_argument("--condition_depth_confidence_positive_weight", type=float, default=1.0)
    ap.add_argument("--condition_depth_confidence_negative_weight", type=float, default=3.0)
    ap.add_argument("--condition_mask_erode_px", type=int, default=0)
    ap.add_argument("--condition_mask_blur_px", type=int, default=0)
    ap.add_argument("--condition_rgb_inpaint_px", type=int, default=0)
    ap.add_argument("--condition_confidence_power", type=float, default=0.0)
    ap.add_argument("--condition_confidence_floor", type=float, default=0.15)
    ap.add_argument("--condition_confidence_normalize", type=int, default=1)
    ap.add_argument("--condition_confidence_as_mask", type=int, default=0)
    ap.add_argument("--image_depth_condition", type=int, default=0) # append normalized ray-depth + validity
    ap.add_argument("--image_visual_hull_depth", type=int, default=0) # derive depth from source masks/cameras
    ap.add_argument("--visual_hull_scale", type=float, default=0.25)
    ap.add_argument("--visual_hull_samples", type=int, default=64)
    ap.add_argument("--visual_hull_min_view_frac", type=float, default=0.75)
    ap.add_argument("--visual_hull_mask_margin", type=int, default=1)
    ap.add_argument("--image_voxel_hull_depth", type=int, default=0) # smoother global visual-hull raycast
    ap.add_argument("--voxel_hull_grid", type=int, default=64)
    ap.add_argument("--voxel_hull_scale", type=float, default=0.25)
    ap.add_argument("--voxel_hull_samples", type=int, default=128)
    ap.add_argument("--voxel_hull_bounds_frac", type=float, default=1.05)
    ap.add_argument("--voxel_hull_min_view_frac", type=float, default=0.75)
    ap.add_argument("--voxel_hull_mask_margin", type=int, default=1)
    ap.add_argument("--voxel_hull_dilate", type=int, default=1)
    ap.add_argument("--image_hull_clamp_depth", type=int, default=0) # clamp front depth outliers to source-mask hull
    ap.add_argument("--image_hull_clamp_mode", default="voxel", choices=["visual", "voxel"])
    ap.add_argument("--image_hull_clamp_tol_frac", type=float, default=0.0)
    ap.add_argument("--image_hull_clamp_max_shift_frac", type=float, default=0.0)
    ap.add_argument("--image_plane_sweep_depth", type=int, default=0) # RGB/mask plane-sweep from source views
    ap.add_argument("--image_guided_plane_sweep_depth", type=int, default=0) # refine input depth by local plane sweep
    ap.add_argument("--plane_sweep_scale", type=float, default=0.25)
    ap.add_argument("--plane_sweep_samples", type=int, default=64)
    ap.add_argument("--plane_sweep_refs", type=int, default=4)
    ap.add_argument("--plane_sweep_color_weight", type=float, default=1.0)
    ap.add_argument("--plane_sweep_mask_weight", type=float, default=0.25)
    ap.add_argument("--plane_sweep_front_bias", type=float, default=0.01)
    ap.add_argument("--plane_sweep_mask_margin", type=int, default=1)
    ap.add_argument("--guided_plane_sweep_radius_frac", type=float, default=0.12)
    ap.add_argument("--guided_plane_sweep_prior_weight", type=float, default=0.1)
    ap.add_argument("--guided_plane_sweep_accept_margin", type=float, default=-1.0)
    ap.add_argument("--guided_plane_sweep_top2_margin", type=float, default=-1.0)
    ap.add_argument("--guided_plane_sweep_max_shift_frac", type=float, default=1.0)
    ap.add_argument("--image_visibility_condition", type=int, default=0) # append depth-consistency visibility/confidence
    ap.add_argument("--image_photo_visibility_condition", type=int, default=0) # append RGB reprojection confidence
    ap.add_argument("--image_confidence_condition", type=int, default=0) # append cached depth-estimator confidence
    ap.add_argument("--image_normal_condition", type=int, default=0) # append world-space depth normals
    ap.add_argument("--image_visibility_all_views", type=int, default=1)
    ap.add_argument("--image_visibility_nearest_refs", type=int, default=0)
    ap.add_argument("--image_visibility_bidirectional", type=int, default=0)
    ap.add_argument("--image_visibility_min_support", type=int, default=0)
    ap.add_argument("--image_visibility_decay", type=float, default=0.25)
    ap.add_argument("--image_photo_visibility_refs", type=int, default=2)
    ap.add_argument("--image_photo_visibility_color_decay", type=float, default=12.0)
    ap.add_argument("--image_opacity_fg", type=float, default=0.2)
    ap.add_argument("--image_opacity_bg", type=float, default=0.001)
    ap.add_argument("--image_residual_scale", type=float, default=0.1)
    ap.add_argument("--image_rgb_residual_scale", type=float, default=None)
    ap.add_argument("--image_opacity_residual_scale", type=float, default=None)
    ap.add_argument("--image_scale_frac", type=float, default=0.0)
    ap.add_argument("--image_geom_residual_scale", type=float, default=1.0)
    ap.add_argument("--explicit_depth_head", type=int, default=0)
    ap.add_argument("--explicit_visibility_head", type=int, default=0)
    ap.add_argument("--image_depth_prior_frac", type=float, default=0.0)
    ap.add_argument("--image_depth_skip", type=int, default=0)
    ap.add_argument("--image_depth_residual_scale", type=float, default=0.0)
    ap.add_argument("--residual_rgb_weight", type=float, default=0.0)
    ap.add_argument("--residual_geom_weight", type=float, default=0.0)
    ap.add_argument("--residual_depth_weight", type=float, default=0.0)
    ap.add_argument("--residual_opacity_weight", type=float, default=0.0)
    ap.add_argument("--residual_offset_weight", type=float, default=0.0)
    ap.add_argument("--zero_init_head", type=int, default=0)
    ap.add_argument("--image_visibility_skip", type=int, default=0)
    ap.add_argument("--image_normal_scale_frac", type=float, default=0.0)
    ap.add_argument("--image_boundary_scale_mult", type=float, default=1.0)
    ap.add_argument("--image_boundary_width", type=int, default=0)
    ap.add_argument("--image_camera_quat", type=int, default=0)
    ap.add_argument("--image_normal_quat", type=int, default=0)
    ap.add_argument("--depth_head_scale", type=float, default=1.0)
    ap.add_argument("--visibility_head_scale", type=float, default=1.0)
    ap.add_argument("--freeze_decoder", type=int, default=0)
    ap.add_argument("--min_free_vram_gb", type=float, default=8.0)    # local safety guard
    args = ap.parse_args()
    for _name in ("cond_subdir", "cond_depth_subdir", "cond_conf_subdir"):
        _val = getattr(args, _name, None)
        if isinstance(_val, str) and _val.lower() in {"", "none", "null", "default"}:
            setattr(args, _name, None)
    if args.fusion_depth_filter and args.fusion_filter_mode != "opacity" and (
        args.anchor_rgb_weight > 0 or args.anchor_opacity_weight > 0
        or args.anchor_scale_weight > 0 or args.anchor_visibility_weight > 0
        or args.anchor_depth_weight > 0
    ):
        raise ValueError(
            "--fusion_depth_filter with hard pruning changes anchor grid length; "
            "use --fusion_filter_mode opacity for direct per-anchor losses"
        )
    if args.image_visibility_condition and not args.image_depth_condition:
        raise ValueError("--image_visibility_condition requires --image_depth_condition")
    if args.image_photo_visibility_condition and not args.image_depth_condition:
        raise ValueError("--image_photo_visibility_condition requires --image_depth_condition")
    if args.image_confidence_condition and args.condition_confidence_power <= 0:
        raise ValueError("--image_confidence_condition requires --condition_confidence_power > 0")
    if args.image_normal_condition and not args.image_depth_condition:
        raise ValueError("--image_normal_condition requires --image_depth_condition")
    if args.image_visual_hull_depth and not args.image_depth_condition:
        raise ValueError("--image_visual_hull_depth requires --image_depth_condition")
    if args.image_voxel_hull_depth and not args.image_depth_condition:
        raise ValueError("--image_voxel_hull_depth requires --image_depth_condition")
    if args.image_hull_clamp_depth and not args.image_depth_condition:
        raise ValueError("--image_hull_clamp_depth requires --image_depth_condition")
    if args.image_plane_sweep_depth and not args.image_depth_condition:
        raise ValueError("--image_plane_sweep_depth requires --image_depth_condition")
    if args.image_guided_plane_sweep_depth and not args.image_depth_condition:
        raise ValueError("--image_guided_plane_sweep_depth requires --image_depth_condition")
    n_depth_sources = (int(args.image_visual_hull_depth) + int(args.image_voxel_hull_depth)
                       + int(args.image_plane_sweep_depth)
                       + int(args.image_guided_plane_sweep_depth))
    if n_depth_sources > 1:
        raise ValueError("choose at most one image-derived depth source")
    if args.support_gate_unet and not args.image_depth_condition:
        raise ValueError("--support_gate_unet requires --image_depth_condition")
    if args.surface_confidence_unet and not args.image_depth_condition:
        raise ValueError("--surface_confidence_unet requires --image_depth_condition")
    if args.surface_refine_unet and not args.image_depth_condition:
        raise ValueError("--surface_refine_unet requires --image_depth_condition")
    dev = "cuda"
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    free_b, _ = torch.cuda.mem_get_info()
    free_gb = free_b / (1024 ** 3)
    if free_gb < args.min_free_vram_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB CUDA memory is free; refusing to start. "
            f"Lower --min_free_vram_gb or stop other GPU containers."
        )

    import json
    import numpy as np
    import sys
    from torch.utils.data import DataLoader
    from decoder.data import (ObjaverseLatentDataset, depth_target_on_grid, entry_uid,
                              latent_path_for_entry, load_cameras, load_depth_view,
                              object_dir_for_entry, opengl_c2w_to_opencv_w2c,
                              zdepth_to_raydist)
    from decoder.clean.geometry import ray_dirs_world
    from decoder.render import render_views, _ssim
    from decoder.clean.network import CleanGSDecoder
    from decoder.clean.losses import (mask_alpha_l1, scale_hinge, VGGPerceptual,
                                       render_expected_depth, scale_invariant_depth_loss,
                                       absolute_depth_loss,
                                       opacity_scale_reg, opacity_entropy)
    import torch.nn.functional as Fnn
    from decoder.clean.metrics import fg_masked_psnr, sharpness_ratio
    from decoder.clean.fusion import (rgbd_fit_sh_colors, rgbd_target_view_surface,
                                      rgbd_target_view_surface_splat,
                                      rgbd_tsdf_filter_params, rgbd_tsdf_fuse,
                                      voxel_fuse_params)
    from decoder.clean.condition_refine import (ConditionDepthAffineHead,
                                                ConditionDepthConfidenceUNet,
                                                ConditionMaskRefineUNet,
                                                ConditionPoseHead,
                                                ConditionRGBDRefineUNet,
                                                ConditionRGBDViewRefineUNet,
                                                ConditionRGBRefineUNet,
                                                OutputAlphaRefineUNet,
                                                apply_depth_confidence_head,
                                                apply_rgbd_refiner,
                                                apply_mask_refiner,
                                                apply_output_alpha_refiner,
                                                apply_rgb_refiner,
                                                rgb_border_mask)
    from decoder.clean.phase2_data import (Phase2Dataset, available_frame_indices,
                                           depth_path_at, frame_path_at,
                                           load_conf_view_at, load_depth_view_at,
                                           load_masks_at, load_views_at,
                                           resolve_view_spec)
    from decoder.clean.geometry import depth_bounds

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n")

    aux_ckpt_paths = [
        Path(p.strip()) for p in str(args.resume_aux_from or "").split(",")
        if p.strip()
    ]

    # --- data: subsampling loader for training; full-view plain dataset for eval ---
    cond_spec = None
    cond_default_views = None
    if args.condition_source == "fixed":
        cond_spec = args.cond_view_indices
        if cond_spec == "" and args.cond_subdir:
            cond_spec = "available"
        elif cond_spec == "":
            cond_default_views = args.anchor_views

    def _filter_condition_entries(entries: list, label: str,
                                  require_nonempty: bool = True) -> list:
        if not args.filter_missing_condition or args.condition_source != "fixed":
            return entries
        kept = []
        examples = []
        min_views = max(int(args.filter_missing_condition_min_views), 1)
        for entry in entries:
            uid = entry_uid(entry)
            obj_dir = object_dir_for_entry(V4, entry)
            try:
                if cond_spec == "available":
                    idxs = available_frame_indices(obj_dir, subdir=args.cond_subdir)
                else:
                    cams = load_cameras(obj_dir / "cameras.json")
                    idxs = resolve_view_spec(
                        cond_spec,
                        cams["w2c"].shape[0],
                        obj_dir=obj_dir,
                        subdir=args.cond_subdir,
                        n_orbit_views=cams["num_orbit_views"],
                        default_n=cond_default_views,
                    )
                if idxs is None or len(idxs) < min_views:
                    raise FileNotFoundError(f"only {0 if idxs is None else len(idxs)} condition views")
                if args.cond_subdir:
                    for view_i in idxs:
                        if not frame_path_at(obj_dir, view_i, subdir=args.cond_subdir).exists():
                            raise FileNotFoundError(f"missing condition frame {view_i:03d}")
                if args.cond_depth_subdir:
                    for view_i in idxs:
                        if not depth_path_at(obj_dir, view_i, subdir=args.cond_depth_subdir).exists():
                            raise FileNotFoundError(f"missing condition depth {view_i:03d}")
                if args.cond_visibility_depth_subdir:
                    for view_i in idxs:
                        if not depth_path_at(obj_dir, view_i, subdir=args.cond_visibility_depth_subdir).exists():
                            raise FileNotFoundError(f"missing visibility depth {view_i:03d}")
            except Exception as ex:
                if len(examples) < 5:
                    examples.append(f"{uid[:10]}:{type(ex).__name__}")
                continue
            kept.append(entry)
        print(
            f"[phase2] condition filter {label}: kept={len(kept)}/{len(entries)} "
            f"min_views={min_views} examples={examples}",
            flush=True,
        )
        if require_nonempty and not kept:
            raise RuntimeError(
                f"condition filter removed all {label} objects; check cond_subdir/depth sidecars"
            )
        return kept

    train_ds = Phase2Dataset(V4, "train", MANIFEST, k_views=args.k_views,
                             cond_subdir=args.cond_subdir,
                             cond_depth_subdir=args.cond_depth_subdir,
                             cond_visibility_depth_subdir=args.cond_visibility_depth_subdir,
                             cond_conf_subdir=args.cond_conf_subdir,
                             cond_view_spec=cond_spec,
                             cond_default_views=cond_default_views,
                             strict_cond_depth=bool(args.strict_condition_depth))
    train_ds.entries = _filter_condition_entries(train_ds.entries, "train")
    if args.max_train_objects > 0:
        train_ds.entries = train_ds.entries[:min(args.max_train_objects, len(train_ds.entries))]
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers,
                        collate_fn=_collate_first, drop_last=True,
                        persistent_workers=args.num_workers > 0)

    def _infer_latent_grid() -> tuple[int, int, int]:
        for entry in train_ds.entries:
            latent_path = latent_path_for_entry(V4, entry)
            if not latent_path.exists():
                continue
            latent = np.load(latent_path, mmap_mode="r")
            if latent.ndim != 4:
                uid = entry_uid(entry)
                raise ValueError(f"expected latent.npy shape (C,T,H,W), got {latent.shape} for {uid}")
            return int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3])
        raise FileNotFoundError("could not infer latent grid: no train latent.npy files found")

    inferred_t, inferred_h, inferred_w = _infer_latent_grid()
    args.latent_t = int(args.latent_t) if args.latent_t > 0 else inferred_t
    args.latent_h = int(args.latent_h) if args.latent_h > 0 else inferred_h
    args.latent_w = int(args.latent_w) if args.latent_w > 0 else inferred_w
    if (args.latent_t, args.latent_h, args.latent_w) != (inferred_t, inferred_h, inferred_w):
        raise ValueError(
            f"configured latent grid {(args.latent_t, args.latent_h, args.latent_w)} "
            f"does not match dataset {(inferred_t, inferred_h, inferred_w)}"
        )
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    load_eval_depth = bool(args.oracle_anchor_depth or args.image_depth_condition
                           or args.image_visibility_condition or args.fusion_depth_filter)
    ev = ObjaverseLatentDataset(V4, "eval", manifest_path=MANIFEST,
                                load_depths=load_eval_depth,
                                max_views=args.eval_views_per_object)
    te = ObjaverseLatentDataset(V4, "test", manifest_path=MANIFEST,
                                load_depths=load_eval_depth,
                                max_views=args.eval_views_per_object)
    tr = ObjaverseLatentDataset(V4, "train", manifest_path=MANIFEST,
                                load_depths=load_eval_depth,
                                max_views=args.eval_views_per_object)
    ev.entries = _filter_condition_entries(ev.entries, "eval")
    te.entries = _filter_condition_entries(te.entries, "test", require_nonempty=False)
    tr.entries = _filter_condition_entries(tr.entries, "train_eval")
    if args.n_heldout_eval > 0:
        n_ev = min(len(ev), args.n_heldout_eval)
        heldout = _load_safe(ev, n_ev)
        rem = args.n_heldout_eval - len(heldout)
        if rem > 0:
            heldout += _load_safe(te, min(len(te), rem))
    else:
        heldout = _load_safe(ev, len(ev)) + _load_safe(te, len(te))
    train_eval = _load_safe(tr, min(args.n_train_eval, len(tr)))

    condition_mask_refine_head = None
    if args.condition_mask_refine_unet:
        condition_mask_refine_head = ConditionMaskRefineUNet(
            hidden=args.condition_mask_refine_hidden
        ).to(dev)
        for ckpt_path in aux_ckpt_paths:
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
            state = ckpt.get("condition_mask_refine_head")
            if state is not None:
                condition_mask_refine_head.load_state_dict(state)
                print(
                    f"[phase2] loaded auxiliary condition_mask_refine_head from {ckpt_path}",
                    flush=True,
                )

    def _apply_condition_mask_source(frames: torch.Tensor,
                                     masks: torch.Tensor) -> torch.Tensor:
        """Choose the mask supplied to the conditioning branch.

        ``gt`` is the historical path and uses dataset masks. The RGB-derived
        options are inference-realistic for LTX-decoded conditioning because
        they use only the decoded frame pixels and the known white-background
        rendering convention.
        """
        if args.condition_mask_source == "gt":
            return masks.to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)

        def refine_mask(prior: torch.Tensor) -> torch.Tensor:
            if condition_mask_refine_head is None:
                return prior
            head_device = next(condition_mask_refine_head.parameters()).device
            enable_grad = torch.is_grad_enabled() and frames.device == head_device
            with torch.set_grad_enabled(enable_grad):
                refined = apply_mask_refiner(
                    condition_mask_refine_head,
                    frames.to(device=head_device),
                    prior.to(device=head_device),
                    args.condition_mask_refine_scale,
                )
            return refined.to(device=frames.device, dtype=frames.dtype)

        rgb = frames.to(dtype=torch.float32).clamp(0.0, 1.0)
        if args.condition_mask_source == "rgb_white":
            bg = torch.ones((rgb.shape[0], 1, 1, 3), device=rgb.device, dtype=rgb.dtype)
        elif args.condition_mask_source == "rgb_border":
            out = rgb_border_mask(
                frames,
                threshold=args.condition_mask_rgb_threshold,
                softness=args.condition_mask_rgb_softness,
            )
            x = out.permute(0, 3, 1, 2)
            if args.condition_mask_rgb_dilate_px > 0:
                r = int(args.condition_mask_rgb_dilate_px)
                x = Fnn.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r)
            if args.condition_mask_rgb_erode_px > 0:
                r = int(args.condition_mask_rgb_erode_px)
                x = 1.0 - Fnn.max_pool2d(1.0 - x, kernel_size=2 * r + 1, stride=1, padding=r)
            out = x.permute(0, 2, 3, 1).to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)
            return refine_mask(out)
        else:
            raise ValueError(f"unknown condition_mask_source={args.condition_mask_source}")
        score = torch.linalg.vector_norm(rgb - bg, dim=-1, keepdim=True) / math.sqrt(3.0)
        thresh = float(args.condition_mask_rgb_threshold)
        softness = float(args.condition_mask_rgb_softness)
        if softness > 0:
            out = torch.sigmoid((score - thresh) / max(softness, 1e-6))
        else:
            out = (score > thresh).to(rgb.dtype)
        x = out.permute(0, 3, 1, 2)
        if args.condition_mask_rgb_dilate_px > 0:
            r = int(args.condition_mask_rgb_dilate_px)
            x = Fnn.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r)
        if args.condition_mask_rgb_erode_px > 0:
            r = int(args.condition_mask_rgb_erode_px)
            x = 1.0 - Fnn.max_pool2d(1.0 - x, kernel_size=2 * r + 1, stride=1, padding=r)
        out = x.permute(0, 2, 3, 1).to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)
        return refine_mask(out)

    def _fit_condition_color_calibration() -> torch.Tensor | None:
        if args.condition_color_calibration == "none":
            return None
        if args.condition_source != "fixed" or not args.cond_subdir:
            print("[phase2] color calibration skipped: requires fixed conditioning frames",
                  flush=True)
            return None
        max_obj = len(train_ds.entries)
        if args.condition_color_calib_max_objects > 0:
            max_obj = min(max_obj, args.condition_color_calib_max_objects)
        ata = torch.zeros(4, 4, dtype=torch.float64)
        atb = torch.zeros(4, 3, dtype=torch.float64)
        n_px = 0
        n_views = 0
        for entry in train_ds.entries[:max_obj]:
            obj_dir = object_dir_for_entry(V4, entry)
            try:
                cams = load_cameras(obj_dir / "cameras.json")
                idxs = resolve_view_spec(
                    cond_spec,
                    cams["w2c"].shape[0],
                    obj_dir=obj_dir,
                    subdir=args.cond_subdir,
                    n_orbit_views=cams["num_orbit_views"],
                    default_n=cond_default_views,
                )
                if idxs is None:
                    continue
                if args.condition_color_calib_views > 0:
                    idxs = idxs[:args.condition_color_calib_views]
                cond_frames_calib = load_views_at(obj_dir, idxs, subdir=args.cond_subdir)
                cond_rgb = cond_frames_calib.reshape(-1, 3)
                target_rgb = load_views_at(obj_dir, idxs, subdir=None).reshape(-1, 3)
                mask = _apply_condition_mask_source(
                    cond_frames_calib,
                    load_masks_at(obj_dir, idxs),
                ).reshape(-1) > 0.5
            except Exception as ex:
                print(
                    f"[phase2] color calibration skip {uid[:10]} "
                    f"({type(ex).__name__})",
                    flush=True,
                )
                continue
            if not mask.any():
                continue
            x = cond_rgb[mask].to(torch.float64)
            y = target_rgb[mask].to(torch.float64)
            if args.condition_color_calib_sample_px > 0 and x.shape[0] > args.condition_color_calib_sample_px:
                stride = math.ceil(x.shape[0] / args.condition_color_calib_sample_px)
                x = x[::stride]
                y = y[::stride]
            feat = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
            ata += feat.T @ feat
            atb += feat.T @ y
            n_px += int(x.shape[0])
            n_views += len(idxs)
        if n_px < 16:
            print("[phase2] color calibration skipped: not enough foreground pixels",
                  flush=True)
            return None
        ridge = max(float(args.condition_color_calib_ridge), 0.0)
        reg = torch.eye(4, dtype=torch.float64) * ridge
        reg[-1, -1] = 0.0
        sol = torch.linalg.solve(ata + reg, atb).to(torch.float32)
        w = sol[:3]
        b = sol[3]
        print(
            "[phase2] color calibration train_affine "
            f"objs={max_obj} views={n_views} px={n_px} "
            f"W={w.flatten().tolist()} b={b.tolist()}",
            flush=True,
        )
        return sol

    color_calib = _fit_condition_color_calibration()
    condition_rgb_refine_head = None
    if args.condition_rgb_refine_unet:
        condition_rgb_refine_head = ConditionRGBRefineUNet(
            hidden=args.condition_rgb_refine_hidden
        ).to(dev)
    condition_rgbd_refine_head = None
    if args.condition_rgbd_refine_unet:
        rgbd_in_channels = 6 + (4 if args.condition_rgbd_refine_multiview_features else 0)
        if args.condition_rgbd_refine_arch == "view":
            condition_rgbd_refine_head = ConditionRGBDViewRefineUNet(
                hidden=args.condition_rgbd_refine_hidden,
                in_channels=rgbd_in_channels,
                max_views=args.condition_rgbd_refine_max_views,
                context_layers=args.condition_rgbd_refine_context_layers,
                context_heads=args.condition_rgbd_refine_context_heads,
            ).to(dev)
        else:
            condition_rgbd_refine_head = ConditionRGBDRefineUNet(
                hidden=args.condition_rgbd_refine_hidden,
                in_channels=rgbd_in_channels,
            ).to(dev)
    condition_pose_head = None
    if args.condition_pose_head:
        condition_pose_head = ConditionPoseHead(
            hidden=args.condition_pose_hidden,
            max_views=args.condition_pose_max_views,
            context_layers=args.condition_pose_context_layers,
            context_heads=args.condition_pose_context_heads,
        ).to(dev)
        if args.freeze_condition_pose_head:
            for p_pose in condition_pose_head.parameters():
                p_pose.requires_grad_(False)
            condition_pose_head.eval()
    if args.condition_pose_use_predicted and condition_pose_head is None:
        raise ValueError("--condition_pose_use_predicted requires --condition_pose_head 1")
    condition_depth_affine_head = None
    if args.condition_depth_affine_head:
        condition_depth_affine_head = ConditionDepthAffineHead(
            in_features=12,
            hidden=args.condition_depth_affine_hidden,
            layers=args.condition_depth_affine_layers,
        ).to(dev)
    condition_depth_confidence_head = None
    if args.condition_depth_confidence_unet:
        conf_in_channels = (
            6 + (4 if args.condition_depth_confidence_multiview_features else 0)
        )
        condition_depth_confidence_head = ConditionDepthConfidenceUNet(
            hidden=args.condition_depth_confidence_hidden,
            in_channels=conf_in_channels,
        ).to(dev)

    def _apply_condition_color_calibration(frames: torch.Tensor) -> torch.Tensor:
        if color_calib is None:
            return frames
        sol = color_calib.to(device=frames.device, dtype=frames.dtype)
        shape = frames.shape
        flat = frames.reshape(-1, 3)
        out_rgb = flat @ sol[:3] + sol[3]
        return out_rgb.reshape(shape).clamp(0.0, 1.0)

    def _apply_condition_rgb_refine(frames: torch.Tensor,
                                    masks: torch.Tensor) -> torch.Tensor:
        if condition_rgb_refine_head is None:
            return frames
        return apply_rgb_refiner(
            condition_rgb_refine_head,
            frames,
            masks,
            args.condition_rgb_refine_scale,
        )

    def _append_condition_rgb_refine_gt_loss(refined: torch.Tensor,
                                             target: torch.Tensor | None,
                                             masks: torch.Tensor) -> None:
        if (condition_rgb_refine_head is None
                or args.condition_rgb_refine_gt_weight <= 0
                or target is None
                or not torch.is_grad_enabled()):
            return
        target = target.to(device=refined.device, dtype=refined.dtype)
        if target.shape != refined.shape:
            return
        alpha_min = min(max(float(args.condition_rgb_refine_gt_alpha_min), 0.0), 1.0)
        valid = masks.to(device=refined.device, dtype=refined.dtype).clamp(0.0, 1.0)
        valid = (valid > alpha_min).to(dtype=refined.dtype)
        denom = (valid.sum() * refined.shape[-1]).clamp_min(1.0)
        condition_rgb_refine_gt_terms.append(((refined - target).abs() * valid).sum() / denom)

    def _apply_condition_rgbd_refine(frames: torch.Tensor,
                                     fg: torch.Tensor,
                                     depths: torch.Tensor | None,
                                     K_all: torch.Tensor,
                                     c2w_all: torch.Tensor,
                                     radius: float,
                                     target_frames: torch.Tensor | None = None,
                                     target_depths: torch.Tensor | None = None
                                     ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if condition_rgbd_refine_head is None or depths is None:
            return frames, depths
        depth_frac, depth_valid, factors = _depth_frac_valid_factor(
            depths, fg, K_all, c2w_all, radius
        )
        apply_valid = depth_valid
        if args.condition_rgbd_refine_apply_erode_px > 0:
            apply_valid = apply_valid * _erode_mask_2d(
                fg[..., 0].clamp(0.0, 1.0),
                args.condition_rgbd_refine_apply_erode_px,
            )
        mv_feat = None
        if args.condition_rgbd_refine_multiview_features:
            mv_feat = _depth_multiview_support_maps(
                depths,
                fg,
                K_all,
                c2w_all,
                radius,
                args.condition_rgbd_refine_multiview_tol_frac,
                args.condition_rgbd_refine_multiview_refs,
                args.condition_rgbd_refine_multiview_radius_px,
            )
        head_device = next(condition_rgbd_refine_head.parameters()).device
        head_dtype = next(condition_rgbd_refine_head.parameters()).dtype
        frame_in = frames.to(device=head_device, dtype=head_dtype)
        fg_in = fg.to(device=head_device, dtype=head_dtype)
        frac_in = depth_frac.to(device=head_device, dtype=head_dtype)
        valid_in = depth_valid.to(device=head_device, dtype=head_dtype)
        apply_in = apply_valid.to(device=head_device, dtype=head_dtype)
        mv_in = mv_feat.to(device=head_device, dtype=head_dtype) if mv_feat is not None else None
        if args.condition_rgbd_refine_detach_inputs:
            frame_in = frame_in.detach()
            fg_in = fg_in.detach()
            frac_in = frac_in.detach()
            valid_in = valid_in.detach()
            apply_in = apply_in.detach()
            if mv_in is not None:
                mv_in = mv_in.detach()
        refined_frames_h, refined_frac_h, rgb_delta, depth_delta = apply_rgbd_refiner(
            condition_rgbd_refine_head,
            frame_in,
            fg_in,
            frac_in,
            valid_in,
            args.condition_rgbd_refine_rgb_scale,
            args.condition_rgbd_refine_depth_scale,
            extra_features=mv_in,
            apply_valid=apply_in,
        )
        refined_frames = refined_frames_h.to(device=frames.device, dtype=frames.dtype)
        refined_frac = refined_frac_h.to(device=depths.device, dtype=depths.dtype)
        refined_z = _depth_frac_to_z(
            refined_frac,
            factors.to(device=depths.device, dtype=depths.dtype),
            c2w_all,
            radius,
        )
        refined_depths = torch.where(apply_valid > 0.5, refined_z, depths)

        if torch.is_grad_enabled():
            valid_h = apply_in[:, None]
            denom = valid_h.sum().clamp_min(1.0)
            condition_rgbd_refine_delta_terms.append(
                0.5 * (
                    (rgb_delta.square() * valid_h).sum()
                    / (denom * max(rgb_delta.shape[1], 1))
                    + (depth_delta.square() * valid_h).sum() / denom
                )
            )
            if depth_delta.shape[-1] > 1 and depth_delta.shape[-2] > 1:
                valid_px = apply_in
                valid_x = valid_px[:, :, 1:] * valid_px[:, :, :-1]
                valid_y = valid_px[:, 1:, :] * valid_px[:, :-1, :]
                delta_all = torch.cat([rgb_delta, depth_delta], dim=1)
                tv_x = ((delta_all[:, :, :, 1:] - delta_all[:, :, :, :-1]).abs()
                        * valid_x[:, None]).sum() / valid_x.sum().clamp_min(1.0)
                tv_y = ((delta_all[:, :, 1:, :] - delta_all[:, :, :-1, :]).abs()
                        * valid_y[:, None]).sum() / valid_y.sum().clamp_min(1.0)
                condition_rgbd_refine_tv_terms.append(0.5 * (tv_x + tv_y))
            if target_frames is not None:
                target = target_frames.to(device=head_device, dtype=head_dtype)
                if target.shape == refined_frames_h.shape:
                    alpha_min = min(max(float(args.condition_rgbd_refine_gt_alpha_min), 0.0), 1.0)
                    valid_rgb = (fg_in > alpha_min).to(dtype=head_dtype)
                    denom_rgb = (valid_rgb.sum() * refined_frames_h.shape[-1]).clamp_min(1.0)
                    condition_rgbd_refine_rgb_gt_terms.append(
                        ((refined_frames_h - target).abs() * valid_rgb).sum() / denom_rgb
                    )
            if target_depths is not None:
                tgt_frac, tgt_valid, _ = _depth_frac_valid_factor(
                    target_depths.to(device=depths.device, dtype=depths.dtype),
                    fg, K_all, c2w_all, radius,
                )
                tgt_frac_h = tgt_frac.to(device=head_device, dtype=head_dtype)
                tgt_valid_h = tgt_valid.to(device=head_device, dtype=head_dtype)
                valid_depth = (tgt_valid_h > 0.5) & (apply_in > 0.5)
                if valid_depth.any():
                    condition_rgbd_refine_depth_gt_terms.append(
                        Fnn.huber_loss(
                            refined_frac_h[valid_depth],
                            tgt_frac_h[valid_depth],
                            delta=0.02,
                        )
                    )
        return refined_frames, refined_depths

    def _condition_depth_affine_features(frames: torch.Tensor,
                                         fg: torch.Tensor,
                                         depth_frac: torch.Tensor,
                                         depth_valid: torch.Tensor) -> torch.Tensor:
        mask = fg[..., 0].to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)
        valid = depth_valid.to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0) * mask
        denom = valid.sum(dim=(1, 2)).clamp_min(1.0)
        h, w = depth_frac.shape[-2:]
        yy = torch.linspace(-1.0, 1.0, h, device=frames.device, dtype=frames.dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=frames.device, dtype=frames.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        area = mask.mean(dim=(1, 2), keepdim=False)[:, None]
        valid_area = (depth_valid.to(dtype=frames.dtype) > 0.5).to(dtype=frames.dtype).mean(
            dim=(1, 2), keepdim=False
        )[:, None]
        depth_mean = (depth_frac * valid).sum(dim=(1, 2), keepdim=False) / denom
        depth_var = ((depth_frac - depth_mean[:, None, None]).square() * valid).sum(
            dim=(1, 2), keepdim=False
        ) / denom
        rgb = frames.to(dtype=frames.dtype).clamp(0.0, 1.0)
        rgb_mean = (rgb * valid[..., None]).sum(dim=(1, 2)) / denom[:, None]
        rgb_var = ((rgb - rgb_mean[:, None, None, :]).square() * valid[..., None]).sum(
            dim=(1, 2)
        ) / denom[:, None]
        cx = (grid_x[None] * mask).sum(dim=(1, 2), keepdim=False) / mask.sum(
            dim=(1, 2), keepdim=False
        ).clamp_min(1.0)
        cy = (grid_y[None] * mask).sum(dim=(1, 2), keepdim=False) / mask.sum(
            dim=(1, 2), keepdim=False
        ).clamp_min(1.0)
        return torch.cat([
            area,
            valid_area,
            depth_mean[:, None].clamp(0.0, 1.0),
            depth_var.sqrt()[:, None].clamp(0.0, 1.0),
            rgb_mean.clamp(0.0, 1.0),
            rgb_var.sqrt().clamp(0.0, 1.0),
            cx[:, None].clamp(-1.0, 1.0),
            cy[:, None].clamp(-1.0, 1.0),
        ], dim=-1)

    def _apply_condition_depth_affine(frames: torch.Tensor,
                                      fg: torch.Tensor,
                                      depths: torch.Tensor | None,
                                      K_all: torch.Tensor,
                                      c2w_all: torch.Tensor,
                                      radius: float,
                                      target_depths: torch.Tensor | None = None
                                      ) -> torch.Tensor | None:
        if condition_depth_affine_head is None or depths is None:
            return depths
        depth_frac, depth_valid, factors = _depth_frac_valid_factor(
            depths, fg, K_all, c2w_all, radius
        )
        head_device = next(condition_depth_affine_head.parameters()).device
        head_dtype = next(condition_depth_affine_head.parameters()).dtype
        feat = _condition_depth_affine_features(
            frames.to(device=head_device, dtype=head_dtype),
            fg.to(device=head_device, dtype=head_dtype),
            depth_frac.to(device=head_device, dtype=head_dtype),
            depth_valid.to(device=head_device, dtype=head_dtype),
        )
        if args.condition_depth_affine_detach_inputs:
            feat = feat.detach()
        raw = condition_depth_affine_head(feat)
        scale_delta = torch.tanh(raw[:, 0]) * max(float(args.condition_depth_affine_scale_range), 0.0)
        shift = torch.tanh(raw[:, 1]) * max(float(args.condition_depth_affine_shift_range), 0.0)
        scale = 1.0 + scale_delta
        refined_frac_h = (
            depth_frac.to(device=head_device, dtype=head_dtype) * scale[:, None, None]
            + shift[:, None, None]
        ).clamp(1e-4, 1.0 - 1e-4)
        refined_frac = refined_frac_h.to(device=depths.device, dtype=depths.dtype)
        refined_z = _depth_frac_to_z(
            refined_frac,
            factors.to(device=depths.device, dtype=depths.dtype),
            c2w_all,
            radius,
        )
        refined_depths = torch.where(depth_valid > 0.5, refined_z, depths)

        if torch.is_grad_enabled():
            condition_depth_affine_delta_terms.append(
                0.5 * (scale_delta.square().mean() + shift.square().mean())
            )
            if target_depths is not None and args.condition_depth_affine_gt_weight > 0:
                tgt_frac, tgt_valid, _ = _depth_frac_valid_factor(
                    target_depths.to(device=depths.device, dtype=depths.dtype),
                    fg, K_all, c2w_all, radius,
                )
                tgt_frac_h = tgt_frac.to(device=head_device, dtype=head_dtype)
                tgt_valid_h = tgt_valid.to(device=head_device, dtype=head_dtype)
                valid_h = (
                    (tgt_valid_h > 0.5)
                    & (depth_valid.to(device=head_device, dtype=head_dtype) > 0.5)
                    & (fg[..., 0].to(device=head_device, dtype=head_dtype) > 0.5)
                )
                if valid_h.any():
                    condition_depth_affine_gt_terms.append(
                        Fnn.huber_loss(
                            refined_frac_h[valid_h],
                            tgt_frac_h[valid_h],
                            delta=0.02,
                        )
                    )
        return refined_depths

    def _apply_condition_depth_confidence(frames: torch.Tensor,
                                          fg: torch.Tensor,
                                          depths: torch.Tensor | None,
                                          K_all: torch.Tensor,
                                          c2w_all: torch.Tensor,
                                          radius: float,
                                          target_depths: torch.Tensor | None = None,
                                          base_confs: torch.Tensor | None = None
                                          ) -> torch.Tensor | None:
        if condition_depth_confidence_head is None:
            return base_confs
        if depths is None:
            return base_confs
        depth_frac, depth_valid, _ = _depth_frac_valid_factor(
            depths, fg, K_all, c2w_all, radius
        )
        mv_feat = None
        if args.condition_depth_confidence_multiview_features:
            mv_feat = _depth_multiview_support_maps(
                depths,
                fg,
                K_all,
                c2w_all,
                radius,
                args.condition_depth_confidence_multiview_tol_frac,
                args.condition_depth_confidence_multiview_refs,
                args.condition_depth_confidence_multiview_radius_px,
            )
        head_device = next(condition_depth_confidence_head.parameters()).device
        head_dtype = next(condition_depth_confidence_head.parameters()).dtype
        frame_in = frames.to(device=head_device, dtype=head_dtype)
        fg_in = fg.to(device=head_device, dtype=head_dtype)
        frac_in = depth_frac.to(device=head_device, dtype=head_dtype)
        valid_in = depth_valid.to(device=head_device, dtype=head_dtype)
        mv_in = mv_feat.to(device=head_device, dtype=head_dtype) if mv_feat is not None else None
        if args.condition_depth_confidence_detach_inputs:
            frame_in = frame_in.detach()
            fg_in = fg_in.detach()
            frac_in = frac_in.detach()
            valid_in = valid_in.detach()
            if mv_in is not None:
                mv_in = mv_in.detach()
        chunk_views = max(int(args.condition_depth_confidence_chunk_views), 0)
        if chunk_views > 0 and frame_in.shape[0] > chunk_views:
            conf_chunks: list[torch.Tensor] = []
            delta_chunks: list[torch.Tensor] = []
            for start in range(0, frame_in.shape[0], chunk_views):
                end = start + chunk_views
                mv_chunk = mv_in[start:end] if mv_in is not None else None
                conf_c, delta_c = apply_depth_confidence_head(
                    condition_depth_confidence_head,
                    frame_in[start:end],
                    fg_in[start:end],
                    frac_in[start:end],
                    valid_in[start:end],
                    args.condition_depth_confidence_init,
                    args.condition_depth_confidence_delta_scale,
                    args.condition_depth_confidence_floor,
                    extra_features=mv_chunk,
                )
                conf_chunks.append(conf_c)
                delta_chunks.append(delta_c)
            conf_h = torch.cat(conf_chunks, dim=0)
            delta_h = torch.cat(delta_chunks, dim=0)
        else:
            conf_h, delta_h = apply_depth_confidence_head(
                condition_depth_confidence_head,
                frame_in,
                fg_in,
                frac_in,
                valid_in,
                args.condition_depth_confidence_init,
                args.condition_depth_confidence_delta_scale,
                args.condition_depth_confidence_floor,
                extra_features=mv_in,
            )
        conf = conf_h.to(device=depths.device, dtype=depths.dtype)
        if torch.is_grad_enabled():
            valid_h = valid_in[:, None]
            if valid_h.any():
                condition_depth_confidence_delta_terms.append(
                    (delta_h.square() * valid_h).sum() / valid_h.sum().clamp_min(1.0)
                )
                if delta_h.shape[-1] > 1 and delta_h.shape[-2] > 1:
                    valid_px = valid_in
                    valid_x = valid_px[:, :, 1:] * valid_px[:, :, :-1]
                    valid_y = valid_px[:, 1:, :] * valid_px[:, :-1, :]
                    if valid_x.any() and valid_y.any():
                        tv_x = ((delta_h[:, :, :, 1:] - delta_h[:, :, :, :-1]).abs()
                                * valid_x[:, None]).sum() / valid_x.sum().clamp_min(1.0)
                        tv_y = ((delta_h[:, :, 1:, :] - delta_h[:, :, :-1, :]).abs()
                                * valid_y[:, None]).sum() / valid_y.sum().clamp_min(1.0)
                        condition_depth_confidence_tv_terms.append(0.5 * (tv_x + tv_y))
            if (target_depths is not None
                    and args.condition_depth_confidence_gt_weight > 0):
                target_frac, target_valid, _ = _depth_frac_valid_factor(
                    target_depths.to(device=depths.device, dtype=depths.dtype),
                    fg,
                    K_all,
                    c2w_all,
                    radius,
                )
                target_frac_h = target_frac.to(device=head_device, dtype=head_dtype)
                target_valid_h = target_valid.to(device=head_device, dtype=head_dtype)
                valid = (target_valid_h > 0.5) & (valid_in > 0.5)
                if valid.any():
                    tol = max(float(args.condition_depth_confidence_tol_frac), 1e-6)
                    target_conf, label_valid = _depth_confidence_targets(
                        frac_in,
                        target_frac_h,
                        valid,
                        tol,
                        float(args.condition_depth_confidence_neg_tol_frac),
                    )
                    floor = min(max(float(args.condition_depth_confidence_floor), 0.0), 1.0)
                    prob = ((conf_h - floor) / max(1.0 - floor, 1e-6)).clamp(1e-4, 1.0 - 1e-4)
                    pos_w = max(float(args.condition_depth_confidence_positive_weight), 0.0)
                    neg_w = max(float(args.condition_depth_confidence_negative_weight), 0.0)
                    weight = torch.where(
                        target_conf > 0.5,
                        prob.new_full(prob.shape, pos_w),
                        prob.new_full(prob.shape, neg_w),
                    )
                    if label_valid.any():
                        bce = Fnn.binary_cross_entropy(prob, target_conf, reduction="none")
                        condition_depth_confidence_gt_terms.append(
                            (bce[label_valid] * weight[label_valid]).sum()
                            / weight[label_valid].sum().clamp_min(1e-6)
                        )
        return conf

    def _condition_pose_depth_feature(frames: torch.Tensor,
                                      fg: torch.Tensor,
                                      depths: torch.Tensor | None,
                                      K_all: torch.Tensor,
                                      c2w_all: torch.Tensor,
                                      radius: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Build pose-head depth features.

        ``ray_frac`` preserves compatibility with the first pose-head pretrain.
        ``local`` avoids using known camera bounds and is the mode to use for a
        future generated orbit where camera poses are absent.
        """
        if depths is None:
            depth_frac = frames.new_zeros(frames.shape[0], frames.shape[1], frames.shape[2])
            depth_valid = frames.new_zeros(frames.shape[0], frames.shape[1], frames.shape[2])
            return depth_frac, depth_valid
        if args.condition_pose_depth_norm == "ray_frac":
            depth_frac, depth_valid, _ = _depth_frac_valid_factor(
                depths, fg, K_all, c2w_all, radius
            )
            return depth_frac, depth_valid
        fracs, valids = [], []
        for i in range(depths.shape[0]):
            z = depths[i].to(device=frames.device, dtype=frames.dtype)
            valid = (
                torch.isfinite(z)
                & (z > 1e-6)
                & (z < 1e5)
                & (fg[i, ..., 0] > 0.5)
            )
            if valid.any():
                vals = z[valid].to(torch.float32)
                lo = torch.quantile(vals, 0.05).to(device=z.device, dtype=z.dtype)
                hi = torch.quantile(vals, 0.95).to(device=z.device, dtype=z.dtype)
                span = (hi - lo).clamp_min(1e-6)
                frac = ((z - lo) / span).clamp(1e-4, 1.0 - 1e-4)
                frac = torch.where(valid, frac, frac.new_full(frac.shape, 1e-4))
            else:
                frac = z.new_full(z.shape, 1e-4)
            fracs.append(frac)
            valids.append(valid.to(dtype=z.dtype))
        return torch.stack(fracs), torch.stack(valids)

    def _condition_pose_features(frames: torch.Tensor,
                                 fg: torch.Tensor,
                                 depths: torch.Tensor | None,
                                 K_all: torch.Tensor,
                                 c2w_all: torch.Tensor,
                                 radius: float) -> torch.Tensor | None:
        if condition_pose_head is None:
            return None
        if depths is not None:
            depth_frac, depth_valid = _condition_pose_depth_feature(
                frames, fg, depths, K_all, c2w_all, radius
            )
        else:
            depth_frac = frames.new_zeros(frames.shape[0], frames.shape[1], frames.shape[2])
            depth_valid = frames.new_zeros(frames.shape[0], frames.shape[1], frames.shape[2])
        head_device = next(condition_pose_head.parameters()).device
        head_dtype = next(condition_pose_head.parameters()).dtype
        mask = fg.to(device=head_device, dtype=head_dtype).clamp(0.0, 1.0)
        rgb = frames.to(device=head_device, dtype=head_dtype).clamp(0.0, 1.0)
        frac = depth_frac.to(device=head_device, dtype=head_dtype).clamp(0.0, 1.0)
        valid = depth_valid.to(device=head_device, dtype=head_dtype).clamp(0.0, 1.0)
        feat = torch.cat([
            (rgb * mask).permute(0, 3, 1, 2),
            mask.permute(0, 3, 1, 2),
            frac[:, None],
            valid[:, None],
        ], dim=1)
        if args.condition_pose_detach_inputs:
            feat = feat.detach()
        return feat

    def _predict_condition_pose_raw(frames: torch.Tensor,
                                    fg: torch.Tensor,
                                    depths: torch.Tensor | None,
                                    K_all: torch.Tensor,
                                    c2w_all: torch.Tensor,
                                    radius: float) -> torch.Tensor | None:
        feat = _condition_pose_features(frames, fg, depths, K_all, c2w_all, radius)
        if feat is None:
            return None
        return condition_pose_head(feat)

    def _condition_pose_targets(c2w_all: torch.Tensor,
                                radius: float,
                                device: torch.device,
                                dtype: torch.dtype
                                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2w_h = c2w_all.to(device=device, dtype=dtype)
        center = c2w_h[:, :3, 3]
        target_center = Fnn.normalize(center, dim=-1, eps=1e-6)
        # OpenGL cameras look down local -Z, so world forward is -c2w[:, :, 2].
        target_forward = Fnn.normalize(-c2w_h[:, :3, 2], dim=-1, eps=1e-6)
        dist = torch.linalg.norm(center, dim=-1).clamp_min(1e-6)
        target_log_dist = torch.log(dist / max(float(radius), 1e-6))
        return target_center, target_forward, target_log_dist

    def _condition_pose_loss_terms(raw: torch.Tensor,
                                   c2w_all: torch.Tensor,
                                   radius: float
                                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_center = Fnn.normalize(raw[:, :3], dim=-1, eps=1e-6)
        pred_forward = Fnn.normalize(raw[:, 3:6], dim=-1, eps=1e-6)
        pred_log_dist = raw[:, 6]
        target_center, target_forward, target_log_dist = _condition_pose_targets(
            c2w_all, radius, raw.device, raw.dtype
        )

        center_loss = (1.0 - (pred_center * target_center).sum(dim=-1).clamp(-1.0, 1.0)).mean()
        forward_loss = (1.0 - (pred_forward * target_forward).sum(dim=-1).clamp(-1.0, 1.0)).mean()
        dist_loss = Fnn.huber_loss(pred_log_dist, target_log_dist, delta=0.05)
        return center_loss, forward_loss, dist_loss

    def _append_condition_pose_loss(raw: torch.Tensor | None,
                                    c2w_all: torch.Tensor,
                                    radius: float) -> None:
        if (raw is None
                or args.condition_pose_weight <= 0
                or not torch.is_grad_enabled()):
            return
        center_loss, forward_loss, dist_loss = _condition_pose_loss_terms(
            raw, c2w_all, radius
        )
        condition_pose_center_terms.append(center_loss)
        condition_pose_forward_terms.append(forward_loss)
        condition_pose_dist_terms.append(dist_loss)

    def _condition_pose_metrics(raw: torch.Tensor | None,
                                c2w_all: torch.Tensor,
                                radius: float) -> dict[str, float]:
        if raw is None:
            return {}
        pred_center = Fnn.normalize(raw[:, :3], dim=-1, eps=1e-6)
        pred_forward = Fnn.normalize(raw[:, 3:6], dim=-1, eps=1e-6)
        pred_log_dist = raw[:, 6]
        target_center, target_forward, target_log_dist = _condition_pose_targets(
            c2w_all, radius, raw.device, raw.dtype
        )
        center_dot = (pred_center * target_center).sum(dim=-1).clamp(-1.0, 1.0)
        forward_dot = (pred_forward * target_forward).sum(dim=-1).clamp(-1.0, 1.0)
        rad_to_deg = 180.0 / math.pi
        return {
            "pose_center_deg": float(torch.acos(center_dot).mean().detach() * rad_to_deg),
            "pose_forward_deg": float(torch.acos(forward_dot).mean().detach() * rad_to_deg),
            "pose_logdist_abs": float((pred_log_dist - target_log_dist).abs().mean().detach()),
        }

    def _condition_pose_raw_to_c2w(raw: torch.Tensor,
                                   radius: float) -> torch.Tensor:
        dtype, device = raw.dtype, raw.device
        center_dir = Fnn.normalize(raw[:, :3], dim=-1, eps=1e-6)
        forward = Fnn.normalize(raw[:, 3:6], dim=-1, eps=1e-6)
        dist = torch.exp(raw[:, 6]).clamp(0.05, 20.0) * max(float(radius), 1e-6)
        eye = center_dir * dist[:, None]
        up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).expand_as(forward)
        alt_up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).expand_as(forward)
        near_parallel = ((forward * up).sum(dim=-1).abs() > 0.98)[:, None]
        up = torch.where(near_parallel, alt_up, up)
        right = torch.cross(forward, up, dim=-1)
        right = Fnn.normalize(right, dim=-1, eps=1e-6)
        true_up = torch.cross(right, forward, dim=-1)
        true_up = Fnn.normalize(true_up, dim=-1, eps=1e-6)
        c2w_pred = torch.eye(4, dtype=dtype, device=device)[None].repeat(raw.shape[0], 1, 1)
        c2w_pred[:, :3, 0] = right
        c2w_pred[:, :3, 1] = true_up
        c2w_pred[:, :3, 2] = -forward
        c2w_pred[:, :3, 3] = eye
        return c2w_pred

    def _apply_condition_pose_camera(
        frames: torch.Tensor,
        fg: torch.Tensor,
        depths: torch.Tensor | None,
        K_all: torch.Tensor,
        c2w_all: torch.Tensor,
        w2c_all: torch.Tensor,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        raw = _predict_condition_pose_raw(frames, fg, depths, K_all, c2w_all, radius)
        _append_condition_pose_loss(raw, c2w_all, radius)
        metrics = _condition_pose_metrics(raw, c2w_all, radius)
        if raw is None or not args.condition_pose_use_predicted:
            return c2w_all, w2c_all, metrics
        pred_c2w = _condition_pose_raw_to_c2w(raw, radius).to(
            device=c2w_all.device, dtype=c2w_all.dtype
        )
        pred_w2c = opengl_c2w_to_opencv_w2c(pred_c2w).to(
            device=w2c_all.device, dtype=w2c_all.dtype
        )
        return pred_c2w, pred_w2c, metrics

    # --- model / optim ---
    image_cond_active = bool(args.image_condition or args.image_head_skip
                             or args.image_depth_condition or args.image_visibility_condition
                             or args.image_photo_visibility_condition
                             or args.image_confidence_condition
                             or args.image_normal_condition)
    image_cond_channels = 0
    if image_cond_active:
        image_cond_channels = 4 + (2 if args.image_depth_condition else 0) + (
            1 if (args.image_visibility_condition or args.image_photo_visibility_condition
                  or args.image_confidence_condition) else 0
        ) + (3 if args.image_normal_condition else 0)
    model = CleanGSDecoder(scale_cap_frac=args.scale_cap_frac, ups_stages=args.ups_stages,
                           mean_offset_frac=args.mean_offset_frac,
                           upsample_mode=args.upsample_mode,
                           latent_skip=bool(args.latent_skip),
                           coord_inject=bool(args.coord_inject),
                           coord_fourier=args.coord_fourier,
                           image_cond_channels=image_cond_channels,
                           image_head_skip=bool(args.image_head_skip),
                           image_opacity_fg=args.image_opacity_fg,
                           image_opacity_bg=args.image_opacity_bg,
                           image_residual_scale=args.image_residual_scale,
                           image_rgb_residual_scale=args.image_rgb_residual_scale,
                           image_opacity_residual_scale=args.image_opacity_residual_scale,
                           image_scale_frac=args.image_scale_frac,
                           image_geom_residual_scale=args.image_geom_residual_scale,
                           explicit_depth_head=bool(args.explicit_depth_head),
                           explicit_visibility_head=bool(args.explicit_visibility_head),
                           image_depth_prior_frac=args.image_depth_prior_frac,
                           image_depth_skip=bool(args.image_depth_skip),
                           image_depth_residual_scale=args.image_depth_residual_scale,
                           zero_init_head=bool(args.zero_init_head),
                           image_visibility_skip=bool(args.image_visibility_skip),
                           image_normal_scale_frac=args.image_normal_scale_frac,
                           image_boundary_scale_mult=args.image_boundary_scale_mult,
                           image_boundary_width=args.image_boundary_width,
                           image_camera_quat=bool(args.image_camera_quat),
                           image_normal_quat=bool(args.image_normal_quat),
                           depth_head_scale=args.depth_head_scale,
                           visibility_head_scale=args.visibility_head_scale,
                           latent_t=args.latent_t,
                           latent_h=args.latent_h,
                           latent_w=args.latent_w).to(dev)
    blend_head = None
    if args.anchor_render_mode == "learned_fill":
        blend_head = LearnedFillBlend(
            hidden=args.anchor_learned_fill_hidden,
            layers=args.anchor_learned_fill_layers,
            rgb_residual_scale=args.anchor_learned_fill_rgb_residual_scale,
        ).to(dev)
    elif args.anchor_render_mode == "learned_iblend_fill":
        if args.anchor_blend_topk > args.anchor_views:
            raise ValueError("--anchor_render_mode learned_iblend_fill requires --anchor_blend_topk <= --anchor_views")
        if args.anchor_learned_fill_arch == "unet":
            blend_head = LearnedIblendFillUNet(
                topk=args.anchor_blend_topk,
                hidden=args.anchor_learned_fill_hidden,
                rgb_residual_scale=args.anchor_learned_fill_rgb_residual_scale,
            ).to(dev)
        else:
            blend_head = LearnedIblendFillBlend(
                topk=args.anchor_blend_topk,
                hidden=args.anchor_learned_fill_hidden,
                layers=args.anchor_learned_fill_layers,
                rgb_residual_scale=args.anchor_learned_fill_rgb_residual_scale,
            ).to(dev)
    depth_refine_head = None
    if args.depth_refine_unet:
        if not args.image_depth_condition:
            raise ValueError("--depth_refine_unet requires --image_depth_condition")
        dref_in_channels = 6 + (4 if args.depth_refine_multiview_features else 0)
        depth_refine_head = DepthRefineUNet(
            hidden=args.depth_refine_hidden,
            in_channels=dref_in_channels,
        ).to(dev)
    support_gate_head = None
    if args.support_gate_unet:
        if not args.image_depth_condition:
            raise ValueError("--support_gate_unet requires --image_depth_condition")
        support_gate_head = SupportGateUNet(hidden=args.support_gate_hidden).to(dev)
    surface_confidence_head = None
    if args.surface_confidence_unet:
        if not args.image_depth_condition:
            raise ValueError("--surface_confidence_unet requires --image_depth_condition")
        surface_confidence_head = SurfaceConfidenceUNet(
            hidden=args.surface_confidence_hidden
        ).to(dev)
    surface_refine_head = None
    if args.surface_refine_unet:
        if not args.image_depth_condition:
            raise ValueError("--surface_refine_unet requires --image_depth_condition")
        surface_refine_head = SurfaceRefineUNet(
            hidden=args.surface_refine_hidden
        ).to(dev)
    fusion_candidate_head = None
    if args.fusion_candidate_gate:
        if args.fusion_voxel_size_frac <= 0 and not args.use_surface_token_decoder:
            raise ValueError("--fusion_candidate_gate requires --fusion_voxel_size_frac > 0")
        if not args.fusion_voxel_score_depth:
            raise ValueError("--fusion_candidate_gate requires --fusion_voxel_score_depth")
        fcand_in_channels = FUSION_CANDIDATE_FEATURE_CHANNELS + (
            FUSION_CANDIDATE_COORD_FEATURE_CHANNELS
            if args.fusion_candidate_coord_features else 0
        ) + (
            FUSION_CANDIDATE_RICH_FEATURE_CHANNELS
            if args.fusion_candidate_rich_features else 0
        ) + (
            FUSION_CANDIDATE_VOXEL_FEATURE_CHANNELS
            if args.fusion_candidate_voxel_features else 0
        ) + (
            FUSION_CANDIDATE_NEIGHBOR_FEATURE_CHANNELS
            if args.fusion_candidate_neighbor_features else 0
        )
        fusion_candidate_head = FusionCandidateGate(
            in_channels=fcand_in_channels,
            hidden=args.fusion_candidate_hidden,
            layers=args.fusion_candidate_layers,
        ).to(dev)
    output_alpha_refine_head = None
    if args.output_alpha_refine_unet:
        output_alpha_refine_head = OutputAlphaRefineUNet(
            hidden=args.output_alpha_refine_hidden
        ).to(dev)
    sparse_voxel_fusion_head = None
    if args.use_sparse_voxel_fusion and args.use_mlp_voxel_fusion:
        raise ValueError("Choose only one voxel fusion head")
    if args.use_sparse_voxel_fusion + args.use_mlp_voxel_fusion + args.use_message_voxel_fusion > 1:
        raise ValueError("Choose only one voxel fusion head")
    if args.use_sparse_voxel_fusion:
        if args.fusion_voxel_size_frac <= 0:
            raise ValueError("--use_sparse_voxel_fusion requires --fusion_voxel_size_frac > 0")
        from decoder.clean.sparse_voxel_fusion import SparseVoxelFusion
        sparse_voxel_fusion_head = SparseVoxelFusion(
            hidden=args.sparse_voxel_hidden,
            depth_res_frac=args.sparse_voxel_depth_res_frac,
            rgb_res_scale=args.sparse_voxel_rgb_res_scale,
            opacity_res_scale=args.sparse_voxel_opacity_res_scale,
            vis_delta=args.sparse_voxel_vis_delta,
            enhance_only=bool(args.sparse_voxel_enhance_only),
            target_vis_pos_min=args.sparse_voxel_target_vis_pos_min,
            target_vis_neg_max=args.sparse_voxel_target_vis_neg_max,
            target_vis_positive_weight=args.sparse_voxel_target_vis_positive_weight,
            target_vis_negative_weight=args.sparse_voxel_target_vis_negative_weight,
        ).to(dev)
    elif args.use_message_voxel_fusion:
        if args.fusion_voxel_size_frac <= 0:
            raise ValueError("--use_message_voxel_fusion requires --fusion_voxel_size_frac > 0")
        from decoder.clean.sparse_voxel_fusion import DenseVoxelMessageFusionMLP
        sparse_voxel_fusion_head = DenseVoxelMessageFusionMLP(
            hidden=args.sparse_voxel_hidden,
            layers=args.mlp_voxel_layers,
            message_radius=args.mlp_voxel_message_radius,
            depth_res_frac=args.sparse_voxel_depth_res_frac,
            rgb_res_scale=args.sparse_voxel_rgb_res_scale,
            opacity_res_scale=args.sparse_voxel_opacity_res_scale,
            vis_delta=args.sparse_voxel_vis_delta,
            enhance_only=bool(args.sparse_voxel_enhance_only),
            neighbor_radius=args.mlp_voxel_neighbor_radius,
            target_vis_pos_min=args.sparse_voxel_target_vis_pos_min,
            target_vis_neg_max=args.sparse_voxel_target_vis_neg_max,
            target_vis_positive_weight=args.sparse_voxel_target_vis_positive_weight,
            target_vis_negative_weight=args.sparse_voxel_target_vis_negative_weight,
        ).to(dev)
    elif args.use_mlp_voxel_fusion:
        if args.fusion_voxel_size_frac <= 0:
            raise ValueError("--use_mlp_voxel_fusion requires --fusion_voxel_size_frac > 0")
        from decoder.clean.sparse_voxel_fusion import DenseVoxelFusionMLP
        sparse_voxel_fusion_head = DenseVoxelFusionMLP(
            hidden=args.sparse_voxel_hidden,
            layers=args.mlp_voxel_layers,
            depth_res_frac=args.sparse_voxel_depth_res_frac,
            rgb_res_scale=args.sparse_voxel_rgb_res_scale,
            opacity_res_scale=args.sparse_voxel_opacity_res_scale,
            vis_delta=args.sparse_voxel_vis_delta,
            enhance_only=bool(args.sparse_voxel_enhance_only),
            neighbor_radius=args.mlp_voxel_neighbor_radius,
            target_vis_pos_min=args.sparse_voxel_target_vis_pos_min,
            target_vis_neg_max=args.sparse_voxel_target_vis_neg_max,
            target_vis_positive_weight=args.sparse_voxel_target_vis_positive_weight,
            target_vis_negative_weight=args.sparse_voxel_target_vis_negative_weight,
        ).to(dev)
    surface_token_decoder = None
    surface_token_view_selector = None
    canonical_voxel_decoder = None
    if args.use_surface_token_decoder and args.use_canonical_voxel_decoder:
        raise ValueError("Choose only one direct learned decoder path")
    if args.use_surface_token_decoder:
        from decoder.clean.surface_token_decoder import RGBDSurfaceTokenDecoder
        surface_token_decoder = RGBDSurfaceTokenDecoder(
            latent_channels=128,
            hidden=args.surface_token_hidden,
            slots=args.surface_token_slots,
            layers=args.surface_token_layers,
            heads=args.surface_token_heads,
            latent_layers=args.surface_token_latent_layers,
            latent_pool=args.surface_token_latent_pool,
            latent_gate_init=args.surface_token_latent_gate_init,
            slot_refine_layers=args.surface_token_slot_refine_layers,
            slot_refine_mlp_ratio=args.surface_token_slot_refine_mlp_ratio,
            slot_refine_gate_init=args.surface_token_slot_refine_gate_init,
            grid_h=args.surface_token_grid_h,
            grid_w=args.surface_token_grid_w,
            mean_res_frac=args.surface_token_mean_res_frac,
            rgb_res_scale=args.surface_token_rgb_res_scale,
            scale_frac=args.surface_token_scale_frac,
            normal_scale_frac=args.surface_token_normal_scale_frac,
            scale_res_scale=args.surface_token_scale_res_scale,
            quat_res_scale=args.surface_token_quat_res_scale,
            opacity_init=args.surface_token_opacity_init,
            checkpoint_blocks=bool(args.surface_token_checkpoint_blocks),
            detail_layer=args.surface_token_detail_layer,
            detail_mean_res_frac=args.surface_token_detail_mean_res_frac,
            detail_rgb_res_scale=args.surface_token_detail_rgb_res_scale,
            detail_scale_frac=args.surface_token_detail_scale_frac,
            detail_normal_scale_frac=args.surface_token_detail_normal_scale_frac,
            detail_scale_res_scale=args.surface_token_detail_scale_res_scale,
            detail_quat_res_scale=args.surface_token_detail_quat_res_scale,
            detail_opacity_init=args.surface_token_detail_opacity_init,
            source_rgb_dropout_prob=args.surface_token_source_rgb_dropout_prob,
            learned_scale_base=bool(args.surface_token_learned_scale_base),
            learned_scale_head=bool(args.surface_token_learned_scale_head),
            learned_scale_min_frac=args.surface_token_learned_scale_min_frac,
            learned_scale_max_frac=args.surface_token_learned_scale_max_frac,
            learned_opacity_bias=bool(args.surface_token_learned_opacity_bias),
            learned_opacity_prior=bool(args.surface_token_learned_opacity_prior),
            learned_output_scales=bool(args.surface_token_learned_output_scales),
            learned_color_affine=bool(args.surface_token_learned_color_affine),
            color_affine_scale=args.surface_token_color_affine_scale,
            learned_policy_head=bool(args.surface_token_learned_policy_head),
            policy_depth_res_frac=args.surface_token_policy_depth_res_frac,
            policy_move_res_frac=args.surface_token_policy_move_res_frac,
            policy_scale_res_scale=args.surface_token_policy_scale_res_scale,
            policy_opacity_res_scale=args.surface_token_policy_opacity_res_scale,
            policy_view_res_scale=args.surface_token_policy_view_res_scale,
            policy_confidence_res_scale=args.surface_token_policy_confidence_res_scale,
            policy_keep_res_scale=args.surface_token_policy_keep_res_scale,
            policy_coverage_scale_res_scale=args.surface_token_policy_coverage_scale_res_scale,
            policy_birth_res_scale=args.surface_token_policy_birth_res_scale,
            learned_policy_output_scales=bool(args.surface_token_learned_policy_output_scales),
            learned_source_depth_confidence_head=bool(
                args.surface_token_learned_source_depth_confidence_head
            ),
            source_depth_res_frac=args.surface_token_source_depth_res_frac,
            source_confidence_res_scale=args.surface_token_source_confidence_res_scale,
            learned_source_depth_confidence_scales=bool(
                args.surface_token_learned_source_depth_confidence_scales
            ),
            proposal_count=args.surface_token_proposal_count,
            proposal_scale_frac=args.surface_token_proposal_scale_frac,
            proposal_normal_scale_frac=args.surface_token_proposal_normal_scale_frac,
            proposal_scale_res_scale=args.surface_token_proposal_scale_res_scale,
            proposal_quat_res_scale=args.surface_token_proposal_quat_res_scale,
            proposal_rgb_res_scale=args.surface_token_proposal_rgb_res_scale,
            proposal_extent_frac=args.surface_token_proposal_extent_frac,
            proposal_coverage_scale_res_scale=args.surface_token_proposal_coverage_scale_res_scale,
            proposal_opacity_init=args.surface_token_proposal_opacity_init,
            proposal_seed_surface=bool(args.surface_token_proposal_seed_surface),
            proposal_seed_pool=args.surface_token_proposal_seed_pool,
            proposal_surface_res_frac=args.surface_token_proposal_surface_res_frac,
            proposal_anchor_mode=args.surface_token_proposal_anchor_mode,
            proposal_anchor_temp=args.surface_token_proposal_anchor_temp,
            proposal_anchor_local_window=args.surface_token_proposal_anchor_local_window,
            proposal_anchor_gate_init=args.surface_token_proposal_anchor_gate_init,
            proposal_anchor_mix_res_scale=args.surface_token_proposal_anchor_mix_res_scale,
            proposal_anchor_even_prior=args.surface_token_proposal_anchor_even_prior,
            learned_proposal_policy_head=bool(args.surface_token_learned_proposal_policy_head),
            proposal_policy_keep_res_scale=args.surface_token_proposal_policy_keep_res_scale,
            proposal_policy_confidence_res_scale=args.surface_token_proposal_policy_confidence_res_scale,
            proposal_policy_coverage_res_scale=args.surface_token_proposal_policy_coverage_res_scale,
            learned_proposal_scale_base=bool(args.surface_token_learned_proposal_scale_base),
            learned_proposal_scale_head=bool(args.surface_token_learned_proposal_scale_head),
            learned_proposal_scale_min_frac=args.surface_token_learned_proposal_scale_min_frac,
            learned_proposal_scale_max_frac=args.surface_token_learned_proposal_scale_max_frac,
            depth_normal_quat=bool(args.surface_token_depth_normal_quat or args.image_normal_quat),
            depth_normal_blend=args.surface_token_depth_normal_blend,
            learned_depth_normal_blend=bool(args.surface_token_learned_depth_normal_blend),
            learned_depth_normal_blend_head=bool(args.surface_token_learned_depth_normal_blend_head),
            depth_normal_blend_head_scale=args.surface_token_depth_normal_blend_head_scale,
        ).to(dev)
        if args.surface_token_learned_view_selector:
            from decoder.clean.surface_token_decoder import SurfaceTokenViewSelector
            surface_token_view_selector = SurfaceTokenViewSelector(
                latent_channels=128,
                hidden=args.surface_token_view_selector_hidden,
                score_scale=args.surface_token_view_selector_score_scale,
                gate_scale=args.surface_token_view_selector_gate_scale,
            ).to(dev)
    if args.use_canonical_voxel_decoder:
        from decoder.clean.canonical_voxel_decoder import CanonicalVoxelDecoder
        canonical_voxel_decoder = CanonicalVoxelDecoder(
            latent_channels=128,
            hidden=args.canonical_voxel_hidden,
            layers=args.canonical_voxel_layers,
            heads=args.canonical_voxel_heads,
            latent_layers=args.canonical_voxel_latent_layers,
            scene_slots=args.canonical_voxel_scene_slots,
            grid_h=args.canonical_voxel_grid_h,
            grid_w=args.canonical_voxel_grid_w,
            latent_pool=args.canonical_voxel_latent_pool,
            message_radius=args.canonical_voxel_message_radius,
            voxel_size_frac=args.canonical_voxel_size_frac,
            max_voxels=args.canonical_voxel_max_voxels,
            gaussians_per_voxel=args.canonical_voxel_gaussians_per_voxel,
            child_offset_mult=args.canonical_voxel_child_offset_mult,
            mean_res_voxels=args.canonical_voxel_mean_res_voxels,
            rgb_res_scale=args.canonical_voxel_rgb_res_scale,
            tangent_scale_mult=args.canonical_voxel_tangent_scale_mult,
            normal_scale_mult=args.canonical_voxel_normal_scale_mult,
            scale_res_scale=args.canonical_voxel_scale_res_scale,
            quat_res_scale=args.canonical_voxel_quat_res_scale,
            opacity_init=args.canonical_voxel_opacity_init,
            opacity_support_floor=args.canonical_voxel_opacity_support_floor,
            opacity_support_target=args.canonical_voxel_opacity_support_target,
            detail_sampling=bool(args.canonical_voxel_detail_sampling),
            detail_color_mix=args.canonical_voxel_detail_color_mix,
            detail_depth_tol_frac=args.canonical_voxel_detail_depth_tol_frac,
            detail_score_temp=args.canonical_voxel_detail_score_temp,
            detail_chunk=args.canonical_voxel_detail_chunk,
            view_feature_channels=args.canonical_voxel_view_feature_channels,
            view_feature_scale=args.canonical_voxel_view_feature_scale,
            opacity_prior_weight=args.canonical_voxel_opacity_prior_weight,
            zero_init_head=bool(args.canonical_voxel_zero_init_head),
            source_consistency_refine=bool(args.canonical_source_vis_learned_refine),
            source_consistency_hidden=args.canonical_source_vis_refine_hidden,
            source_consistency_opacity_strength=args.canonical_source_vis_refine_opacity_strength,
            source_consistency_rgb_scale=args.canonical_source_vis_refine_rgb_scale,
            source_consistency_scale_res_scale=args.canonical_source_vis_refine_scale_res_scale,
            source_consistency_zero_init=bool(args.canonical_source_vis_refine_zero_init),
        ).to(dev)
    if args.freeze_decoder:
        for p_m in model.parameters():
            p_m.requires_grad_(False)
        model.eval()
    if surface_token_decoder is not None and args.surface_token_train_rgb_head_only:
        for p_m in surface_token_decoder.parameters():
            p_m.requires_grad_(False)
        rgb_head = surface_token_decoder.head[-1]
        if not isinstance(rgb_head, torch.nn.Linear) or rgb_head.out_features != 14:
            raise ValueError("--surface_token_train_rgb_head_only expected a 14-row final Linear head")
        rgb_head.weight.requires_grad_(True)
        rgb_head.bias.requires_grad_(True)
        row_mask = torch.zeros_like(rgb_head.bias)
        row_mask[3:6] = 1.0
        rgb_head.weight.register_hook(
            lambda grad, mask=row_mask.reshape(-1, 1): grad * mask.to(device=grad.device, dtype=grad.dtype)
        )
        rgb_head.bias.register_hook(
            lambda grad, mask=row_mask: grad * mask.to(device=grad.device, dtype=grad.dtype)
        )
        print("[phase2] surface-token RGB-head-only training enabled", flush=True)
    if surface_token_decoder is not None and args.surface_token_train_new_capacity_only:
        new_prefixes = (
            "latent_token_proj.",
            "latent_blocks.",
            "slot_refine_blocks.",
            "opacity_prior.",
            "color_affine.",
            "policy_head.",
            "source_depth_confidence_head.",
            "depth_normal_blend_head.",
            "detail_head.",
            "proposal_",
        )
        new_names = {
            "latent_gate_logit",
            "slot_refine_gate_logit",
            "log_scale_base",
            "opacity_logit_bias",
            "log_output_scales",
            "policy_log_output_scales",
            "source_depth_confidence_log_scales",
            "depth_normal_blend_logit",
        }
        train_new = 0
        total_new = 0
        for name, p_m in surface_token_decoder.named_parameters():
            is_new = name in new_names or any(name.startswith(prefix) for prefix in new_prefixes)
            p_m.requires_grad_(is_new)
            total_new += int(p_m.numel()) if is_new else 0
            train_new += int(p_m.numel()) if is_new and p_m.requires_grad else 0
        print(
            "[phase2] surface-token new-capacity-only training enabled "
            f"trainable={train_new:,} matched={total_new:,}",
            flush=True,
        )
    if surface_token_decoder is not None and args.surface_token_train_policy_heads_only:
        policy_prefixes = (
            "head.",
            "detail_head.",
            "opacity_prior.",
            "color_affine.",
            "policy_head.",
            "source_depth_confidence_head.",
            "depth_normal_blend_head.",
            "proposal_",
        )
        policy_names = {
            "log_scale_base",
            "opacity_logit_bias",
            "log_output_scales",
            "policy_log_output_scales",
            "source_depth_confidence_log_scales",
            "depth_normal_blend_logit",
            "proposal_log_scale_base",
            "proposal_opacity_logit_bias",
        }
        train_policy = 0
        total_policy = 0
        for name, p_m in surface_token_decoder.named_parameters():
            is_policy = name in policy_names or any(
                name.startswith(prefix) for prefix in policy_prefixes
            )
            p_m.requires_grad_(is_policy)
            total_policy += int(p_m.numel()) if is_policy else 0
            train_policy += int(p_m.numel()) if is_policy and p_m.requires_grad else 0
        print(
            "[phase2] surface-token policy/head-only training enabled "
            f"trainable={train_policy:,} matched={total_policy:,}",
            flush=True,
        )
    if surface_token_decoder is not None and args.surface_token_train_detail_only:
        detail_head = getattr(surface_token_decoder, "detail_head", None)
        if detail_head is None:
            raise ValueError("--surface_token_train_detail_only requires --surface_token_detail_layer 1")
        for p_m in surface_token_decoder.parameters():
            p_m.requires_grad_(False)
        for p_m in detail_head.parameters():
            p_m.requires_grad_(True)
        print("[phase2] surface-token detail-head-only training enabled", flush=True)
    if depth_refine_head is not None and args.freeze_depth_refine_head:
        for p_m in depth_refine_head.parameters():
            p_m.requires_grad_(False)
        depth_refine_head.eval()
    percep = VGGPerceptual().to(dev) if args.perceptual_weight > 0 else None
    adaptive_loss = None
    if args.adaptive_loss_weights:
        names = [n.strip() for n in args.adaptive_loss_names.split(",") if n.strip()]
        adaptive_loss = AdaptiveLossBalancer(
            names,
            logvar_min=args.adaptive_loss_logvar_min,
            logvar_max=args.adaptive_loss_logvar_max,
        ).to(dev)
        print(f"[phase2] adaptive loss weights enabled: {','.join(names)}", flush=True)

    def _adapt_has(name: str) -> bool:
        return adaptive_loss is not None and adaptive_loss.has(name)

    def _loss_weighted(name: str, raw: torch.Tensor, fixed_weight: float) -> torch.Tensor:
        if adaptive_loss is not None and adaptive_loss.has(name):
            return adaptive_loss(name, raw, fixed_weight)
        if fixed_weight <= 0:
            return raw.new_zeros(())
        return float(fixed_weight) * raw

    amp_enabled = args.train_precision == "bf16" and dev == "cuda"
    if args.train_precision == "bf16" and not amp_enabled:
        print("[phase2] bf16 requested but CUDA is unavailable; using fp32", flush=True)

    def _train_precision_context():
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        )

    def _renderable_params(params: dict) -> dict:
        if not amp_enabled:
            return params
        out = {}
        for key, val in params.items():
            if torch.is_tensor(val) and torch.is_floating_point(val):
                out[key] = val.float()
            else:
                out[key] = val
        return out

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if blend_head is not None:
        trainable_params += list(blend_head.parameters())
    if depth_refine_head is not None:
        trainable_params += [p for p in depth_refine_head.parameters() if p.requires_grad]
    if support_gate_head is not None:
        trainable_params += list(support_gate_head.parameters())
    if surface_confidence_head is not None:
        trainable_params += list(surface_confidence_head.parameters())
    if surface_refine_head is not None:
        trainable_params += list(surface_refine_head.parameters())
    if fusion_candidate_head is not None:
        trainable_params += list(fusion_candidate_head.parameters())
    if condition_mask_refine_head is not None:
        trainable_params += list(condition_mask_refine_head.parameters())
    if condition_rgb_refine_head is not None:
        trainable_params += list(condition_rgb_refine_head.parameters())
    if condition_rgbd_refine_head is not None:
        trainable_params += list(condition_rgbd_refine_head.parameters())
    if condition_pose_head is not None and not args.freeze_condition_pose_head:
        trainable_params += list(condition_pose_head.parameters())
    if condition_depth_affine_head is not None:
        trainable_params += list(condition_depth_affine_head.parameters())
    if condition_depth_confidence_head is not None:
        trainable_params += list(condition_depth_confidence_head.parameters())
    if output_alpha_refine_head is not None:
        trainable_params += list(output_alpha_refine_head.parameters())
    if sparse_voxel_fusion_head is not None:
        trainable_params += list(sparse_voxel_fusion_head.parameters())
    if surface_token_decoder is not None:
        trainable_params += [p for p in surface_token_decoder.parameters() if p.requires_grad]
    if surface_token_view_selector is not None:
        trainable_params += [
            p for p in surface_token_view_selector.parameters() if p.requires_grad
        ]
    if canonical_voxel_decoder is not None:
        trainable_params += list(canonical_voxel_decoder.parameters())
    if adaptive_loss is not None:
        trainable_params += list(adaptive_loss.parameters())
    if not trainable_params and not args.eval_only:
        raise ValueError(
            "no trainable parameters; disable --freeze_decoder or enable a learned auxiliary head"
        )
    opt = None
    sched = None
    opt_params = trainable_params
    proposal_lr_mult = max(float(args.surface_token_proposal_lr_mult), 0.0)
    proposal_policy_lr_mult = max(float(args.surface_token_proposal_policy_lr_mult), 0.0)
    policy_lr_mult = max(float(args.surface_token_policy_lr_mult), 0.0)
    proposal_opacity_lr_mult = max(float(args.surface_token_proposal_opacity_lr_mult), 0.0)
    depth_normal_blend_lr_mult = max(
        float(args.surface_token_depth_normal_blend_lr_mult), 0.0
    )
    if (surface_token_decoder is not None
            and trainable_params
            and (proposal_lr_mult != 1.0
                 or proposal_policy_lr_mult != 1.0
                 or policy_lr_mult != 1.0
                 or proposal_opacity_lr_mult != 1.0
                 or depth_normal_blend_lr_mult != 1.0)):
        proposal_ids: set[int] = set()
        proposal_policy_ids: set[int] = set()
        policy_ids: set[int] = set()
        proposal_opacity_ids: set[int] = set()
        depth_normal_blend_ids: set[int] = set()
        proposal_params: list[torch.nn.Parameter] = []
        proposal_policy_params: list[torch.nn.Parameter] = []
        policy_params: list[torch.nn.Parameter] = []
        proposal_opacity_params: list[torch.nn.Parameter] = []
        depth_normal_blend_params: list[torch.nn.Parameter] = []
        for name, p_m in surface_token_decoder.named_parameters():
            if not p_m.requires_grad:
                continue
            if (name.startswith("policy_head.")
                    or name.startswith("source_depth_confidence_head.")
                    or name == "policy_log_output_scales"
                    or name == "source_depth_confidence_log_scales"):
                policy_ids.add(id(p_m))
                policy_params.append(p_m)
            elif name == "depth_normal_blend_logit" or name.startswith(
                "depth_normal_blend_head."
            ):
                depth_normal_blend_ids.add(id(p_m))
                depth_normal_blend_params.append(p_m)
            elif "proposal" not in name:
                continue
            elif name == "proposal_opacity_logit_bias":
                proposal_opacity_ids.add(id(p_m))
                proposal_opacity_params.append(p_m)
            elif name.startswith("proposal_policy_head."):
                proposal_policy_ids.add(id(p_m))
                proposal_policy_params.append(p_m)
            else:
                proposal_ids.add(id(p_m))
                proposal_params.append(p_m)
        special_ids = (
            proposal_ids
            | proposal_policy_ids
            | policy_ids
            | proposal_opacity_ids
            | depth_normal_blend_ids
        )
        base_params = [p_m for p_m in trainable_params if id(p_m) not in special_ids]
        opt_params = [{"params": base_params, "lr": args.lr, "weight_decay": 0.05}]
        if proposal_params:
            opt_params.append({
                "params": proposal_params,
                "lr": args.lr * proposal_lr_mult,
                "weight_decay": 0.05,
            })
        if proposal_policy_params:
            opt_params.append({
                "params": proposal_policy_params,
                "lr": args.lr * proposal_policy_lr_mult,
                "weight_decay": 0.05,
            })
        if policy_params:
            opt_params.append({
                "params": policy_params,
                "lr": args.lr * policy_lr_mult,
                "weight_decay": 0.05,
            })
        if proposal_opacity_params:
            opt_params.append({
                "params": proposal_opacity_params,
                "lr": args.lr * proposal_opacity_lr_mult,
                "weight_decay": 0.0,
            })
        if depth_normal_blend_params:
            opt_params.append({
                "params": depth_normal_blend_params,
                "lr": args.lr * depth_normal_blend_lr_mult,
                "weight_decay": 0.0,
            })
        print(
            "[phase2] proposal optimizer groups "
            f"base={sum(p.numel() for p in base_params):,} "
            f"proposal={sum(p.numel() for p in proposal_params):,}x{proposal_lr_mult:g} "
            f"proposal_policy={sum(p.numel() for p in proposal_policy_params):,}x{proposal_policy_lr_mult:g} "
            f"policy={sum(p.numel() for p in policy_params):,}x{policy_lr_mult:g} "
            f"proposal_opacity={sum(p.numel() for p in proposal_opacity_params):,}x{proposal_opacity_lr_mult:g} "
            f"depth_normal_blend={sum(p.numel() for p in depth_normal_blend_params):,}x{depth_normal_blend_lr_mult:g}",
            flush=True,
        )
    if trainable_params:
        if args.optimizer == "adafactor":
            opt = torch.optim.Adafactor(opt_params, lr=args.lr, weight_decay=0.05)
        else:
            opt = torch.optim.AdamW(
                opt_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05
            )
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=args.warmup
        )
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(args.steps - args.warmup, 1)
        )
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup])
    start_step = 0

    def _load_condition_pose_state(state: dict, source: Path | str) -> None:
        if condition_pose_head is None:
            return
        state = dict(state)
        if "view_pos" in state and state["view_pos"].shape != condition_pose_head.view_pos.shape:
            src = state["view_pos"].to(device=condition_pose_head.view_pos.device,
                                       dtype=condition_pose_head.view_pos.dtype)
            dst = condition_pose_head.view_pos.detach().clone()
            n = min(src.shape[1], dst.shape[1])
            dst[:, :n] = src[:, :n]
            state["view_pos"] = dst
            print(
                f"[phase2] adapted condition_pose_head view_pos from "
                f"{tuple(src.shape)} to {tuple(dst.shape)} while loading {source}",
                flush=True,
            )
        condition_pose_head.load_state_dict(state)

    def _load_phase2_checkpoint(ckpt_path: Path, *, lean: bool) -> dict:
        """Load a checkpoint without hydrating optimizer state for reset/eval probes.

        Surface-token checkpoints can carry multi-GB Adam state.  For eval-only
        runs and reset-on-resume continuation probes, those tensors are never
        used; loading them directly onto the GPU wastes VRAM and can pressure
        Docker's RAM limit before the first useful forward pass.
        """
        kwargs = {
            "map_location": "cpu" if lean else dev,
            "weights_only": False,
        }
        if lean:
            kwargs["mmap"] = True
        try:
            ckpt = torch.load(ckpt_path, **kwargs)
        except TypeError:
            kwargs.pop("mmap", None)
            ckpt = torch.load(ckpt_path, **kwargs)
        if lean:
            ckpt.pop("optimizer", None)
            ckpt.pop("scheduler", None)
        return ckpt

    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        lean_resume = bool(args.reset_optimizer_on_resume or args.eval_only)
        ckpt = _load_phase2_checkpoint(ckpt_path, lean=lean_resume)
        model.load_state_dict(ckpt["model"])
        if blend_head is not None and ckpt.get("blend_head") is not None:
            blend_head.load_state_dict(ckpt["blend_head"])
        if depth_refine_head is not None and ckpt.get("depth_refine_head") is not None:
            depth_refine_head.load_state_dict(ckpt["depth_refine_head"])
        if support_gate_head is not None and ckpt.get("support_gate_head") is not None:
            support_gate_head.load_state_dict(ckpt["support_gate_head"])
        if (surface_confidence_head is not None
                and ckpt.get("surface_confidence_head") is not None):
            surface_confidence_head.load_state_dict(ckpt["surface_confidence_head"])
        if (surface_refine_head is not None
                and ckpt.get("surface_refine_head") is not None):
            surface_refine_head.load_state_dict(ckpt["surface_refine_head"])
        if (fusion_candidate_head is not None
                and ckpt.get("fusion_candidate_head") is not None):
            fusion_candidate_head.load_state_dict(ckpt["fusion_candidate_head"])
        if (condition_rgb_refine_head is not None
                and ckpt.get("condition_rgb_refine_head") is not None):
            condition_rgb_refine_head.load_state_dict(ckpt["condition_rgb_refine_head"])
        if (condition_rgbd_refine_head is not None
                and ckpt.get("condition_rgbd_refine_head") is not None):
            condition_rgbd_refine_head.load_state_dict(ckpt["condition_rgbd_refine_head"])
        if (condition_pose_head is not None
                and ckpt.get("condition_pose_head") is not None):
            _load_condition_pose_state(ckpt["condition_pose_head"], ckpt_path)
        if (condition_depth_affine_head is not None
                and ckpt.get("condition_depth_affine_head") is not None):
            condition_depth_affine_head.load_state_dict(ckpt["condition_depth_affine_head"])
        if (condition_depth_confidence_head is not None
                and ckpt.get("condition_depth_confidence_head") is not None):
            condition_depth_confidence_head.load_state_dict(ckpt["condition_depth_confidence_head"])
        if (condition_mask_refine_head is not None
                and ckpt.get("condition_mask_refine_head") is not None):
            condition_mask_refine_head.load_state_dict(ckpt["condition_mask_refine_head"])
        if (output_alpha_refine_head is not None
                and ckpt.get("output_alpha_refine_head") is not None):
            output_alpha_refine_head.load_state_dict(ckpt["output_alpha_refine_head"])
        if (sparse_voxel_fusion_head is not None
                and ckpt.get("sparse_voxel_fusion_head") is not None):
            sparse_voxel_fusion_head.load_state_dict(ckpt["sparse_voxel_fusion_head"])
        if (surface_token_decoder is not None
                and ckpt.get("surface_token_decoder") is not None):
            missing, unexpected = surface_token_decoder.load_state_dict(
                ckpt["surface_token_decoder"], strict=False
            )
            if missing or unexpected:
                print(
                    "[phase2] surface_token_decoder partial load "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
        if (surface_token_view_selector is not None
                and ckpt.get("surface_token_view_selector") is not None):
            missing, unexpected = surface_token_view_selector.load_state_dict(
                ckpt["surface_token_view_selector"], strict=False
            )
            if missing or unexpected:
                print(
                    "[phase2] surface_token_view_selector partial load "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
        if (canonical_voxel_decoder is not None
                and ckpt.get("canonical_voxel_decoder") is not None):
            missing, unexpected = canonical_voxel_decoder.load_state_dict(
                ckpt["canonical_voxel_decoder"], strict=False
            )
            if missing or unexpected:
                print(
                    "[phase2] canonical_voxel_decoder partial load "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
        if adaptive_loss is not None and ckpt.get("adaptive_loss") is not None:
            missing, unexpected = adaptive_loss.load_state_dict(
                ckpt["adaptive_loss"], strict=False
            )
            if missing or unexpected:
                print(
                    "[phase2] adaptive_loss partial load "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
        if (opt is not None
                and not args.reset_optimizer_on_resume
                and ckpt.get("optimizer") is not None):
            opt.load_state_dict(ckpt["optimizer"])
        if (sched is not None
                and not args.reset_optimizer_on_resume
                and ckpt.get("scheduler") is not None):
            sched.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", -1)) + 1
        if args.reset_optimizer_on_resume:
            print("[phase2] reset optimizer/scheduler state after resume", flush=True)
        if lean_resume:
            print("[phase2] lean checkpoint load skipped optimizer/scheduler tensors", flush=True)
        print(f"[phase2] resumed from {ckpt_path} at step {start_step}", flush=True)
        del ckpt
        gc.collect()
    if aux_ckpt_paths:
        loaded = []
        for ckpt_path in aux_ckpt_paths:
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
            if blend_head is not None and ckpt.get("blend_head") is not None:
                blend_head.load_state_dict(ckpt["blend_head"])
                loaded.append(f"{ckpt_path}:blend_head")
            if depth_refine_head is not None and ckpt.get("depth_refine_head") is not None:
                depth_refine_head.load_state_dict(ckpt["depth_refine_head"])
                loaded.append(f"{ckpt_path}:depth_refine_head")
            if support_gate_head is not None and ckpt.get("support_gate_head") is not None:
                support_gate_head.load_state_dict(ckpt["support_gate_head"])
                loaded.append(f"{ckpt_path}:support_gate_head")
            if (surface_confidence_head is not None
                    and ckpt.get("surface_confidence_head") is not None):
                surface_confidence_head.load_state_dict(ckpt["surface_confidence_head"])
                loaded.append(f"{ckpt_path}:surface_confidence_head")
            if (surface_refine_head is not None
                    and ckpt.get("surface_refine_head") is not None):
                surface_refine_head.load_state_dict(ckpt["surface_refine_head"])
                loaded.append(f"{ckpt_path}:surface_refine_head")
            if (fusion_candidate_head is not None
                    and ckpt.get("fusion_candidate_head") is not None):
                fusion_candidate_head.load_state_dict(ckpt["fusion_candidate_head"])
                loaded.append(f"{ckpt_path}:fusion_candidate_head")
            if (condition_rgb_refine_head is not None
                    and ckpt.get("condition_rgb_refine_head") is not None):
                condition_rgb_refine_head.load_state_dict(ckpt["condition_rgb_refine_head"])
                loaded.append(f"{ckpt_path}:condition_rgb_refine_head")
            if (condition_rgbd_refine_head is not None
                    and ckpt.get("condition_rgbd_refine_head") is not None):
                condition_rgbd_refine_head.load_state_dict(ckpt["condition_rgbd_refine_head"])
                loaded.append(f"{ckpt_path}:condition_rgbd_refine_head")
            if (condition_pose_head is not None
                    and ckpt.get("condition_pose_head") is not None):
                _load_condition_pose_state(ckpt["condition_pose_head"], ckpt_path)
                loaded.append(f"{ckpt_path}:condition_pose_head")
            if (condition_depth_affine_head is not None
                    and ckpt.get("condition_depth_affine_head") is not None):
                condition_depth_affine_head.load_state_dict(
                    ckpt["condition_depth_affine_head"]
                )
                loaded.append(f"{ckpt_path}:condition_depth_affine_head")
            if (condition_depth_confidence_head is not None
                    and ckpt.get("condition_depth_confidence_head") is not None):
                condition_depth_confidence_head.load_state_dict(
                    ckpt["condition_depth_confidence_head"]
                )
                loaded.append(f"{ckpt_path}:condition_depth_confidence_head")
            if (condition_mask_refine_head is not None
                    and ckpt.get("condition_mask_refine_head") is not None):
                condition_mask_refine_head.load_state_dict(ckpt["condition_mask_refine_head"])
                loaded.append(f"{ckpt_path}:condition_mask_refine_head")
            if (output_alpha_refine_head is not None
                    and ckpt.get("output_alpha_refine_head") is not None):
                output_alpha_refine_head.load_state_dict(ckpt["output_alpha_refine_head"])
                loaded.append(f"{ckpt_path}:output_alpha_refine_head")
            if (surface_token_decoder is not None
                    and ckpt.get("surface_token_decoder") is not None):
                missing, unexpected = surface_token_decoder.load_state_dict(
                    ckpt["surface_token_decoder"], strict=False
                )
                if missing or unexpected:
                    print(
                        "[phase2] surface_token_decoder auxiliary partial load "
                        f"missing={len(missing)} unexpected={len(unexpected)}",
                        flush=True,
                    )
                loaded.append(f"{ckpt_path}:surface_token_decoder")
            if (surface_token_view_selector is not None
                    and ckpt.get("surface_token_view_selector") is not None):
                missing, unexpected = surface_token_view_selector.load_state_dict(
                    ckpt["surface_token_view_selector"], strict=False
                )
                if missing or unexpected:
                    print(
                        "[phase2] surface_token_view_selector auxiliary partial load "
                        f"missing={len(missing)} unexpected={len(unexpected)}",
                        flush=True,
                    )
                loaded.append(f"{ckpt_path}:surface_token_view_selector")
            if (canonical_voxel_decoder is not None
                    and ckpt.get("canonical_voxel_decoder") is not None):
                missing, unexpected = canonical_voxel_decoder.load_state_dict(
                    ckpt["canonical_voxel_decoder"], strict=False
                )
                if missing or unexpected:
                    print(
                        "[phase2] canonical_voxel_decoder auxiliary partial load "
                        f"missing={len(missing)} unexpected={len(unexpected)}",
                        flush=True,
                    )
                loaded.append(f"{ckpt_path}:canonical_voxel_decoder")
        if not loaded:
            raise ValueError(
                "--resume_aux_from found no compatible auxiliary heads in "
                f"{','.join(str(p) for p in aux_ckpt_paths)}"
            )
        print(f"[phase2] loaded auxiliary heads: {','.join(loaded)}", flush=True)
    model_params = sum(p.numel() for p in model.parameters())
    blend_params = sum(p.numel() for p in blend_head.parameters()) if blend_head is not None else 0
    depth_refine_params = (
        sum(p.numel() for p in depth_refine_head.parameters())
        if depth_refine_head is not None else 0
    )
    support_gate_params = (
        sum(p.numel() for p in support_gate_head.parameters())
        if support_gate_head is not None else 0
    )
    surface_confidence_params = (
        sum(p.numel() for p in surface_confidence_head.parameters())
        if surface_confidence_head is not None else 0
    )
    surface_refine_params = (
        sum(p.numel() for p in surface_refine_head.parameters())
        if surface_refine_head is not None else 0
    )
    fusion_candidate_params = (
        sum(p.numel() for p in fusion_candidate_head.parameters())
        if fusion_candidate_head is not None else 0
    )
    rgb_refine_params = (
        sum(p.numel() for p in condition_rgb_refine_head.parameters())
        if condition_rgb_refine_head is not None else 0
    )
    rgbd_refine_params = (
        sum(p.numel() for p in condition_rgbd_refine_head.parameters())
        if condition_rgbd_refine_head is not None else 0
    )
    pose_params = (
        sum(p.numel() for p in condition_pose_head.parameters())
        if condition_pose_head is not None else 0
    )
    depth_affine_params = (
        sum(p.numel() for p in condition_depth_affine_head.parameters())
        if condition_depth_affine_head is not None else 0
    )
    depth_confidence_params = (
        sum(p.numel() for p in condition_depth_confidence_head.parameters())
        if condition_depth_confidence_head is not None else 0
    )
    mask_refine_params = (
        sum(p.numel() for p in condition_mask_refine_head.parameters())
        if condition_mask_refine_head is not None else 0
    )
    output_alpha_refine_params = (
        sum(p.numel() for p in output_alpha_refine_head.parameters())
        if output_alpha_refine_head is not None else 0
    )
    surface_token_params = (
        sum(p.numel() for p in surface_token_decoder.parameters())
        if surface_token_decoder is not None else 0
    )
    surface_token_view_selector_params = (
        sum(p.numel() for p in surface_token_view_selector.parameters())
        if surface_token_view_selector is not None else 0
    )
    canonical_voxel_params = (
        sum(p.numel() for p in canonical_voxel_decoder.parameters())
        if canonical_voxel_decoder is not None else 0
    )
    adaptive_loss_params = (
        sum(p.numel() for p in adaptive_loss.parameters())
        if adaptive_loss is not None else 0
    )
    n_params = (
        model_params + blend_params + depth_refine_params + support_gate_params
        + surface_confidence_params + surface_refine_params + fusion_candidate_params
        + mask_refine_params + rgb_refine_params + rgbd_refine_params
        + pose_params + depth_affine_params + depth_confidence_params
        + output_alpha_refine_params + surface_token_params + canonical_voxel_params
        + surface_token_view_selector_params + adaptive_loss_params
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    run = _init_wandb(args, n_params, len(train_ds), model.map_h * model.map_w)
    torch.cuda.reset_peak_memory_stats()

    def _mem_stats():
        import resource
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "cuda_alloc_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "cuda_reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
            "cuda_peak_alloc_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "cuda_free_gb": free_b / (1024 ** 3),
            "cuda_total_gb": total_b / (1024 ** 3),
            "rss_max_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
        }

    def _mem_msg(stats):
        return (
            f"mem=peak{stats['cuda_peak_alloc_gb']:.1f}G "
            f"res{stats['cuda_reserved_gb']:.1f}G "
            f"free{stats['cuda_free_gb']:.1f}/{stats['cuda_total_gb']:.0f}G "
            f"rss{stats['rss_max_gb']:.1f}G"
        )

    def _depth_weight(base: float, step: int) -> float:
        if base <= 0:
            return 0.0
        if args.depth_anneal_start < 0 or args.depth_anneal_end <= args.depth_anneal_start:
            return base
        if step <= args.depth_anneal_start:
            return base
        if step >= args.depth_anneal_end:
            return base * args.depth_anneal_final_mult
        t = (step - args.depth_anneal_start) / (args.depth_anneal_end - args.depth_anneal_start)
        mult = 1.0 + t * (args.depth_anneal_final_mult - 1.0)
        return base * mult

    def _ramped_weight(base: float, start: int, ramp: int, step: int) -> float:
        if base <= 0 or step < start:
            return 0.0
        if ramp <= 0:
            return base
        return base * min(1.0, (step - start) / ramp)

    def _fg_gradient_loss(render, target, fg):
        fg_b = fg.bool()
        gx_r = render[:, :, 1:, :] - render[:, :, :-1, :]
        gx_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        mx = fg_b[:, :, 1:, :] & fg_b[:, :, :-1, :]
        gy_r = render[:, 1:, :, :] - render[:, :-1, :, :]
        gy_t = target[:, 1:, :, :] - target[:, :-1, :, :]
        my = fg_b[:, 1:, :, :] & fg_b[:, :-1, :, :]
        loss = render.new_zeros(())
        terms = 0
        if mx.any():
            loss = loss + Fnn.l1_loss(gx_r[mx.expand_as(gx_r)], gx_t[mx.expand_as(gx_t)])
            terms += 1
        if my.any():
            loss = loss + Fnn.l1_loss(gy_r[my.expand_as(gy_r)], gy_t[my.expand_as(gy_t)])
            terms += 1
        return loss / max(terms, 1)

    def _detail_teacher_loss(render: torch.Tensor,
                             target: torch.Tensor,
                             alpha: torch.Tensor,
                             fg: torch.Tensor,
                             edge_thresh: float,
                             alpha_min: float,
                             artifact_weight: float) -> torch.Tensor:
        """Preserve real target detail while discouraging target-smooth artifacts.

        Unlike the blunt foreground gradient loss, this only matches RGB
        gradients where the target image itself has foreground texture/edges.
        A smaller artifact term penalizes render gradients in target-smooth
        foreground regions, which helps avoid lattice/shell sharpness.
        """
        edge_thresh = max(float(edge_thresh), 1e-6)
        alpha_min = float(alpha_min)
        artifact_weight = max(float(artifact_weight), 0.0)
        fg_f = fg.detach().clamp(0.0, 1.0)
        alpha_f = alpha.detach().clamp(0.0, 1.0)

        def axis_loss(gr: torch.Tensor, gt: torch.Tensor,
                      fg_pair: torch.Tensor,
                      alpha_pair: torch.Tensor) -> torch.Tensor:
            pair = fg_pair * (alpha_pair > alpha_min).to(dtype=gr.dtype)
            mag_t = gt.detach().abs().mean(dim=-1, keepdim=True)
            mag_r = gr.abs().mean(dim=-1, keepdim=True)
            detail_w = (torch.relu(mag_t - edge_thresh) / edge_thresh).clamp(0.0, 1.0)
            detail_w = detail_w * pair
            loss_v = gr.new_zeros(())
            den = detail_w.sum()
            if den > 0:
                loss_v = (
                    (gr - gt).abs().mean(dim=-1, keepdim=True) * detail_w
                ).sum() / den.clamp_min(1e-6)
            if artifact_weight > 0:
                smooth_w = (1.0 - (mag_t / edge_thresh).clamp(0.0, 1.0)) * pair
                smooth_den = smooth_w.sum()
                if smooth_den > 0:
                    art = (mag_r * smooth_w).sum() / smooth_den.clamp_min(1e-6)
                    loss_v = loss_v + artifact_weight * art
            return loss_v

        gx_r = render[:, :, 1:, :] - render[:, :, :-1, :]
        gx_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        gx_fg = torch.minimum(fg_f[:, :, 1:, :], fg_f[:, :, :-1, :])
        gx_alpha = torch.minimum(alpha_f[:, :, 1:, :], alpha_f[:, :, :-1, :])
        gy_r = render[:, 1:, :, :] - render[:, :-1, :, :]
        gy_t = target[:, 1:, :, :] - target[:, :-1, :, :]
        gy_fg = torch.minimum(fg_f[:, 1:, :, :], fg_f[:, :-1, :, :])
        gy_alpha = torch.minimum(alpha_f[:, 1:, :, :], alpha_f[:, :-1, :, :])
        return 0.5 * (
            axis_loss(gx_r, gx_t, gx_fg, gx_alpha)
            + axis_loss(gy_r, gy_t, gy_fg, gy_alpha)
        )

    def _alpha_gradient_loss(alpha: torch.Tensor,
                             target: torch.Tensor,
                             band_px: int = 2) -> torch.Tensor:
        """Match alpha edge gradients to the GT silhouette in a narrow edge band."""
        gx_a = alpha[:, :, 1:, :] - alpha[:, :, :-1, :]
        gx_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        gy_a = alpha[:, 1:, :, :] - alpha[:, :-1, :, :]
        gy_t = target[:, 1:, :, :] - target[:, :-1, :, :]

        edge = alpha.new_zeros(alpha.shape)
        ex = gx_t.detach().abs()
        ey = gy_t.detach().abs()
        edge[:, :, 1:, :] = torch.maximum(edge[:, :, 1:, :], ex)
        edge[:, :, :-1, :] = torch.maximum(edge[:, :, :-1, :], ex)
        edge[:, 1:, :, :] = torch.maximum(edge[:, 1:, :, :], ey)
        edge[:, :-1, :, :] = torch.maximum(edge[:, :-1, :, :], ey)
        if band_px > 0:
            e = edge.permute(0, 3, 1, 2)
            e = Fnn.max_pool2d(
                e,
                kernel_size=2 * int(band_px) + 1,
                stride=1,
                padding=int(band_px),
            )
            edge = e.permute(0, 2, 3, 1)
        wx = torch.maximum(edge[:, :, 1:, :], edge[:, :, :-1, :])
        wy = torch.maximum(edge[:, 1:, :, :], edge[:, :-1, :, :])
        loss = alpha.new_zeros(())
        terms = 0
        if bool((wx > 0).any()):
            loss = loss + ((gx_a - gx_t).abs() * wx).sum() / wx.sum().clamp_min(1.0)
            terms += 1
        if bool((wy > 0).any()):
            loss = loss + ((gy_a - gy_t).abs() * wy).sum() / wy.sum().clamp_min(1.0)
            terms += 1
        return loss / max(terms, 1)

    def _alpha_interior_smooth_loss(alpha: torch.Tensor,
                                    target: torch.Tensor,
                                    edge_band_px: int = 4) -> torch.Tensor:
        """Suppress regular alpha texture inside the object silhouette."""
        fg = (target.detach() > 0.5).to(dtype=alpha.dtype)
        edge = alpha.new_zeros(alpha.shape)
        gx_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        gy_t = target[:, 1:, :, :] - target[:, :-1, :, :]
        ex = gx_t.detach().abs()
        ey = gy_t.detach().abs()
        edge[:, :, 1:, :] = torch.maximum(edge[:, :, 1:, :], ex)
        edge[:, :, :-1, :] = torch.maximum(edge[:, :, :-1, :], ex)
        edge[:, 1:, :, :] = torch.maximum(edge[:, 1:, :, :], ey)
        edge[:, :-1, :, :] = torch.maximum(edge[:, :-1, :, :], ey)
        if edge_band_px > 0:
            e = edge.permute(0, 3, 1, 2)
            e = Fnn.max_pool2d(
                e,
                kernel_size=2 * int(edge_band_px) + 1,
                stride=1,
                padding=int(edge_band_px),
            )
            edge = e.permute(0, 2, 3, 1)
        interior = fg * (edge <= 0).to(dtype=alpha.dtype)
        gx_a = alpha[:, :, 1:, :] - alpha[:, :, :-1, :]
        gy_a = alpha[:, 1:, :, :] - alpha[:, :-1, :, :]
        wx = torch.minimum(interior[:, :, 1:, :], interior[:, :, :-1, :])
        wy = torch.minimum(interior[:, 1:, :, :], interior[:, :-1, :, :])
        loss = alpha.new_zeros(())
        terms = 0
        if bool((wx > 0).any()):
            loss = loss + (gx_a.abs() * wx).sum() / wx.sum().clamp_min(1.0)
            terms += 1
        if bool((wy > 0).any()):
            loss = loss + (gy_a.abs() * wy).sum() / wy.sum().clamp_min(1.0)
            terms += 1
        return loss / max(terms, 1)

    def _visibility_condition(view_idx: int, fg: torch.Tensor,
                              depths: torch.Tensor | None,
                              K_all: torch.Tensor | None,
                              c2w_all: torch.Tensor | None,
                              w2c_all: torch.Tensor | None,
                              radius: float | None) -> torch.Tensor:
        h, w = fg.shape[1], fg.shape[2]
        if depths is None or K_all is None or c2w_all is None or w2c_all is None or radius is None:
            return fg.new_zeros(1, h, w)
        with torch.no_grad():
            z_src = depths[view_idx].to(device=fg.device, dtype=fg.dtype)
            valid_src = (z_src < 1e5) & (fg[view_idx, ..., 0] > 0.5)
            t_src = zdepth_to_raydist(z_src, K_all[view_idx].to(device=fg.device, dtype=fg.dtype))
            dirs = ray_dirs_world(K_all[view_idx], c2w_all[view_idx], h, w).to(
                device=fg.device, dtype=fg.dtype
            )
            origin = c2w_all[view_idx, :3, 3].to(device=fg.device, dtype=fg.dtype)
            means = origin[None] + t_src.reshape(-1, 1) * dirs
            conflicts = torch.zeros(h * w, dtype=fg.dtype, device=fg.device)
            supports = torch.zeros(h * w, dtype=fg.dtype, device=fg.device)
            tol = args.fusion_depth_tol_frac * float(radius)
            margin = max(args.fusion_bg_margin_px, 0)
            if args.image_visibility_all_views:
                ref_views = list(range(min(depths.shape[0], K_all.shape[0])))
            else:
                ref_views = _eval_anchor_indices(K_all.shape[0], args.anchor_views)
            if args.image_visibility_nearest_refs > 0 and c2w_all is not None:
                candidates = [v for v in ref_views if v != view_idx and v < depths.shape[0]]
                if candidates:
                    cand_t = torch.as_tensor(candidates, device=fg.device, dtype=torch.long)
                    centers = c2w_all[:, :3, 3].to(device=fg.device, dtype=fg.dtype)
                    d = torch.linalg.norm(centers[cand_t] - centers[view_idx:view_idx + 1], dim=1)
                    order = d.argsort()[:args.image_visibility_nearest_refs].tolist()
                    ref_views = [candidates[i] for i in order]
                else:
                    ref_views = []
            support_views = ref_views
            if args.image_visibility_min_support > 0:
                support_views = [view_idx] + [v for v in ref_views if v != view_idx]
            for ref_view in support_views:
                if ref_view >= depths.shape[0]:
                    continue
                cam = means @ w2c_all[ref_view, :3, :3].T + w2c_all[ref_view, :3, 3]
                z = cam[:, 2]
                fx, fy = K_all[ref_view, 0, 0], K_all[ref_view, 1, 1]
                cx, cy = K_all[ref_view, 0, 2], K_all[ref_view, 1, 2]
                u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
                v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
                inb = (z > 1e-6) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
                if not inb.any():
                    continue

                fg_view = fg[ref_view, ..., 0].to(device=fg.device) > 0.5
                if margin > 0:
                    fg_view = Fnn.max_pool2d(
                        fg_view.float()[None, None],
                        kernel_size=2 * margin + 1,
                        stride=1,
                        padding=margin,
                    )[0, 0] > 0.5
                z_view = depths[ref_view].to(device=fg.device, dtype=fg.dtype)

                idx = torch.nonzero(inb, as_tuple=False).squeeze(1)
                sampled_fg, depth_match, front_conflict, bidir_conflict = _sample_depth_support_window(
                    fg_view,
                    z_view,
                    u[idx],
                    v[idx],
                    z[idx],
                    tol,
                    int(args.support_sample_radius_px),
                )
                if args.image_visibility_min_support > 0 and depth_match.any():
                    supports[idx[depth_match]] += 1.0
                if ref_view == view_idx:
                    continue
                if args.image_visibility_bidirectional:
                    depth_conflict = bidir_conflict
                else:
                    depth_conflict = front_conflict
                conflict = (~sampled_fg) | depth_conflict
                if conflict.any():
                    conflicts[idx[conflict]] += 1.0
            if args.image_visibility_min_support > 0:
                missing_support = (float(args.image_visibility_min_support) - supports).clamp_min(0.0)
            else:
                missing_support = conflicts.new_zeros(conflicts.shape)
            visibility = torch.exp(-args.image_visibility_decay * (conflicts + missing_support)).reshape(h, w)
            visibility = visibility * valid_src.to(fg.dtype)
            return visibility[None]

    def _photo_visibility_condition(view_idx: int, frames: torch.Tensor,
                                    fg: torch.Tensor,
                                    depths: torch.Tensor | None,
                                    K_all: torch.Tensor | None,
                                    c2w_all: torch.Tensor | None,
                                    w2c_all: torch.Tensor | None) -> torch.Tensor:
        h, w = fg.shape[1], fg.shape[2]
        if depths is None or K_all is None or c2w_all is None or w2c_all is None:
            return fg.new_zeros(1, h, w)
        with torch.no_grad():
            z_src = depths[view_idx].to(device=frames.device, dtype=frames.dtype)
            valid_src = (z_src < 1e5) & (fg[view_idx, ..., 0] > 0.5)
            if not valid_src.any():
                return fg.new_zeros(1, h, w)
            t_src = zdepth_to_raydist(z_src, K_all[view_idx].to(device=frames.device, dtype=frames.dtype))
            dirs = ray_dirs_world(K_all[view_idx], c2w_all[view_idx], h, w).to(
                device=frames.device, dtype=frames.dtype
            )
            origin = c2w_all[view_idx, :3, 3].to(device=frames.device, dtype=frames.dtype)
            means = origin[None] + t_src.reshape(-1, 1) * dirs

            n = K_all.shape[0]
            refs = [j for j in range(n) if j != view_idx]
            if args.image_photo_visibility_refs > 0 and len(refs) > args.image_photo_visibility_refs:
                centers = c2w_all[:, :3, 3].to(device=frames.device, dtype=frames.dtype)
                ref_t = torch.as_tensor(refs, device=frames.device, dtype=torch.long)
                dist = torch.linalg.norm(centers[ref_t] - centers[view_idx:view_idx + 1], dim=1)
                order = dist.argsort()[:args.image_photo_visibility_refs].tolist()
                refs = [refs[i] for i in order]
            if not refs:
                return valid_src.to(frames.dtype)[None]

            src_rgb = frames[view_idx].reshape(h * w, 3).to(device=frames.device, dtype=frames.dtype)
            accum = torch.zeros(h * w, device=frames.device, dtype=frames.dtype)
            count = torch.zeros(h * w, device=frames.device, dtype=frames.dtype)
            for ref_i in refs:
                cam = means @ w2c_all[ref_i, :3, :3].T + w2c_all[ref_i, :3, 3]
                z = cam[:, 2]
                fx, fy = K_all[ref_i, 0, 0], K_all[ref_i, 1, 1]
                cx, cy = K_all[ref_i, 0, 2], K_all[ref_i, 1, 2]
                u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
                v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
                inb = (z > 1e-6) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
                grid_x = (u / max(w - 1, 1)) * 2.0 - 1.0
                grid_y = (v / max(h - 1, 1)) * 2.0 - 1.0
                grid = torch.stack([grid_x, grid_y], -1).view(1, h * w, 1, 2)
                ref_img = frames[ref_i].permute(2, 0, 1)[None].to(device=frames.device, dtype=frames.dtype)
                ref_mask = fg[ref_i].permute(2, 0, 1)[None].to(device=frames.device, dtype=frames.dtype)
                samp_rgb = Fnn.grid_sample(
                    ref_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
                ).view(3, h * w).T
                samp_mask = Fnn.grid_sample(
                    ref_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=True
                ).view(h * w)
                color = (samp_rgb - src_rgb).abs().mean(dim=-1)
                score = torch.exp(-args.image_photo_visibility_color_decay * color)
                score = score * samp_mask.clamp(0.0, 1.0) * inb.to(frames.dtype)
                accum = accum + score
                count = count + inb.to(frames.dtype)
            vis = accum / count.clamp_min(1.0)
            vis = vis.reshape(h, w) * valid_src.to(frames.dtype)
            return vis.clamp(0.0, 1.0)[None]

    def _surface_normal_condition(view_idx: int, fg: torch.Tensor,
                                  depths: torch.Tensor | None,
                                  K_all: torch.Tensor | None,
                                  c2w_all: torch.Tensor | None) -> torch.Tensor:
        h, w = fg.shape[1], fg.shape[2]
        if depths is None or K_all is None or c2w_all is None:
            return fg.new_zeros(3, h, w)
        with torch.no_grad():
            z = depths[view_idx].to(device=fg.device, dtype=fg.dtype)
            valid = (z < 1e5) & (fg[view_idx, ..., 0] > 0.5)
            if h < 3 or w < 3 or not valid.any():
                return fg.new_zeros(3, h, w)
            t = zdepth_to_raydist(z, K_all[view_idx].to(device=fg.device, dtype=fg.dtype))
            dirs = ray_dirs_world(K_all[view_idx], c2w_all[view_idx], h, w).to(
                device=fg.device, dtype=fg.dtype
            ).reshape(h, w, 3)
            origin = c2w_all[view_idx, :3, 3].to(device=fg.device, dtype=fg.dtype)
            pts = origin.view(1, 1, 3) + t[..., None] * dirs
            dx = pts[1:-1, 2:] - pts[1:-1, :-2]
            dy = pts[2:, 1:-1] - pts[:-2, 1:-1]
            n = Fnn.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
            valid_c = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
                       & valid[2:, 1:-1] & valid[:-2, 1:-1])
            view_vec = Fnn.normalize(origin.view(1, 1, 3) - pts[1:-1, 1:-1], dim=-1)
            n = torch.where((n * view_vec).sum(-1, keepdim=True) < 0, -n, n)
            normal = fg.new_zeros(h, w, 3)
            normal[1:-1, 1:-1] = n * valid_c[..., None].to(fg.dtype)
            return normal.permute(2, 0, 1)

    def _scaled_intrinsics(K_all: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
        K_s = K_all.clone()
        K_s[:, 0, 0] *= sx
        K_s[:, 0, 2] *= sx
        K_s[:, 1, 1] *= sy
        K_s[:, 1, 2] *= sy
        return K_s

    def _visual_hull_depths(fg: torch.Tensor, K_all: torch.Tensor,
                            c2w_all: torch.Tensor, w2c_all: torch.Tensor,
                            radius: float) -> torch.Tensor:
        """Silhouette-carved front depth from conditioning masks/cameras.

        This is a deterministic feed-forward prior. It does not use GT depth and
        does not optimize per object; it just finds the first ray sample whose
        3D point projects inside enough source silhouettes.
        """
        n_views, H, W = fg.shape[0], fg.shape[1], fg.shape[2]
        scale = min(max(args.visual_hull_scale, 0.02), 1.0)
        vh = max(8, int(round(H * scale)))
        vw = max(8, int(round(W * scale)))
        device, dtype = fg.device, fg.dtype
        masks = Fnn.interpolate(
            fg.permute(0, 3, 1, 2).to(dtype=dtype), size=(vh, vw), mode="area"
        ).clamp(0.0, 1.0)
        margin = max(args.visual_hull_mask_margin, 0)
        if margin > 0:
            masks = Fnn.max_pool2d(masks, kernel_size=2 * margin + 1, stride=1, padding=margin)
        K_s = _scaled_intrinsics(K_all.to(device=device, dtype=dtype), vw / W, vh / H)
        c2w_s = c2w_all.to(device=device, dtype=dtype)
        w2c_s = w2c_all.to(device=device, dtype=dtype)
        samples = torch.linspace(0.0, 1.0, max(args.visual_hull_samples, 2),
                                 device=device, dtype=dtype)
        min_views = max(1, min(n_views, int(math.ceil(args.visual_hull_min_view_frac * n_views))))
        out_z = []
        for src_i in range(n_views):
            dirs = ray_dirs_world(K_s[src_i], c2w_s[src_i], vh, vw).to(device=device, dtype=dtype)
            origin = c2w_s[src_i, :3, 3]
            near, far = depth_bounds(c2w_s[src_i], radius, model.half_frac)
            t = near + samples * (far - near)
            pts = origin.view(1, 1, 3) + dirs[:, None, :] * t.view(1, -1, 1)
            pts_flat = pts.reshape(-1, 3)
            inside_count = torch.zeros(pts_flat.shape[0], device=device, dtype=dtype)
            for ref_i in range(n_views):
                cam = pts_flat @ w2c_s[ref_i, :3, :3].T + w2c_s[ref_i, :3, 3]
                z = cam[:, 2]
                fx, fy = K_s[ref_i, 0, 0], K_s[ref_i, 1, 1]
                cx, cy = K_s[ref_i, 0, 2], K_s[ref_i, 1, 2]
                u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
                v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
                inb = (z > 1e-6) & (u >= 0) & (u <= vw - 1) & (v >= 0) & (v <= vh - 1)
                grid_x = (u / max(vw - 1, 1)) * 2.0 - 1.0
                grid_y = (v / max(vh - 1, 1)) * 2.0 - 1.0
                grid = torch.stack([grid_x, grid_y], -1).view(1, -1, 1, 2)
                m = Fnn.grid_sample(masks[ref_i:ref_i + 1], grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=True)
                inside_count += ((m.view(-1) > 0.25) & inb).to(dtype)
            inside = inside_count.reshape(vh * vw, -1) >= min_views
            src_fg = masks[src_i, 0].reshape(-1) > 0.25
            any_hit = inside.any(dim=1) & src_fg
            first = inside.float().argmax(dim=1)
            t_hit = t[first]
            pts_hit = origin.view(1, 3) + dirs * t_hit[:, None]
            cam_src = pts_hit @ w2c_s[src_i, :3, :3].T + w2c_s[src_i, :3, 3]
            z_hit = torch.where(any_hit, cam_src[:, 2], cam_src.new_full((vh * vw,), 1e10))
            z_lr = z_hit.reshape(1, 1, vh, vw)
            z_full = Fnn.interpolate(z_lr, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            out_z.append(z_full)
        return torch.stack(out_z)

    def _voxel_hull_depths(fg: torch.Tensor, K_all: torch.Tensor,
                           c2w_all: torch.Tensor, w2c_all: torch.Tensor,
                           radius: float) -> torch.Tensor:
        """Build a coarse visual hull volume, then raycast it from each source view."""
        n_views, H, W = fg.shape[0], fg.shape[1], fg.shape[2]
        scale = min(max(args.voxel_hull_scale, 0.02), 1.0)
        vh = max(8, int(round(H * scale)))
        vw = max(8, int(round(W * scale)))
        grid_n = max(args.voxel_hull_grid, 8)
        device, dtype = fg.device, fg.dtype
        masks = Fnn.interpolate(
            fg.permute(0, 3, 1, 2).to(dtype=dtype), size=(vh, vw), mode="area"
        ).clamp(0.0, 1.0)
        margin = max(args.voxel_hull_mask_margin, 0)
        if margin > 0:
            masks = Fnn.max_pool2d(masks, kernel_size=2 * margin + 1, stride=1, padding=margin)
        K_s = _scaled_intrinsics(K_all.to(device=device, dtype=dtype), vw / W, vh / H)
        c2w_s = c2w_all.to(device=device, dtype=dtype)
        w2c_s = w2c_all.to(device=device, dtype=dtype)
        half = max(float(radius) * args.voxel_hull_bounds_frac, 1e-3)
        lin = torch.linspace(-half, half, grid_n, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        inside_count = torch.zeros(pts.shape[0], device=device, dtype=dtype)
        for view_i in range(n_views):
            cam = pts @ w2c_s[view_i, :3, :3].T + w2c_s[view_i, :3, 3]
            z = cam[:, 2]
            fx, fy = K_s[view_i, 0, 0], K_s[view_i, 1, 1]
            cx, cy = K_s[view_i, 0, 2], K_s[view_i, 1, 2]
            u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
            v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
            inb = (z > 1e-6) & (u >= 0) & (u <= vw - 1) & (v >= 0) & (v <= vh - 1)
            grid_x = (u / max(vw - 1, 1)) * 2.0 - 1.0
            grid_y = (v / max(vh - 1, 1)) * 2.0 - 1.0
            sample_grid = torch.stack([grid_x, grid_y], -1).view(1, -1, 1, 2)
            m = Fnn.grid_sample(masks[view_i:view_i + 1], sample_grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
            inside_count += ((m.view(-1) > 0.25) & inb).to(dtype)
        min_views = max(1, min(n_views, int(math.ceil(args.voxel_hull_min_view_frac * n_views))))
        occ = (inside_count.reshape(grid_n, grid_n, grid_n) >= min_views).to(dtype)
        if args.voxel_hull_dilate > 0:
            occ = Fnn.max_pool3d(
                occ[None, None],
                kernel_size=2 * args.voxel_hull_dilate + 1,
                stride=1,
                padding=args.voxel_hull_dilate,
            )[0, 0]
        occ_vol = occ[None, None]
        n_samples = max(args.voxel_hull_samples, 2)
        sample_frac = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
        out_z = []
        for src_i in range(n_views):
            dirs = ray_dirs_world(K_s[src_i], c2w_s[src_i], vh, vw).to(device=device, dtype=dtype)
            origin = c2w_s[src_i, :3, 3]
            near, far = depth_bounds(c2w_s[src_i], radius, model.half_frac)
            t = near + sample_frac * (far - near)
            pts_ray = origin.view(1, 1, 3) + dirs[:, None, :] * t.view(1, -1, 1)
            coords = (pts_ray / half).clamp(-1.1, 1.1)
            sample_grid = coords.reshape(1, vh * vw * n_samples, 1, 1, 3)
            occ_s = Fnn.grid_sample(occ_vol, sample_grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=True)
            inside = occ_s.view(vh * vw, n_samples) > 0.25
            src_fg = masks[src_i, 0].reshape(vh * vw) > 0.25
            any_hit = inside.any(dim=1) & src_fg
            first = inside.float().argmax(dim=1)
            t_hit = t[first]
            pts_hit = origin.view(1, 3) + dirs * t_hit[:, None]
            cam_src = pts_hit @ w2c_s[src_i, :3, :3].T + w2c_s[src_i, :3, 3]
            z_hit = torch.where(any_hit, cam_src[:, 2], cam_src.new_full((vh * vw,), 1e10))
            z_lr = z_hit.reshape(1, 1, vh, vw)
            z_full = Fnn.interpolate(z_lr, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            out_z.append(z_full)
        return torch.stack(out_z)

    def _hull_clamped_depths(prior_depths: torch.Tensor | None,
                             fg: torch.Tensor,
                             K_all: torch.Tensor,
                             c2w_all: torch.Tensor,
                             w2c_all: torch.Tensor,
                             radius: float) -> torch.Tensor | None:
        """Clamp only DA3/front-depth outliers to the source-mask visual hull.

        Full visual-hull depth replacement was too coarse. This variant keeps
        the input depth everywhere except pixels that sit clearly in front of
        the mask hull along the same source ray, which are usually floaters.
        """
        if prior_depths is None or not args.image_hull_clamp_depth:
            return prior_depths
        if args.image_hull_clamp_mode == "visual":
            hull_z = _visual_hull_depths(fg, K_all, c2w_all, w2c_all, radius)
        else:
            hull_z = _voxel_hull_depths(fg, K_all, c2w_all, w2c_all, radius)
        prior = prior_depths.to(device=fg.device, dtype=fg.dtype)
        hull_z = hull_z.to(device=prior.device, dtype=prior.dtype)
        valid = (
            torch.isfinite(prior)
            & torch.isfinite(hull_z)
            & (prior > 1e-6)
            & (prior < 1e5)
            & (hull_z > 1e-6)
            & (hull_z < 1e5)
            & (fg[..., 0] > 0.5)
        )
        tol = max(float(args.image_hull_clamp_tol_frac) * float(radius), 0.0)
        too_front = valid & (prior < hull_z - tol)
        if not too_front.any():
            return prior
        max_shift = max(float(args.image_hull_clamp_max_shift_frac), 0.0) * float(radius)
        if max_shift > 0:
            fixed = prior + (hull_z - prior).clamp_min(0.0).clamp_max(max_shift)
        else:
            fixed = hull_z
        return torch.where(too_front, fixed, prior)

    def _target_visual_hull_masks(source_fg: torch.Tensor,
                                  source_K: torch.Tensor,
                                  source_c2w: torch.Tensor,
                                  source_w2c: torch.Tensor,
                                  target_K: torch.Tensor,
                                  target_c2w: torch.Tensor,
                                  target_w2c: torch.Tensor,
                                  radius: float,
                                  height: int,
                                  width: int) -> torch.Tensor:
        """Project the source-mask visual hull into arbitrary target cameras.

        This is a feed-forward visibility prior for nearest-fill rendering. It
        uses only conditioning masks/cameras, not target-view RGB or masks.
        """
        n_src, H, W = source_fg.shape[0], source_fg.shape[1], source_fg.shape[2]
        scale = min(max(args.voxel_hull_scale, 0.02), 1.0)
        vh = max(8, int(round(height * scale)))
        vw = max(8, int(round(width * scale)))
        grid_n = max(args.voxel_hull_grid, 8)
        device, dtype = source_fg.device, source_fg.dtype
        src_masks = Fnn.interpolate(
            source_fg.permute(0, 3, 1, 2).to(dtype=dtype),
            size=(max(8, int(round(H * scale))), max(8, int(round(W * scale)))),
            mode="area",
        ).clamp(0.0, 1.0)
        mh, mw = src_masks.shape[-2:]
        margin = max(args.voxel_hull_mask_margin, 0)
        if margin > 0:
            src_masks = Fnn.max_pool2d(
                src_masks, kernel_size=2 * margin + 1, stride=1, padding=margin
            )
        K_src = _scaled_intrinsics(source_K.to(device=device, dtype=dtype), mw / W, mh / H)
        c2w_src = source_c2w.to(device=device, dtype=dtype)
        w2c_src = source_w2c.to(device=device, dtype=dtype)
        half = max(float(radius) * args.voxel_hull_bounds_frac, 1e-3)
        lin = torch.linspace(-half, half, grid_n, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        inside_count = torch.zeros(pts.shape[0], device=device, dtype=dtype)
        for view_i in range(n_src):
            cam = pts @ w2c_src[view_i, :3, :3].T + w2c_src[view_i, :3, 3]
            z = cam[:, 2]
            fx, fy = K_src[view_i, 0, 0], K_src[view_i, 1, 1]
            cx, cy = K_src[view_i, 0, 2], K_src[view_i, 1, 2]
            u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
            v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
            inb = (z > 1e-6) & (u >= 0) & (u <= mw - 1) & (v >= 0) & (v <= mh - 1)
            grid_x = (u / max(mw - 1, 1)) * 2.0 - 1.0
            grid_y = (v / max(mh - 1, 1)) * 2.0 - 1.0
            sample_grid = torch.stack([grid_x, grid_y], -1).view(1, -1, 1, 2)
            m = Fnn.grid_sample(
                src_masks[view_i:view_i + 1], sample_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True
            )
            inside_count += ((m.view(-1) > 0.25) & inb).to(dtype)
        min_views = max(1, min(n_src, int(math.ceil(args.voxel_hull_min_view_frac * n_src))))
        occ = (inside_count.reshape(grid_n, grid_n, grid_n) >= min_views).to(dtype)
        if args.voxel_hull_dilate > 0:
            occ = Fnn.max_pool3d(
                occ[None, None],
                kernel_size=2 * args.voxel_hull_dilate + 1,
                stride=1,
                padding=args.voxel_hull_dilate,
            )[0, 0]
        occ_vol = occ[None, None]
        K_tgt = _scaled_intrinsics(target_K.to(device=device, dtype=dtype), vw / width, vh / height)
        c2w_tgt = target_c2w.to(device=device, dtype=dtype)
        n_samples = max(args.voxel_hull_samples, 2)
        sample_frac = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
        out = []
        for tgt_i in range(target_K.shape[0]):
            dirs = ray_dirs_world(K_tgt[tgt_i], c2w_tgt[tgt_i], vh, vw).to(
                device=device, dtype=dtype
            )
            origin = c2w_tgt[tgt_i, :3, 3]
            near, far = depth_bounds(c2w_tgt[tgt_i], radius, model.half_frac)
            t = near + sample_frac * (far - near)
            pts_ray = origin.view(1, 1, 3) + dirs[:, None, :] * t.view(1, -1, 1)
            coords = (pts_ray / half).clamp(-1.1, 1.1)
            sample_grid = coords.reshape(1, vh * vw * n_samples, 1, 1, 3)
            occ_s = Fnn.grid_sample(
                occ_vol, sample_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True
            )
            hit = (occ_s.view(vh * vw, n_samples) > 0.25).any(dim=1)
            mask_lr = hit.to(dtype).reshape(1, 1, vh, vw)
            mask = Fnn.interpolate(
                mask_lr, size=(height, width), mode="bilinear", align_corners=False
            )[0, 0].clamp(0.0, 1.0)
            out.append(mask[..., None])
        return torch.stack(out, 0)

    def _nearest_ref_views(src_i: int, c2w_all: torch.Tensor) -> list[int]:
        n = c2w_all.shape[0]
        refs = [j for j in range(n) if j != src_i]
        if args.plane_sweep_refs <= 0 or args.plane_sweep_refs >= len(refs):
            return refs
        centers = c2w_all[:, :3, 3]
        d = torch.linalg.norm(centers[refs] - centers[src_i:src_i + 1], dim=1)
        order = d.argsort()[:args.plane_sweep_refs].tolist()
        return [refs[i] for i in order]

    def _plane_sweep_depths(frames: torch.Tensor, fg: torch.Tensor,
                            K_all: torch.Tensor, c2w_all: torch.Tensor,
                            w2c_all: torch.Tensor, radius: float) -> torch.Tensor:
        """Photometric/mask plane sweep over decoded source views.

        This is deterministic feed-forward depth estimation: no GT depth and no
        per-object optimization. It finds the ray sample whose projection has
        the lowest RGB + silhouette cost in nearby decoded views.
        """
        n_views, H, W = fg.shape[0], fg.shape[1], fg.shape[2]
        scale = min(max(args.plane_sweep_scale, 0.02), 1.0)
        vh = max(8, int(round(H * scale)))
        vw = max(8, int(round(W * scale)))
        device, dtype = frames.device, frames.dtype
        imgs = Fnn.interpolate(
            frames.permute(0, 3, 1, 2).to(dtype=dtype), size=(vh, vw),
            mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        masks = Fnn.interpolate(
            fg.permute(0, 3, 1, 2).to(dtype=dtype), size=(vh, vw), mode="area"
        ).clamp(0.0, 1.0)
        margin = max(args.plane_sweep_mask_margin, 0)
        if margin > 0:
            masks = Fnn.max_pool2d(masks, kernel_size=2 * margin + 1, stride=1, padding=margin)
        K_s = _scaled_intrinsics(K_all.to(device=device, dtype=dtype), vw / W, vh / H)
        c2w_s = c2w_all.to(device=device, dtype=dtype)
        w2c_s = w2c_all.to(device=device, dtype=dtype)
        n_samples = max(args.plane_sweep_samples, 2)
        sample_frac = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
        out_z = []
        for src_i in range(n_views):
            refs = _nearest_ref_views(src_i, c2w_s)
            dirs = ray_dirs_world(K_s[src_i], c2w_s[src_i], vh, vw).to(device=device, dtype=dtype)
            origin = c2w_s[src_i, :3, 3]
            near, far = depth_bounds(c2w_s[src_i], radius, model.half_frac)
            t = near + sample_frac * (far - near)
            pts = origin.view(1, 1, 3) + dirs[:, None, :] * t.view(1, -1, 1)
            pts_flat = pts.reshape(-1, 3)
            src_rgb = imgs[src_i].permute(1, 2, 0).reshape(vh * vw, 1, 3)
            src_fg = masks[src_i, 0].reshape(vh * vw) > 0.25
            cost = sample_frac.view(1, -1) * args.plane_sweep_front_bias
            cost = cost.expand(vh * vw, n_samples).clone()
            n_cost = 0
            for ref_i in refs:
                cam = pts_flat @ w2c_s[ref_i, :3, :3].T + w2c_s[ref_i, :3, 3]
                z = cam[:, 2]
                fx, fy = K_s[ref_i, 0, 0], K_s[ref_i, 1, 1]
                cx, cy = K_s[ref_i, 0, 2], K_s[ref_i, 1, 2]
                u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
                v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
                inb = (z > 1e-6) & (u >= 0) & (u <= vw - 1) & (v >= 0) & (v <= vh - 1)
                grid_x = (u / max(vw - 1, 1)) * 2.0 - 1.0
                grid_y = (v / max(vh - 1, 1)) * 2.0 - 1.0
                grid = torch.stack([grid_x, grid_y], -1).view(1, -1, 1, 2)
                ref_rgb = Fnn.grid_sample(imgs[ref_i:ref_i + 1], grid, mode="bilinear",
                                          padding_mode="zeros", align_corners=True)
                ref_mask = Fnn.grid_sample(masks[ref_i:ref_i + 1], grid, mode="bilinear",
                                           padding_mode="zeros", align_corners=True)
                ref_rgb = ref_rgb.view(3, vh * vw, n_samples).permute(1, 2, 0)
                ref_mask = ref_mask.view(vh * vw, n_samples) * inb.view(vh * vw, n_samples).to(dtype)
                color = (ref_rgb - src_rgb).abs().mean(dim=-1)
                cost = cost + args.plane_sweep_color_weight * color * ref_mask
                cost = cost + args.plane_sweep_mask_weight * (1.0 - ref_mask)
                n_cost += 1
            if n_cost > 0:
                cost = cost / float(n_cost)
            cost = torch.where(src_fg[:, None], cost, cost.new_full(cost.shape, 1e6))
            best = cost.argmin(dim=1)
            t_hit = t[best]
            pts_hit = origin.view(1, 3) + dirs * t_hit[:, None]
            cam_src = pts_hit @ w2c_s[src_i, :3, :3].T + w2c_s[src_i, :3, 3]
            z_hit = torch.where(src_fg, cam_src[:, 2], cam_src.new_full((vh * vw,), 1e10))
            z_lr = z_hit.reshape(1, 1, vh, vw)
            z_full = Fnn.interpolate(z_lr, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            out_z.append(z_full)
        return torch.stack(out_z)

    def _guided_plane_sweep_depths(frames: torch.Tensor, fg: torch.Tensor,
                                   prior_depths: torch.Tensor | None,
                                   K_all: torch.Tensor, c2w_all: torch.Tensor,
                                   w2c_all: torch.Tensor, radius: float) -> torch.Tensor:
        """Local plane sweep around an input depth prior.

        Depth-Anything gives useful per-view shape but its independent monocular
        estimates are not multi-view consistent. This keeps its metric placement
        as a prior, then searches a narrow ray interval using nearby source
        views for RGB/mask agreement. It is still feed-forward; no per-object
        optimization state is introduced.
        """
        if prior_depths is None:
            return _plane_sweep_depths(frames, fg, K_all, c2w_all, w2c_all, radius)
        n_views, H, W = fg.shape[0], fg.shape[1], fg.shape[2]
        scale = min(max(args.plane_sweep_scale, 0.02), 1.0)
        vh = max(8, int(round(H * scale)))
        vw = max(8, int(round(W * scale)))
        device, dtype = frames.device, frames.dtype
        imgs = Fnn.interpolate(
            frames.permute(0, 3, 1, 2).to(dtype=dtype), size=(vh, vw),
            mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        masks = Fnn.interpolate(
            fg.permute(0, 3, 1, 2).to(dtype=dtype), size=(vh, vw), mode="area"
        ).clamp(0.0, 1.0)
        margin = max(args.plane_sweep_mask_margin, 0)
        if margin > 0:
            masks = Fnn.max_pool2d(masks, kernel_size=2 * margin + 1, stride=1, padding=margin)
        prior_z = prior_depths.to(device=device, dtype=dtype)
        prior_z = Fnn.interpolate(prior_z[:, None], size=(vh, vw), mode="nearest")[:, 0]
        K_s = _scaled_intrinsics(K_all.to(device=device, dtype=dtype), vw / W, vh / H)
        c2w_s = c2w_all.to(device=device, dtype=dtype)
        w2c_s = w2c_all.to(device=device, dtype=dtype)
        n_samples = max(args.plane_sweep_samples, 2)
        offsets = torch.linspace(-1.0, 1.0, n_samples, device=device, dtype=dtype)
        interval = max(args.guided_plane_sweep_radius_frac, 1e-4) * float(radius)
        out_z = []
        for src_i in range(n_views):
            refs = _nearest_ref_views(src_i, c2w_s)
            dirs = ray_dirs_world(K_s[src_i], c2w_s[src_i], vh, vw).to(device=device, dtype=dtype)
            origin = c2w_s[src_i, :3, 3]
            near, far = depth_bounds(c2w_s[src_i], radius, model.half_frac)
            prior_t = zdepth_to_raydist(prior_z[src_i], K_s[src_i]).reshape(vh * vw)
            valid_prior = (prior_z[src_i].reshape(-1) < 1e5) & (masks[src_i, 0].reshape(-1) > 0.25)
            t = (prior_t[:, None] + offsets[None] * interval).clamp(near, far)
            pts = origin.view(1, 1, 3) + dirs[:, None, :] * t[..., None]
            pts_flat = pts.reshape(-1, 3)
            src_rgb = imgs[src_i].permute(1, 2, 0).reshape(vh * vw, 1, 3)
            src_fg = masks[src_i, 0].reshape(vh * vw) > 0.25
            cost = offsets.abs().view(1, -1) * args.guided_plane_sweep_prior_weight
            cost = cost.expand(vh * vw, n_samples).clone()
            n_cost = 0
            for ref_i in refs:
                cam = pts_flat @ w2c_s[ref_i, :3, :3].T + w2c_s[ref_i, :3, 3]
                z = cam[:, 2]
                fx, fy = K_s[ref_i, 0, 0], K_s[ref_i, 1, 1]
                cx, cy = K_s[ref_i, 0, 2], K_s[ref_i, 1, 2]
                u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
                v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
                inb = (z > 1e-6) & (u >= 0) & (u <= vw - 1) & (v >= 0) & (v <= vh - 1)
                grid_x = (u / max(vw - 1, 1)) * 2.0 - 1.0
                grid_y = (v / max(vh - 1, 1)) * 2.0 - 1.0
                grid = torch.stack([grid_x, grid_y], -1).view(1, -1, 1, 2)
                ref_rgb = Fnn.grid_sample(imgs[ref_i:ref_i + 1], grid, mode="bilinear",
                                          padding_mode="zeros", align_corners=True)
                ref_mask = Fnn.grid_sample(masks[ref_i:ref_i + 1], grid, mode="bilinear",
                                           padding_mode="zeros", align_corners=True)
                ref_rgb = ref_rgb.view(3, vh * vw, n_samples).permute(1, 2, 0)
                ref_mask = ref_mask.view(vh * vw, n_samples) * inb.view(vh * vw, n_samples).to(dtype)
                color = (ref_rgb - src_rgb).abs().mean(dim=-1)
                cost = cost + args.plane_sweep_color_weight * color * ref_mask
                cost = cost + args.plane_sweep_mask_weight * (1.0 - ref_mask)
                n_cost += 1
            if n_cost > 0:
                cost = cost / float(n_cost)
            valid = src_fg & valid_prior
            cost = torch.where(valid[:, None], cost, cost.new_full(cost.shape, 1e6))
            best = cost.argmin(dim=1)
            t_hit = t.gather(1, best[:, None]).squeeze(1)
            if (args.guided_plane_sweep_accept_margin >= 0
                    or args.guided_plane_sweep_top2_margin >= 0
                    or args.guided_plane_sweep_max_shift_frac < 1.0):
                prior_i = int(torch.argmin(offsets.abs()).item())
                prior_cost = cost[:, prior_i]
                best_cost = cost.gather(1, best[:, None]).squeeze(1)
                accept = valid & (best != prior_i)
                if args.guided_plane_sweep_accept_margin >= 0:
                    accept = accept & (
                        (prior_cost - best_cost) >= float(args.guided_plane_sweep_accept_margin)
                    )
                if args.guided_plane_sweep_top2_margin >= 0 and n_samples >= 2:
                    top2 = torch.topk(cost, k=2, dim=1, largest=False).values
                    accept = accept & (
                        (top2[:, 1] - top2[:, 0]) >= float(args.guided_plane_sweep_top2_margin)
                    )
                if args.guided_plane_sweep_max_shift_frac < 1.0:
                    max_shift = max(float(args.guided_plane_sweep_max_shift_frac), 0.0)
                    accept = accept & (offsets[best].abs() <= max_shift)
                t_hit = torch.where(accept, t_hit, prior_t)
            pts_hit = origin.view(1, 3) + dirs * t_hit[:, None]
            cam_src = pts_hit @ w2c_s[src_i, :3, :3].T + w2c_s[src_i, :3, 3]
            z_hit = torch.where(valid, cam_src[:, 2], cam_src.new_full((vh * vw,), 1e10))
            z_lr = z_hit.reshape(1, 1, vh, vw)
            z_full = Fnn.interpolate(z_lr, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            out_z.append(z_full)
        return torch.stack(out_z)

    def _confidence_gate(conf: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Robustly map arbitrary positive confidence to [floor, 1] on FG."""
        if args.condition_confidence_power <= 0:
            return torch.ones_like(mask)
        c = conf.to(device=mask.device, dtype=mask.dtype)
        if args.condition_confidence_normalize:
            valid = torch.isfinite(c) & (mask[0] > 0.5)
            norm = _normalize_confidence_map(c, valid)
        else:
            norm = torch.where(torch.isfinite(c), c.clamp(0.0, 1.0), c.new_zeros(c.shape))
        floor = min(max(float(args.condition_confidence_floor), 0.0), 1.0)
        gate = floor + (1.0 - floor) * norm.pow(float(args.condition_confidence_power))
        return gate[None].clamp(0.0, 1.0)

    def _image_cond(frames: torch.Tensor, fg: torch.Tensor, view_idx: int = 0,
                    depths: torch.Tensor | None = None,
                    confs: torch.Tensor | None = None,
                    K_all: torch.Tensor | None = None,
                    c2w_all: torch.Tensor | None = None,
                    w2c_all: torch.Tensor | None = None,
                    radius: float | None = None) -> torch.Tensor | None:
        if not image_cond_active:
            return None
        ref_rgb = frames[view_idx].permute(2, 0, 1)
        ref_fg = fg[view_idx].permute(2, 0, 1)
        raw_fg = ref_fg.clamp(0.0, 1.0)
        if args.condition_rgb_inpaint_px > 0:
            ref_rgb = _condition_rgb_inpaint(
                ref_rgb, raw_fg, int(args.condition_rgb_inpaint_px)
            )
        if args.condition_mask_erode_px > 0:
            k = 2 * int(args.condition_mask_erode_px) + 1
            inv = 1.0 - ref_fg[None].clamp(0.0, 1.0)
            ref_fg = 1.0 - Fnn.max_pool2d(inv, kernel_size=k, stride=1, padding=k // 2)[0]
            ref_fg = ref_fg.clamp(0.0, 1.0)
        if args.condition_mask_blur_px > 0:
            k = 2 * int(args.condition_mask_blur_px) + 1
            ref_fg = Fnn.avg_pool2d(
                Fnn.pad(ref_fg[None], (k // 2, k // 2, k // 2, k // 2), mode="replicate"),
                kernel_size=k,
                stride=1,
            )[0].clamp(0.0, 1.0) * raw_fg
        if (args.condition_confidence_as_mask and args.condition_confidence_power > 0
                and confs is not None):
            ref_fg = ref_fg * _confidence_gate(confs[view_idx], raw_fg)
        if (args.condition_unsharp_amount > 0 or args.condition_contrast != 1.0
                or args.condition_saturation != 1.0):
            ref_rgb = _condition_rgb_preprocess(ref_rgb, ref_fg)
        chans = [ref_rgb * ref_fg, ref_fg]
        if args.image_depth_condition:
            h, w = ref_fg.shape[-2:]
            if depths is None or K_all is None or c2w_all is None or radius is None:
                depth_frac = ref_fg.new_zeros(1, h, w)
                depth_valid = ref_fg.new_zeros(1, h, w)
            else:
                z = depths[view_idx].to(device=frames.device, dtype=frames.dtype)
                t = zdepth_to_raydist(z, K_all[view_idx].to(device=frames.device, dtype=frames.dtype))
                d_near, d_far = depth_bounds(c2w_all[view_idx], radius, model.half_frac)
                denom = max(d_far - d_near, 1e-6)
                depth_frac = ((t - d_near) / denom).clamp(1e-4, 1.0 - 1e-4)[None]
                depth_valid = ((z < 1e5) & (ref_fg[0] > 0.5)).to(frames.dtype)[None]
                depth_frac = depth_frac * depth_valid
            chans.extend([depth_frac, depth_valid])
        if (args.image_visibility_condition or args.image_photo_visibility_condition
                or args.image_confidence_condition):
            h, w = ref_fg.shape[-2:]
            vis = ref_fg.new_ones(1, h, w)
            if args.image_visibility_condition:
                vis = vis * _visibility_condition(view_idx, fg, depths, K_all, c2w_all, w2c_all, radius)
            if args.image_photo_visibility_condition:
                vis = vis * _photo_visibility_condition(view_idx, frames, fg, depths, K_all, c2w_all, w2c_all)
            if args.image_confidence_condition and confs is not None:
                vis = vis * _confidence_gate(confs[view_idx], raw_fg)
            chans.append(vis.clamp(0.0, 1.0))
        if args.image_normal_condition:
            chans.append(_surface_normal_condition(view_idx, fg, depths, K_all, c2w_all))
        return torch.cat(chans, 0)[None]

    def _condition_rgb_inpaint(rgb: torch.Tensor, mask: torch.Tensor, radius_px: int) -> torch.Tensor:
        """Replace composited boundary RGB with nearby interior foreground color.

        The dataset masks are binary but RGB frames are rendered on a white
        background. Antialiased boundary pixels can therefore be foreground in
        the mask while their RGB is mostly background. Direct RGB skips turn
        those pixels into white splats. This fills only the non-core foreground
        band from an eroded interior average.
        """
        r = max(int(radius_px), 1)
        k = 2 * r + 1
        m = mask.clamp(0.0, 1.0)
        inv = 1.0 - m[None]
        core = 1.0 - Fnn.max_pool2d(inv, kernel_size=k, stride=1, padding=r)[0]
        core = core.clamp(0.0, 1.0)
        pad = (r, r, r, r)
        num = Fnn.avg_pool2d(
            Fnn.pad((rgb * core)[None], pad, mode="replicate"),
            kernel_size=k,
            stride=1,
        )[0]
        den = Fnn.avg_pool2d(
            Fnn.pad(core[None], pad, mode="replicate"),
            kernel_size=k,
            stride=1,
        )[0]
        fill = num / den.clamp_min(1e-4)
        boundary = (m > 0.0) & (core < 0.5) & (den > 1e-4)
        return torch.where(boundary.expand_as(rgb), fill, rgb).clamp(0.0, 1.0)

    def _condition_rgb_preprocess(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Cheap decoded-frame enhancement before direct RGB head skip.

        LTX VAE-decoded views are correctly registered after the decode fix, but
        they can be low-pass filtered. This optional prefilter restores some
        edge contrast without changing geometry or adding a train-time dependency.
        """
        x = rgb[None]
        if args.condition_unsharp_amount > 0:
            k = max(1, int(args.condition_unsharp_kernel))
            if k % 2 == 0:
                k += 1
            pad = k // 2
            blur = Fnn.avg_pool2d(Fnn.pad(x, (pad, pad, pad, pad), mode="replicate"),
                                  kernel_size=k, stride=1)
            x = x + args.condition_unsharp_amount * (x - blur)
        if args.condition_contrast != 1.0:
            m = mask[None].clamp(0.0, 1.0)
            denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            mean = (x * m).sum(dim=(2, 3), keepdim=True) / denom
            x = (x - mean) * args.condition_contrast + mean
        if args.condition_saturation != 1.0:
            gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
            x = gray + args.condition_saturation * (x - gray)
        return x[0].clamp(0.0, 1.0)

    def _eroded_condition_mask(mask: torch.Tensor) -> torch.Tensor:
        out = mask.clamp(0.0, 1.0)
        if args.condition_mask_erode_px > 0:
            k = 2 * int(args.condition_mask_erode_px) + 1
            inv = 1.0 - out[None]
            out = 1.0 - Fnn.max_pool2d(inv, kernel_size=k, stride=1, padding=k // 2)[0]
        return out.clamp(0.0, 1.0)

    def _fusion_detail_flat(frame: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        """Per-grid RGB detail used to shrink fused splats on texture edges."""
        if args.fusion_voxel_detail_scale_min >= 1.0:
            return None
        rgb = frame.permute(2, 0, 1)
        fg_m = _eroded_condition_mask(mask.permute(2, 0, 1))
        if (args.condition_unsharp_amount > 0 or args.condition_contrast != 1.0
                or args.condition_saturation != 1.0):
            rgb = _condition_rgb_preprocess(rgb, fg_m)
        gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        dx = Fnn.pad((gray[:, 1:] - gray[:, :-1]).abs(), (0, 1, 0, 0))
        dy = Fnn.pad((gray[1:, :] - gray[:-1, :]).abs(), (0, 0, 0, 1))
        detail = torch.sqrt(dx * dx + dy * dy) * fg_m[0]
        valid = fg_m[0] > 0.5
        if valid.any():
            q = min(max(float(args.fusion_detail_quantile), 0.5), 0.999)
            hi = torch.quantile(detail[valid], q).clamp_min(1e-4)
            detail = (detail / hi).clamp(0.0, 1.0)
        else:
            detail = detail.zero_()
        if args.fusion_detail_power != 1.0:
            detail = detail.pow(max(float(args.fusion_detail_power), 1e-4))
        if detail.shape != (model.map_h, model.map_w):
            detail = Fnn.interpolate(
                detail[None, None], size=(model.map_h, model.map_w),
                mode="bilinear", align_corners=False,
            )[0, 0].clamp(0.0, 1.0)
        return detail.reshape(-1, 1)

    def _depth_frac_valid_factor(depths: torch.Tensor,
                                 fg: torch.Tensor,
                                 K_all: torch.Tensor,
                                 c2w_all: torch.Tensor,
                                 radius: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Depth Z maps -> normalized ray-depth fraction, validity, and ray factor."""
        fracs, valids, factors = [], [], []
        for i in range(depths.shape[0]):
            z = depths[i]
            valid = (
                torch.isfinite(z)
                & (z > 1e-6)
                & (z < 1e5)
                & (fg[i, ..., 0] > 0.5)
            )
            K_i = K_all[i].to(device=z.device, dtype=z.dtype)
            t = zdepth_to_raydist(z, K_i)
            d_near, d_far = depth_bounds(c2w_all[i], radius, model.half_frac)
            denom = max(d_far - d_near, 1e-6)
            frac = ((t - d_near) / denom).clamp(1e-4, 1.0 - 1e-4)
            frac = torch.where(valid, frac, frac.new_full(frac.shape, 1e-4))
            factor = zdepth_to_raydist(torch.ones_like(z), K_i).clamp_min(1e-6)
            fracs.append(frac)
            valids.append(valid.to(dtype=z.dtype))
            factors.append(factor)
        return torch.stack(fracs), torch.stack(valids), torch.stack(factors)

    def _depth_frac_to_z(frac: torch.Tensor,
                         factors: torch.Tensor,
                         c2w_all: torch.Tensor,
                         radius: float) -> torch.Tensor:
        out = []
        for i in range(frac.shape[0]):
            d_near, d_far = depth_bounds(c2w_all[i], radius, model.half_frac)
            t = d_near + frac[i] * (d_far - d_near)
            out.append(t / factors[i].clamp_min(1e-6))
        return torch.stack(out)

    def _fit_condition_depth_calibration() -> torch.Tensor | None:
        if args.condition_depth_calibration == "none":
            return None
        if args.condition_source != "fixed" or not args.cond_depth_subdir:
            print("[phase2] depth calibration skipped: requires fixed non-default conditioning depth",
                  flush=True)
            return None
        max_obj = len(train_ds.entries)
        if args.condition_depth_calib_max_objects > 0:
            max_obj = min(max_obj, args.condition_depth_calib_max_objects)
        ata = torch.zeros(2, 2, dtype=torch.float64)
        atb = torch.zeros(2, 1, dtype=torch.float64)
        raw_abs = 0.0
        n_px = 0
        n_views = 0
        for entry in train_ds.entries[:max_obj]:
            obj_dir = object_dir_for_entry(V4, entry)
            try:
                cams = load_cameras(obj_dir / "cameras.json")
                idxs = resolve_view_spec(
                    cond_spec,
                    cams["w2c"].shape[0],
                    obj_dir=obj_dir,
                    subdir=args.cond_subdir,
                    n_orbit_views=cams["num_orbit_views"],
                    default_n=cond_default_views,
                )
                if idxs is None:
                    continue
                if args.condition_depth_calib_views > 0:
                    idxs = idxs[:args.condition_depth_calib_views]
                cond_fg = load_masks_at(obj_dir, idxs)
                cond_depth = torch.stack([
                    load_depth_view_at(obj_dir, i, subdir=args.cond_depth_subdir)
                    for i in idxs
                ])
                target_depth = torch.stack([load_depth_view(obj_dir, i) for i in idxs])
                cond_K = cams["K"][torch.as_tensor(idxs, dtype=torch.long)]
                cond_c2w = cams["c2w_opengl"][torch.as_tensor(idxs, dtype=torch.long)]
                radius_i = float(cams["radius"])
            except Exception as ex:
                print(
                    f"[phase2] depth calibration skip {uid[:10]} "
                    f"({type(ex).__name__})",
                    flush=True,
                )
                continue
            depth_frac, depth_valid, _ = _depth_frac_valid_factor(
                cond_depth.float(), cond_fg.float(), cond_K.float(), cond_c2w.float(), radius_i
            )
            target_frac, target_valid, _ = _depth_frac_valid_factor(
                target_depth.float(), cond_fg.float(), cond_K.float(), cond_c2w.float(), radius_i
            )
            valid = (depth_valid > 0.5) & (target_valid > 0.5)
            if not valid.any():
                continue
            x = depth_frac[valid].to(torch.float64)
            y = target_frac[valid].to(torch.float64)
            if args.condition_depth_calib_sample_px > 0 and x.shape[0] > args.condition_depth_calib_sample_px:
                stride = math.ceil(x.shape[0] / args.condition_depth_calib_sample_px)
                x = x[::stride]
                y = y[::stride]
            feat = torch.stack([x, torch.ones_like(x)], dim=1)
            ata += feat.T @ feat
            atb += feat.T @ y[:, None]
            raw_abs += float((x - y).abs().sum())
            n_px += int(x.shape[0])
            n_views += len(idxs)
        if n_px < 16:
            print("[phase2] depth calibration skipped: not enough foreground pixels",
                  flush=True)
            return None
        ridge = max(float(args.condition_depth_calib_ridge), 0.0)
        reg = torch.eye(2, dtype=torch.float64) * ridge
        reg[-1, -1] = 0.0
        sol = torch.linalg.solve(ata + reg, atb).reshape(2).to(torch.float32)
        raw_mae = raw_abs / max(n_px, 1)
        print(
            "[phase2] depth calibration train_affine_frac "
            f"objs={max_obj} views={n_views} px={n_px} "
            f"a={float(sol[0]):.6f} b={float(sol[1]):.6f} raw_frac_mae={raw_mae:.6f}",
            flush=True,
        )
        return sol

    depth_calib = _fit_condition_depth_calibration()

    def _apply_condition_depth_calibration(depths: torch.Tensor | None,
                                           fg: torch.Tensor,
                                           K_all: torch.Tensor,
                                           c2w_all: torch.Tensor,
                                           radius: float) -> torch.Tensor | None:
        if depth_calib is None or depths is None:
            return depths
        depth_frac, depth_valid, factors = _depth_frac_valid_factor(
            depths, fg, K_all, c2w_all, radius
        )
        sol = depth_calib.to(device=depths.device, dtype=depths.dtype)
        calibrated_frac = (depth_frac * sol[0] + sol[1]).clamp(1e-4, 1.0 - 1e-4)
        calibrated_z = _depth_frac_to_z(calibrated_frac, factors, c2w_all, radius)
        return torch.where(depth_valid > 0.5, calibrated_z, depths)

    def _apply_condition_depth_median_cleanup(depths: torch.Tensor | None,
                                              fg: torch.Tensor,
                                              K_all: torch.Tensor,
                                              c2w_all: torch.Tensor,
                                              radius: float) -> torch.Tensor | None:
        radius_px = max(int(args.condition_depth_median_radius_px), 0)
        if radius_px <= 0 or depths is None:
            return depths
        depth_frac, depth_valid, factors = _depth_frac_valid_factor(
            depths, fg, K_all, c2w_all, radius
        )
        median_frac, median_valid = _local_valid_median_map(
            depth_frac,
            depth_valid > 0.5,
            radius_px,
        )
        thresh = max(float(args.condition_depth_median_thresh_frac), 0.0)
        replace = (depth_valid > 0.5) & median_valid
        if thresh > 0:
            replace = replace & ((depth_frac - median_frac).abs() > thresh)
        mix = min(max(float(args.condition_depth_median_mix), 0.0), 1.0)
        clean_frac = torch.where(
            replace,
            depth_frac * (1.0 - mix) + median_frac * mix,
            depth_frac,
        ).clamp(1e-4, 1.0 - 1e-4)
        clean_z = _depth_frac_to_z(clean_frac, factors, c2w_all, radius)
        return torch.where(replace, clean_z, depths)

    def _refine_condition_depths(frames: torch.Tensor,
                                 fg: torch.Tensor,
                                 depths: torch.Tensor | None,
                                 K_all: torch.Tensor,
                                 c2w_all: torch.Tensor,
                                 radius: float,
                                 target_depths: torch.Tensor | None = None) -> torch.Tensor | None:
        depths = _apply_condition_depth_calibration(depths, fg, K_all, c2w_all, radius)
        depths = _apply_condition_depth_median_cleanup(depths, fg, K_all, c2w_all, radius)
        if depth_refine_head is None or depths is None:
            return depths
        depth_frac, depth_valid, factors = _depth_frac_valid_factor(
            depths, fg, K_all, c2w_all, radius
        )
        depth_apply_valid = depth_valid
        if args.depth_refine_apply_erode_px > 0:
            depth_apply_valid = depth_apply_valid * _erode_mask_2d(
                fg[..., 0].clamp(0.0, 1.0),
                args.depth_refine_apply_erode_px,
            )
        rgb = frames.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        mask = fg.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        feat = torch.cat([
            rgb * mask,
            mask,
            depth_frac[:, None],
            depth_valid[:, None],
        ], dim=1)
        mv_feat = None
        if args.depth_refine_multiview_features:
            mv_feat = _depth_multiview_support_maps(
                depths,
                fg,
                K_all,
                c2w_all,
                radius,
                args.depth_refine_multiview_tol_frac,
                args.depth_refine_multiview_refs,
                args.depth_refine_multiview_radius_px,
            )
            feat = torch.cat([feat, mv_feat], dim=1)
        if mv_feat is not None:
            conflict_min = float(args.depth_refine_apply_mv_conflict_min)
            support_max = float(args.depth_refine_apply_mv_support_max)
            coverage_min = float(args.depth_refine_apply_mv_coverage_min)
            if conflict_min > 0.0 or support_max < 1.0 or coverage_min > 0.0:
                suspicious = torch.ones_like(depth_apply_valid, dtype=torch.bool)
                if conflict_min > 0.0:
                    conflict = torch.maximum(mv_feat[:, 1], mv_feat[:, 2])
                    suspicious = suspicious & (conflict >= conflict_min)
                if support_max < 1.0:
                    suspicious = suspicious & (mv_feat[:, 0] <= support_max)
                if coverage_min > 0.0:
                    suspicious = suspicious & (mv_feat[:, 3] >= coverage_min)
                depth_apply_valid = depth_apply_valid * suspicious.to(dtype=depth_apply_valid.dtype)
        if args.depth_refine_detach_inputs:
            feat = feat.detach()

        def _run_depth_refine(x: torch.Tensor) -> torch.Tensor:
            if (args.depth_refine_checkpoint
                    and torch.is_grad_enabled()
                    and depth_refine_head.training):
                from torch.utils.checkpoint import checkpoint

                return checkpoint(depth_refine_head, x, use_reentrant=False)
            return depth_refine_head(x)

        chunk_views = max(int(args.depth_refine_chunk_views), 0)
        if chunk_views > 0 and feat.shape[0] > chunk_views:
            delta = torch.cat([
                _run_depth_refine(feat[start:start + chunk_views])
                for start in range(0, feat.shape[0], chunk_views)
            ], dim=0)
        else:
            delta = _run_depth_refine(feat)
        delta_scale = max(float(args.depth_refine_delta_scale), 0.0)
        if delta_scale > 0:
            delta = delta_scale * torch.tanh(delta)
        else:
            delta = delta * 0.0
        prior_logit = torch.logit(depth_frac[:, None].clamp(1e-4, 1.0 - 1e-4))
        refined_frac = torch.sigmoid(prior_logit + delta)[:, 0]
        refined_frac = torch.where(depth_apply_valid > 0.5, refined_frac, depth_frac)
        refined_z = _depth_frac_to_z(refined_frac, factors, c2w_all, radius)
        refined = torch.where(depth_apply_valid > 0.5, refined_z, depths)

        if torch.is_grad_enabled():
            depth_refine_delta_terms.append((delta * depth_apply_valid[:, None]).square().sum() /
                                            depth_apply_valid.sum().clamp_min(1.0))
            if delta.shape[-1] > 1 and delta.shape[-2] > 1:
                valid_x = depth_apply_valid[:, :, 1:] * depth_apply_valid[:, :, :-1]
                valid_y = depth_apply_valid[:, 1:, :] * depth_apply_valid[:, :-1, :]
                tv_x = ((delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs()
                        * valid_x[:, None]).sum() / valid_x.sum().clamp_min(1.0)
                tv_y = ((delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs()
                        * valid_y[:, None]).sum() / valid_y.sum().clamp_min(1.0)
                depth_refine_tv_terms.append(0.5 * (tv_x + tv_y))
            if target_depths is not None:
                tgt_frac, tgt_valid, _ = _depth_frac_valid_factor(
                    target_depths.to(device=depths.device, dtype=depths.dtype),
                    fg, K_all, c2w_all, radius
                )
                valid = (tgt_valid > 0.5) & (depth_apply_valid > 0.5)
                if valid.any():
                    loss_px = Fnn.huber_loss(
                        refined_frac[valid], tgt_frac[valid], delta=0.02,
                        reduction="none",
                    )
                    outlier_weight = max(float(args.depth_refine_gt_outlier_weight), 0.0)
                    if outlier_weight > 0:
                        prior_err = (depth_frac[valid] - tgt_frac[valid]).abs().detach()
                        power = max(float(args.depth_refine_gt_outlier_power), 1e-6)
                        w_gt = 1.0 + outlier_weight * (prior_err / 0.02).clamp(0.0, 20.0).pow(power)
                        loss_px = loss_px * w_gt / w_gt.mean().clamp_min(1e-6)
                    depth_refine_gt_terms.append(loss_px.mean())
                    if args.depth_refine_metric_gt_weight > 0:
                        spans = []
                        for view_j in range(refined_frac.shape[0]):
                            d_near, d_far = depth_bounds(c2w_all[view_j], radius, model.half_frac)
                            spans.append((d_far - d_near) / max(float(radius), 1e-6))
                        span = refined_frac.new_tensor(spans)[:, None, None]
                        metric_err = (refined_frac - tgt_frac) * span
                        metric_prior_err = ((depth_frac - tgt_frac) * span).abs().detach()
                        delta_m = max(float(args.depth_refine_metric_delta_frac), 1e-6)
                        metric_loss = Fnn.huber_loss(
                            metric_err[valid],
                            torch.zeros_like(metric_err[valid]),
                            delta=delta_m,
                            reduction="none",
                        )
                        w_metric = torch.ones_like(metric_loss)
                        if outlier_weight > 0:
                            power = max(float(args.depth_refine_gt_outlier_power), 1e-6)
                            w_metric = w_metric + outlier_weight * (
                                metric_prior_err[valid] / delta_m
                            ).clamp(0.0, 20.0).pow(power)
                        conflict_weight = max(float(args.depth_refine_conflict_weight), 0.0)
                        if conflict_weight > 0 and mv_feat is not None:
                            conflict_score = torch.maximum(
                                mv_feat[:, 1].clamp(0.0, 1.0),
                                mv_feat[:, 2].clamp(0.0, 1.0),
                            )
                            w_metric = w_metric + conflict_weight * conflict_score[valid]
                        metric_loss = metric_loss * w_metric / w_metric.mean().clamp_min(1e-6)
                        depth_refine_metric_gt_terms.append(metric_loss.mean())
        return refined

    def _load_depths_for_indices(obj_dir: str, idxs: list[int], h: int, w: int,
                                 device: torch.device, dtype: torch.dtype,
                                 subdir: str | None = None,
                                 fallback: torch.Tensor | None = None) -> torch.Tensor:
        out = []
        depth_subdir = args.cond_depth_subdir if subdir is None else subdir
        for i in idxs:
            try:
                out.append(load_depth_view_at(
                    Path(obj_dir), i, subdir=depth_subdir
                ).to(device=device, dtype=dtype))
            except (FileNotFoundError, IndexError, ValueError) as ex:
                if depth_subdir and args.strict_condition_depth and fallback is None:
                    raise FileNotFoundError(
                        f"missing conditioning depth obj={obj_dir} view={i:03d} "
                        f"subdir={depth_subdir!r}"
                    ) from ex
                if fallback is not None:
                    out.append(fallback[len(out)].to(device=device, dtype=dtype))
                else:
                    out.append(torch.zeros(h, w, device=device, dtype=dtype))
        return torch.stack(out)

    def _load_confs_for_indices(obj_dir: str, idxs: list[int], h: int, w: int,
                                device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if not args.cond_conf_subdir:
            return None
        out = []
        for i in idxs:
            try:
                out.append(load_conf_view_at(
                    Path(obj_dir), i, subdir=args.cond_conf_subdir
                ).to(device=device, dtype=dtype))
            except (FileNotFoundError, IndexError, ValueError):
                out.append(torch.ones(h, w, device=device, dtype=dtype))
        return torch.stack(out)

    def _condition_bundle(sample: dict, frames: torch.Tensor, fg: torch.Tensor,
                          w2c: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor,
                          depths: torch.Tensor | None) -> dict:
        """Return the anchor/conditioning views, possibly separate from targets."""
        if args.condition_source != "fixed":
            cond_fg = _apply_condition_mask_source(frames, fg.float())
            cond_frames, cond_depths_raw = _apply_condition_rgbd_refine(
                frames, cond_fg, depths, K, c2w, float(sample["radius"]),
                target_frames=frames, target_depths=depths,
            )
            cond_depths = _refine_condition_depths(
                cond_frames, cond_fg, cond_depths_raw, K, c2w, float(sample["radius"]),
                target_depths=depths,
            )
            cond_depths = _apply_condition_depth_affine(
                cond_frames, cond_fg, cond_depths, K, c2w, float(sample["radius"]),
                target_depths=depths,
            )
            cond_confs = _apply_condition_depth_confidence(
                cond_frames, cond_fg, cond_depths, K, c2w, float(sample["radius"]),
                target_depths=depths, base_confs=None,
            )
            cond_c2w, cond_w2c, pose_metrics = _apply_condition_pose_camera(
                cond_frames, cond_fg, cond_depths, K, c2w, w2c,
                float(sample["radius"])
            )
            return {
                "frames": cond_frames,
                "fg": cond_fg,
                "target_frames": frames,
                "w2c": cond_w2c,
                "K": K,
                "c2w": cond_c2w,
                "depths": cond_depths,
                "visibility_depths": cond_depths,
                "target_depths": depths,
                "confs": cond_confs,
                "pose_metrics": pose_metrics,
            }
        if "cond_frames" in sample:
            cond_frames = sample["cond_frames"].to(device=frames.device, dtype=frames.dtype)
            cond_target_frames = sample.get("cond_target_frames", None)
            cond_target_frames = (
                cond_target_frames.to(device=frames.device, dtype=frames.dtype)
                if cond_target_frames is not None else None
            )
            cond_fg = sample["cond_masks"].to(device=frames.device, dtype=frames.dtype)
            cond_fg = _apply_condition_mask_source(cond_frames, cond_fg)
            cond_K = sample["cond_K"].to(device=frames.device, dtype=frames.dtype)
            cond_c2w = sample["cond_c2w_opengl"].to(device=frames.device, dtype=frames.dtype)
            cond_depths_raw = sample["cond_depths"].to(device=frames.device, dtype=frames.dtype)
            cond_visibility_depths = sample.get("cond_visibility_depths", None)
            cond_visibility_depths = (
                cond_visibility_depths.to(device=frames.device, dtype=frames.dtype)
                if cond_visibility_depths is not None else None
            )
            target_depths = sample.get("cond_target_depths", None)
            target_depths = (
                target_depths.to(device=frames.device, dtype=frames.dtype)
                if target_depths is not None else cond_depths_raw
            )
            # GREAT-QUALITY MODE: use GT depth at the conditioning views as the
            # model input.  The prior at that input config reaches ~20 dB on v7
            # (the documented "oracle" ceiling).  Trades the original "no GT
            # at inference" constraint for actual great visual quality.
            if getattr(args, "cond_use_target_depth", 0):
                cond_depths_raw = target_depths
                cond_visibility_depths = target_depths
            cond_frames, cond_depths_raw = _apply_condition_rgbd_refine(
                cond_frames, cond_fg, cond_depths_raw, cond_K, cond_c2w,
                float(sample["radius"]), target_frames=cond_target_frames,
                target_depths=target_depths,
            )
            cond_frames = _apply_condition_rgb_refine(cond_frames, cond_fg)
            cond_frames = _apply_condition_color_calibration(cond_frames)
            _append_condition_rgb_refine_gt_loss(cond_frames, cond_target_frames, cond_fg)
            cond_depths = _refine_condition_depths(
                cond_frames, cond_fg, cond_depths_raw, cond_K, cond_c2w,
                float(sample["radius"]), target_depths=target_depths,
            )
            cond_depths = _apply_condition_depth_affine(
                cond_frames, cond_fg, cond_depths, cond_K, cond_c2w,
                float(sample["radius"]), target_depths=target_depths,
            )
            cond_confs = _apply_condition_depth_confidence(
                cond_frames, cond_fg, cond_depths, cond_K, cond_c2w,
                float(sample["radius"]), target_depths=target_depths,
                base_confs=sample.get("cond_confs", None).to(
                    device=frames.device, dtype=frames.dtype
                ) if sample.get("cond_confs", None) is not None else None,
            )
            cond_w2c = sample["cond_w2c"].to(device=frames.device, dtype=frames.dtype)
            cond_c2w_out, cond_w2c_out, pose_metrics = _apply_condition_pose_camera(
                cond_frames, cond_fg, cond_depths, cond_K, cond_c2w, cond_w2c,
                float(sample["radius"])
            )
            return {
                "frames": cond_frames,
                "fg": cond_fg,
                "target_frames": (
                    cond_target_frames if cond_target_frames is not None else cond_frames
                ),
                "w2c": cond_w2c_out,
                "K": cond_K,
                "c2w": cond_c2w_out,
                "depths": cond_depths,
                "visibility_depths": (
                    cond_visibility_depths if cond_visibility_depths is not None else cond_depths
                ),
                "target_depths": target_depths,
                "confs": cond_confs,
                "pose_metrics": pose_metrics,
            }

        obj_dir = sample.get("obj_dir")
        if obj_dir is None:
            raise ValueError("--condition_source fixed requires sample['obj_dir']")
        cond_cam_K = sample.get("all_K", K).to(device=K.device, dtype=K.dtype)
        cond_cam_c2w = sample.get("all_c2w_opengl", c2w).to(device=c2w.device, dtype=c2w.dtype)
        cond_cam_w2c = sample.get("all_w2c", w2c).to(device=w2c.device, dtype=w2c.dtype)
        spec = args.cond_view_indices
        if spec == "" and args.cond_subdir:
            spec = "available"
            default_n = None
        elif spec == "":
            default_n = args.anchor_views
        else:
            default_n = None
        idxs = resolve_view_spec(
            spec,
            cond_cam_K.shape[0],
            obj_dir=obj_dir,
            subdir=args.cond_subdir,
            n_orbit_views=int(sample.get("num_orbit_views", cond_cam_K.shape[0])),
            default_n=default_n,
        )
        if idxs is None:
            raise ValueError("--condition_source fixed resolved no conditioning views")
        sel = torch.as_tensor(idxs, device=cond_cam_K.device, dtype=torch.long)
        h, w = frames.shape[1], frames.shape[2]
        cond_depths = None
        if (args.cond_depth_subdir is None
                and depths is not None and depths.shape[0] > int(sel.max())):
            cond_depths = depths[sel]
        else:
            cond_depths = _load_depths_for_indices(obj_dir, idxs, h, w, frames.device, frames.dtype)
        cond_visibility_depths = None
        if args.cond_visibility_depth_subdir:
            cond_visibility_depths = _load_depths_for_indices(
                obj_dir, idxs, h, w, frames.device, frames.dtype,
                subdir=args.cond_visibility_depth_subdir,
                fallback=cond_depths,
            )
        cond_target_depths = depths[sel] if depths is not None and depths.shape[0] > int(sel.max()) else None
        cond_confs = _load_confs_for_indices(obj_dir, idxs, h, w, frames.device, frames.dtype)
        cond_fg = load_masks_at(obj_dir, idxs).to(device=frames.device, dtype=frames.dtype)
        cond_frames = load_views_at(obj_dir, idxs, subdir=args.cond_subdir).to(
            device=frames.device, dtype=frames.dtype
        )
        cond_target_frames = load_views_at(obj_dir, idxs, subdir=None).to(
            device=frames.device, dtype=frames.dtype
        )
        cond_fg = _apply_condition_mask_source(cond_frames, cond_fg)
        cond_K = cond_cam_K[sel]
        cond_c2w = cond_cam_c2w[sel]
        cond_frames, cond_depths = _apply_condition_rgbd_refine(
            cond_frames, cond_fg, cond_depths, cond_K, cond_c2w,
            float(sample["radius"]), target_frames=cond_target_frames,
            target_depths=cond_target_depths,
        )
        cond_frames = _apply_condition_rgb_refine(cond_frames, cond_fg)
        cond_frames = _apply_condition_color_calibration(cond_frames)
        _append_condition_rgb_refine_gt_loss(cond_frames, cond_target_frames, cond_fg)
        cond_depths = _refine_condition_depths(
            cond_frames, cond_fg, cond_depths, cond_K, cond_c2w, float(sample["radius"]),
            target_depths=cond_target_depths,
        )
        cond_depths = _apply_condition_depth_affine(
            cond_frames, cond_fg, cond_depths, cond_K, cond_c2w, float(sample["radius"]),
            target_depths=cond_target_depths,
        )
        cond_confs = _apply_condition_depth_confidence(
            cond_frames, cond_fg, cond_depths, cond_K, cond_c2w, float(sample["radius"]),
            target_depths=cond_target_depths, base_confs=cond_confs,
        )
        cond_w2c = cond_cam_w2c[sel]
        cond_c2w_out, cond_w2c_out, pose_metrics = _apply_condition_pose_camera(
            cond_frames, cond_fg, cond_depths, cond_K, cond_c2w, cond_w2c,
            float(sample["radius"])
        )
        return {
            "frames": cond_frames,
            "fg": cond_fg,
            "target_frames": cond_target_frames,
            "w2c": cond_w2c_out,
            "K": cond_K,
            "c2w": cond_c2w_out,
            "depths": cond_depths,
            "visibility_depths": (
                cond_visibility_depths if cond_visibility_depths is not None else cond_depths
            ),
            "target_depths": cond_target_depths,
            "confs": cond_confs,
            "pose_metrics": pose_metrics,
        }

    def _eval_anchor_indices(n_views: int, n_anchor: int) -> list[int]:
        n_anchor = min(max(n_anchor, 1), n_views)
        if n_anchor == 1:
            return [0]
        # Non-endpoint spacing avoids picking both frame 0 and the loop-closing
        # final frame on 49-view orbits.
        idx = torch.arange(n_anchor, dtype=torch.float32) * (float(n_views) / n_anchor)
        return idx.round().long().clamp_max(n_views - 1).unique().tolist()

    def _cat_params(parts: list[dict]) -> dict:
        out = {}
        for k in parts[0]:
            vals = [p[k] for p in parts if k in p]
            if len(vals) != len(parts):
                continue
            if torch.is_tensor(vals[0]) and vals[0].ndim == 0:
                out[k] = torch.stack(vals, 0)
            else:
                out[k] = torch.cat(vals, 0)
        return out

    def _filter_param_keep(part: dict, keep: torch.Tensor) -> dict:
        n = part["mean"].shape[0]
        out = {}
        for k, v in part.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == n:
                out[k] = v[keep]
            else:
                out[k] = v
        return out

    def _param_scalar_mean(params: dict, key: str) -> torch.Tensor | None:
        v = params.get(key)
        if v is None:
            return None
        return v.float().mean()

    def _quat_local_z(q: torch.Tensor) -> torch.Tensor:
        q = Fnn.normalize(q, dim=-1)
        wq, xq, yq, zq = q.unbind(-1)
        return torch.stack([
            2.0 * (xq * zq + wq * yq),
            2.0 * (yq * zq - wq * xq),
            1.0 - 2.0 * (xq * xq + yq * yq),
        ], dim=-1)

    def _render_params(params: dict, w2c: torch.Tensor, K: torch.Tensor,
                       width: int, height: int, bg: float,
                       sh_degree: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not args.view_opacity_gate:
            return render_views(
                params, w2c, K, width, height, bg=bg,
                eps2d=args.render_eps2d, rasterize_mode=args.rasterize_mode,
                sh_degree=sh_degree,
            )
        floor = min(max(float(args.view_opacity_floor), 0.0), 1.0)
        power = max(float(args.view_opacity_power), 1e-6)
        normals = _quat_local_z(params["quat"]).to(
            device=params["mean"].device, dtype=params["mean"].dtype
        )
        if args.view_opacity_flip:
            normals = -normals
        renders, alphas = [], []
        for vi in range(w2c.shape[0]):
            R = w2c[vi, :3, :3].to(device=params["mean"].device, dtype=params["mean"].dtype)
            t = w2c[vi, :3, 3].to(device=params["mean"].device, dtype=params["mean"].dtype)
            center = -(R.T @ t)
            view_dir = Fnn.normalize(center[None] - params["mean"], dim=-1)
            facing = (normals * view_dir).sum(dim=-1, keepdim=True).clamp_min(0.0)
            gate = floor + (1.0 - floor) * facing.pow(power)
            p_view = dict(params)
            p_view["opacity"] = params["opacity"] * gate.to(dtype=params["opacity"].dtype)
            r_i, a_i = render_views(
                p_view, w2c[vi:vi + 1], K[vi:vi + 1], width, height, bg=bg,
                eps2d=args.render_eps2d, rasterize_mode=args.rasterize_mode,
                sh_degree=sh_degree,
            )
            renders.append(r_i)
            alphas.append(a_i)
        return torch.cat(renders, 0), torch.cat(alphas, 0)

    learned_fill_delta_terms: list[torch.Tensor] = []
    learned_fill_tv_terms: list[torch.Tensor] = []
    learned_fill_oracle_terms: list[torch.Tensor] = []
    depth_refine_delta_terms: list[torch.Tensor] = []
    depth_refine_tv_terms: list[torch.Tensor] = []
    depth_refine_gt_terms: list[torch.Tensor] = []
    depth_refine_metric_gt_terms: list[torch.Tensor] = []
    support_gate_delta_terms: list[torch.Tensor] = []
    support_gate_tv_terms: list[torch.Tensor] = []
    support_gate_gt_terms: list[torch.Tensor] = []
    surface_confidence_delta_terms: list[torch.Tensor] = []
    surface_confidence_tv_terms: list[torch.Tensor] = []
    surface_confidence_gt_terms: list[torch.Tensor] = []
    surface_refine_delta_terms: list[torch.Tensor] = []
    surface_refine_tv_terms: list[torch.Tensor] = []
    surface_refine_rgb_gt_terms: list[torch.Tensor] = []
    surface_refine_rgb_grad_gt_terms: list[torch.Tensor] = []
    fusion_candidate_delta_terms: list[torch.Tensor] = []
    fusion_candidate_gt_terms: list[torch.Tensor] = []
    output_alpha_refine_delta_terms: list[torch.Tensor] = []
    output_alpha_refine_tv_terms: list[torch.Tensor] = []
    condition_rgb_refine_gt_terms: list[torch.Tensor] = []
    condition_rgbd_refine_delta_terms: list[torch.Tensor] = []
    condition_rgbd_refine_tv_terms: list[torch.Tensor] = []
    condition_rgbd_refine_rgb_gt_terms: list[torch.Tensor] = []
    condition_rgbd_refine_depth_gt_terms: list[torch.Tensor] = []
    condition_pose_center_terms: list[torch.Tensor] = []
    condition_pose_forward_terms: list[torch.Tensor] = []
    condition_pose_dist_terms: list[torch.Tensor] = []
    condition_depth_affine_delta_terms: list[torch.Tensor] = []
    condition_depth_affine_gt_terms: list[torch.Tensor] = []
    condition_depth_confidence_delta_terms: list[torch.Tensor] = []
    condition_depth_confidence_tv_terms: list[torch.Tensor] = []
    condition_depth_confidence_gt_terms: list[torch.Tensor] = []

    def _depth_gate_features(frames_i: torch.Tensor,
                             fg_i: torch.Tensor,
                             depth_i: torch.Tensor,
                             K_i: torch.Tensor,
                             c2w_i: torch.Tensor,
                             radius: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth_frac, depth_valid, _ = _depth_frac_valid_factor(
            depth_i[None], fg_i[None], K_i[None], c2w_i[None], radius
        )
        rgb = frames_i.permute(2, 0, 1).clamp(0.0, 1.0)
        mask = fg_i.permute(2, 0, 1).clamp(0.0, 1.0)
        feat = torch.cat([
            rgb * mask,
            mask,
            depth_frac[:, None][0],
            depth_valid[:, None][0],
        ], dim=0)[None]
        return feat, depth_frac[0], depth_valid[0]

    def _support_gate_target(depth_i: torch.Tensor,
                             target_depth_i: torch.Tensor | None,
                             fg_i: torch.Tensor,
                             K_i: torch.Tensor,
                             radius: float,
                             tol_frac: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        valid_prior = (
            torch.isfinite(depth_i)
            & (depth_i > 1e-6)
            & (depth_i < 1e5)
            & (fg_i[..., 0] > 0.5)
        )
        if target_depth_i is None:
            return valid_prior.to(dtype=depth_i.dtype), valid_prior
        gt = target_depth_i.to(device=depth_i.device, dtype=depth_i.dtype)
        valid_gt = torch.isfinite(gt) & (gt > 1e-6) & (gt < 1e5) & (fg_i[..., 0] > 0.5)
        valid = valid_prior & valid_gt
        if not valid.any():
            return valid_prior.to(dtype=depth_i.dtype), valid_prior
        t_prior = zdepth_to_raydist(depth_i, K_i.to(device=depth_i.device, dtype=depth_i.dtype))
        t_gt = zdepth_to_raydist(gt, K_i.to(device=depth_i.device, dtype=depth_i.dtype))
        use_tol_frac = args.support_gate_depth_tol_frac if tol_frac is None else tol_frac
        tol = max(float(use_tol_frac) * float(radius), 1e-6)
        err = ((t_prior - t_gt).abs() / tol).clamp(0.0, 12.0)
        target = torch.exp(-0.5 * err.square()).to(dtype=depth_i.dtype)
        target = torch.where(valid, target, target.new_zeros(()))
        return target, valid

    def _support_gate_multiview_counts(view_i: int,
                                       depth_i: torch.Tensor,
                                       target_depths: torch.Tensor | None,
                                       fg: torch.Tensor,
                                       K_all: torch.Tensor,
                                       c2w_all: torch.Tensor,
                                       w2c_all: torch.Tensor | None,
                                       radius: float,
                                       tol_frac: float | None = None,
                                       max_refs: int | None = None) -> tuple[
                                           torch.Tensor, torch.Tensor, torch.Tensor,
                                           torch.Tensor, torch.Tensor
                                       ]:
        dtype, device = depth_i.dtype, depth_i.device
        h, w = depth_i.shape
        fg_i = fg[view_i].to(device=device)
        valid_prior = (
            torch.isfinite(depth_i)
            & (depth_i > 1e-6)
            & (depth_i < 1e5)
            & (fg_i[..., 0] > 0.5)
        )
        supports = torch.zeros(h * w, device=device, dtype=dtype)
        conflicts = torch.zeros(h * w, device=device, dtype=dtype)
        front_conflicts = torch.zeros(h * w, device=device, dtype=dtype)
        coverage = torch.zeros(h * w, device=device, dtype=dtype)
        if target_depths is None or w2c_all is None or view_i >= target_depths.shape[0]:
            return (
                supports.reshape(h, w),
                conflicts.reshape(h, w),
                front_conflicts.reshape(h, w),
                coverage.reshape(h, w),
                valid_prior,
            )
        valid_flat = valid_prior.reshape(-1)
        if not valid_flat.any():
            return (
                supports.reshape(h, w),
                conflicts.reshape(h, w),
                front_conflicts.reshape(h, w),
                coverage.reshape(h, w),
                valid_prior,
            )

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        K_i = K_all[view_i].to(device=device, dtype=dtype)
        z_flat = depth_i.reshape(-1)
        x = (xx.reshape(-1) - K_i[0, 2]) / K_i[0, 0].clamp_min(1e-6) * z_flat
        y = (yy.reshape(-1) - K_i[1, 2]) / K_i[1, 1].clamp_min(1e-6) * z_flat
        pts_cam = torch.stack([x, y, z_flat], dim=-1)
        c2w_i = c2w_all[view_i].to(device=device, dtype=dtype)
        pts = pts_cam @ c2w_i[:3, :3].T + c2w_i[:3, 3]

        n_ref = min(
            int(target_depths.shape[0]),
            int(fg.shape[0]),
            int(K_all.shape[0]),
            int(w2c_all.shape[0]),
        )
        refs = torch.arange(n_ref, device=device, dtype=torch.long)
        ref_limit = int(args.support_gate_multiview_refs if max_refs is None else max_refs)
        if ref_limit > 0 and n_ref > ref_limit:
            centers = c2w_all[:n_ref, :3, 3].to(device=device, dtype=dtype)
            cur = c2w_all[view_i, :3, 3].to(device=device, dtype=dtype)
            refs = torch.linalg.norm(centers - cur[None], dim=1).argsort()[:ref_limit]

        use_tol_frac = args.support_gate_depth_tol_frac if tol_frac is None else tol_frac
        tol = max(float(use_tol_frac) * float(radius), 1e-6)
        K_i_full = K_all[view_i].to(device=device, dtype=dtype)

        for ref_view_t in refs:
            ref_view = int(ref_view_t.item())
            if ref_view == view_i:
                gt = target_depths[ref_view].to(device=device, dtype=dtype)
                fg_ref = fg[ref_view].to(device=device)[..., 0] > 0.5
                valid_gt = torch.isfinite(gt) & (gt > 1e-6) & (gt < 1e5) & fg_ref
                valid = valid_prior & valid_gt
                if not valid.any():
                    continue
                t_prior = zdepth_to_raydist(depth_i, K_i_full)
                t_gt = zdepth_to_raydist(gt, K_i_full)
                match = valid & ((t_prior - t_gt).abs() <= tol)
                valid_idx = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
                match_idx = torch.nonzero(match.reshape(-1), as_tuple=False).squeeze(1)
                coverage[valid_idx] += 1.0
                if match_idx.numel() > 0:
                    supports[match_idx] += 1.0
                nonmatch_idx = torch.nonzero((valid & ~match).reshape(-1),
                                              as_tuple=False).squeeze(1)
                if nonmatch_idx.numel() > 0:
                    conflicts[nonmatch_idx] += 1.0
                continue

            w2c_ref = w2c_all[ref_view].to(device=device, dtype=dtype)
            cam = pts @ w2c_ref[:3, :3].T + w2c_ref[:3, 3]
            z = cam[:, 2]
            K_ref = K_all[ref_view].to(device=device, dtype=dtype)
            u = K_ref[0, 0] * (cam[:, 0] / z.clamp_min(1e-6)) + K_ref[0, 2]
            v = K_ref[1, 1] * (cam[:, 1] / z.clamp_min(1e-6)) + K_ref[1, 2]
            inb = (
                valid_flat
                & (z > 1e-6)
                & (u >= 0) & (u <= w - 1)
                & (v >= 0) & (v <= h - 1)
            )
            if not inb.any():
                continue
            idx = torch.nonzero(inb, as_tuple=False).squeeze(1)
            ui = u[idx].round().long().clamp(0, w - 1)
            vi = v[idx].round().long().clamp(0, h - 1)

            fg_ref = fg[ref_view].to(device=device)[..., 0] > 0.5
            gt = target_depths[ref_view].to(device=device, dtype=dtype)
            valid_gt = torch.isfinite(gt) & (gt > 1e-6) & (gt < 1e5) & fg_ref
            sampled_fg = fg_ref[vi, ui]
            sampled_valid = valid_gt[vi, ui]
            sampled_z = gt[vi, ui]
            depth_match = sampled_valid & ((z[idx] - sampled_z).abs() <= tol)
            coverage[idx] += 1.0
            if depth_match.any():
                supports[idx[depth_match]] += 1.0
            front_conflict = sampled_valid & (z[idx] < sampled_z - tol)
            conflict = (~sampled_fg) | front_conflict
            if conflict.any():
                conflicts[idx[conflict]] += 1.0
            if front_conflict.any():
                front_conflicts[idx[front_conflict]] += 1.0

        return (
            supports.reshape(h, w),
            conflicts.reshape(h, w),
            front_conflicts.reshape(h, w),
            coverage.reshape(h, w),
            valid_prior,
        )

    def _support_gate_multiview_target(view_i: int,
                                       depth_i: torch.Tensor,
                                       target_depths: torch.Tensor | None,
                                       fg: torch.Tensor,
                                       K_all: torch.Tensor,
                                       c2w_all: torch.Tensor,
                                       w2c_all: torch.Tensor | None,
                                       radius: float,
                                       tol_frac: float | None = None,
                                       max_refs: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if target_depths is None or w2c_all is None or view_i >= target_depths.shape[0]:
            return _support_gate_target(
                depth_i, None, fg[view_i],
                K_all[view_i].to(device=depth_i.device, dtype=depth_i.dtype),
                radius, tol_frac=tol_frac,
            )
        supports, conflicts, _, _, valid_prior = _support_gate_multiview_counts(
            view_i, depth_i, target_depths, fg, K_all, c2w_all, w2c_all, radius,
            tol_frac=tol_frac, max_refs=max_refs,
        )
        denom = supports + conflicts
        valid = valid_prior & (denom > 0)
        target = torch.where(valid, supports / denom.clamp_min(1.0), supports.new_zeros(()))
        return target, valid

    def _apply_support_gate(p: dict,
                            view_i: int,
                            frames: torch.Tensor,
                            fg: torch.Tensor,
                            depths: torch.Tensor | None,
                            target_depths: torch.Tensor | None,
                            K_all: torch.Tensor,
                            c2w_all: torch.Tensor,
                            w2c_all: torch.Tensor | None,
                            radius: float) -> dict:
        if support_gate_head is None or depths is None or view_i >= depths.shape[0]:
            return p
        depth_i = depths[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype)
        feat, _, depth_valid = _depth_gate_features(
            frames[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            fg[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            depth_i,
            K_all[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            c2w_all[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            radius,
        )
        if args.support_gate_detach_inputs:
            feat = feat.detach()
        raw_delta = support_gate_head(feat)
        scale = max(float(args.support_gate_delta_scale), 0.0)
        delta = scale * torch.tanh(raw_delta) if scale > 0 else raw_delta * 0.0
        init = min(max(float(args.support_gate_init), 1e-4), 1.0 - 1e-4)
        prior = math.log(init / (1.0 - init))
        gate = torch.sigmoid(delta + delta.new_tensor(prior))
        floor = min(max(float(args.support_gate_floor), 0.0), 1.0)
        gate = floor + (1.0 - floor) * gate
        if gate.shape[-2:] != (model.map_h, model.map_w):
            gate_map = Fnn.interpolate(gate, size=(model.map_h, model.map_w),
                                       mode="bilinear", align_corners=False)
        else:
            gate_map = gate
        q = dict(p)
        q["opacity"] = p["opacity"] * gate_map[0, 0].reshape(-1, 1).to(dtype=p["opacity"].dtype)
        if torch.is_grad_enabled():
            valid = depth_valid > 0.5
            if valid.any():
                support_gate_delta_terms.append(
                    (delta[0, 0][valid].square()).mean()
                )
            if delta.shape[-1] > 1 and delta.shape[-2] > 1:
                valid_x = depth_valid[:, 1:] * depth_valid[:, :-1]
                valid_y = depth_valid[1:, :] * depth_valid[:-1, :]
                if valid_x.any():
                    tv_x = ((delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs()[0, 0]
                            * valid_x).sum() / valid_x.sum().clamp_min(1.0)
                else:
                    tv_x = delta.new_zeros(())
                if valid_y.any():
                    tv_y = ((delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs()[0, 0]
                            * valid_y).sum() / valid_y.sum().clamp_min(1.0)
                else:
                    tv_y = delta.new_zeros(())
                support_gate_tv_terms.append(0.5 * (tv_x + tv_y))
            if target_depths is not None and view_i < target_depths.shape[0]:
                with torch.no_grad():
                    if args.support_gate_multiview_target:
                        target, target_valid = _support_gate_multiview_target(
                            view_i, depth_i, target_depths, fg, K_all, c2w_all, w2c_all, radius
                        )
                    else:
                        target, target_valid = _support_gate_target(
                            depth_i,
                            target_depths[view_i].to(device=depth_i.device, dtype=depth_i.dtype),
                            fg[view_i].to(device=depth_i.device, dtype=depth_i.dtype),
                            K_all[view_i].to(device=depth_i.device, dtype=depth_i.dtype),
                            radius,
                        )
                if target_valid.any():
                    pred = gate[0, 0].clamp(1e-4, 1.0 - 1e-4)
                    loss = Fnn.binary_cross_entropy(
                        pred[target_valid], target[target_valid], reduction="mean"
                    )
                    support_gate_gt_terms.append(loss)
        return q

    def _surface_confidence_features(frames_i: torch.Tensor,
                                     fg_i: torch.Tensor,
                                     depth_i: torch.Tensor,
                                     K_i: torch.Tensor,
                                     c2w_i: torch.Tensor,
                                     radius: float,
                                     mv_i: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        feat, _, depth_valid = _depth_gate_features(
            frames_i, fg_i, depth_i, K_i, c2w_i, radius
        )
        h, w = depth_valid.shape
        if mv_i is None:
            mv = feat.new_zeros(4, h, w)
        else:
            mv = mv_i.to(device=feat.device, dtype=feat.dtype)
            if mv.shape[-2:] != (h, w):
                mv = Fnn.interpolate(
                    mv[None], size=(h, w), mode="bilinear", align_corners=False
                )[0]
        feat = torch.cat([feat[0], mv], dim=0)[None]
        return feat, depth_valid

    def _apply_surface_confidence_gate(p: dict,
                                       view_i: int,
                                       frames: torch.Tensor,
                                       fg: torch.Tensor,
                                       depths: torch.Tensor | None,
                                       target_depths: torch.Tensor | None,
                                       K_all: torch.Tensor,
                                       c2w_all: torch.Tensor,
                                       w2c_all: torch.Tensor | None,
                                       radius: float,
                                       mv_features: torch.Tensor | None) -> dict:
        if surface_confidence_head is None or depths is None or view_i >= depths.shape[0]:
            return p
        depth_i = depths[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype)
        mv_i = None
        if mv_features is not None and view_i < mv_features.shape[0]:
            mv_i = mv_features[view_i]
        feat, depth_valid = _surface_confidence_features(
            frames[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            fg[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            depth_i,
            K_all[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            c2w_all[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            radius,
            mv_i,
        )
        if args.surface_confidence_detach_inputs:
            feat = feat.detach()
        raw_delta = surface_confidence_head(feat)
        scale = max(float(args.surface_confidence_delta_scale), 0.0)
        delta = scale * torch.tanh(raw_delta) if scale > 0 else raw_delta * 0.0
        init = min(max(float(args.surface_confidence_init), 1e-4), 1.0 - 1e-4)
        prior = math.log(init / (1.0 - init))
        prob = torch.sigmoid(delta + delta.new_tensor(prior))
        floor = min(max(float(args.surface_confidence_floor), 0.0), 1.0)
        gate_raw = prob / init
        gate_strength = min(max(float(args.surface_confidence_gate_strength), 0.0), 1.0)
        gate = 1.0 + gate_strength * (gate_raw - 1.0)
        gate = gate.clamp(min=floor, max=max(1.0 / init, 1.0))
        protect = _surface_confidence_protect_mask(
            mv_i,
            args.surface_confidence_protect_support_min,
            args.surface_confidence_protect_conflict_max,
            args.surface_confidence_protect_coverage_min,
        )
        if protect is not None:
            protect_f = protect.to(device=gate.device, dtype=gate.dtype)[None, None]
            gate = torch.where(protect_f > 0.5, gate.new_ones(()), gate)
            prob = torch.where(protect_f > 0.5, prob.new_full((), init), prob)
            gate_raw = torch.where(protect_f > 0.5, gate_raw.new_ones(()), gate_raw)
        if gate.shape[-2:] != (model.map_h, model.map_w):
            gate_map = Fnn.interpolate(gate, size=(model.map_h, model.map_w),
                                       mode="bilinear", align_corners=False)
            prob_map = Fnn.interpolate(prob, size=(model.map_h, model.map_w),
                                       mode="bilinear", align_corners=False)
            gate_raw_map = Fnn.interpolate(gate_raw, size=(model.map_h, model.map_w),
                                           mode="bilinear", align_corners=False)
        else:
            gate_map = gate
            prob_map = prob
            gate_raw_map = gate_raw
        q = dict(p)
        gated_opacity = p["opacity"] * gate_map[0, 0].reshape(-1, 1).to(dtype=p["opacity"].dtype)
        if args.surface_confidence_opacity_max > 0:
            gated_opacity = gated_opacity.clamp(max=float(args.surface_confidence_opacity_max))
        q["opacity"] = gated_opacity
        scale_strength = min(max(float(args.surface_confidence_scale_strength), 0.0), 1.0)
        if scale_strength > 0 and "scale" in p:
            scale_floor = min(max(float(args.surface_confidence_scale_floor), 0.0), 1.0)
            scale_raw = gate_raw_map.clamp(max=1.0)
            scale_gate = 1.0 + scale_strength * (scale_raw - 1.0)
            scale_gate = scale_gate.clamp(min=scale_floor, max=1.0)
            q["scale"] = p["scale"] * scale_gate[0, 0].reshape(-1, 1).to(dtype=p["scale"].dtype)
        q["_surface_confidence_prob"] = prob_map[0, 0].reshape(-1, 1).to(dtype=p["opacity"].dtype)
        q["_surface_confidence_gate"] = gate_map[0, 0].reshape(-1, 1).to(dtype=p["opacity"].dtype)
        if torch.is_grad_enabled():
            valid = depth_valid > 0.5
            if valid.any():
                surface_confidence_delta_terms.append(
                    (delta[0, 0][valid].square()).mean()
                )
            if delta.shape[-1] > 1 and delta.shape[-2] > 1:
                valid_x = depth_valid[:, 1:] * depth_valid[:, :-1]
                valid_y = depth_valid[1:, :] * depth_valid[:-1, :]
                if valid_x.any():
                    tv_x = ((delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs()[0, 0]
                            * valid_x).sum() / valid_x.sum().clamp_min(1.0)
                else:
                    tv_x = delta.new_zeros(())
                if valid_y.any():
                    tv_y = ((delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs()[0, 0]
                            * valid_y).sum() / valid_y.sum().clamp_min(1.0)
                else:
                    tv_y = delta.new_zeros(())
                surface_confidence_tv_terms.append(0.5 * (tv_x + tv_y))
            if target_depths is not None and view_i < target_depths.shape[0]:
                with torch.no_grad():
                    if w2c_all is not None:
                        supports, conflicts, _, _, valid_prior = _support_gate_multiview_counts(
                            view_i, depth_i, target_depths, fg, K_all, c2w_all, w2c_all, radius,
                            tol_frac=args.surface_confidence_depth_tol_frac,
                            max_refs=args.surface_confidence_multiview_refs,
                        )
                        denom = supports + conflicts
                        target_valid = valid_prior & (denom > 0)
                        target = torch.where(
                            target_valid,
                            supports / denom.clamp_min(1.0),
                            supports.new_zeros(()),
                        )
                    else:
                        target, target_valid = _support_gate_target(
                            depth_i,
                            target_depths[view_i].to(device=depth_i.device, dtype=depth_i.dtype),
                            fg[view_i].to(device=depth_i.device, dtype=depth_i.dtype),
                            K_all[view_i].to(device=depth_i.device, dtype=depth_i.dtype),
                            radius,
                            tol_frac=args.surface_confidence_depth_tol_frac,
                        )
                        supports = target
                        conflicts = 1.0 - target
                    pos_min = float(args.surface_confidence_target_pos_min)
                    neg_max = float(args.surface_confidence_target_neg_max)
                    min_pos_support = float(args.surface_confidence_target_min_pos_support)
                    min_neg_conflicts = float(args.surface_confidence_target_min_neg_conflicts)
                    if (pos_min > 0.0 or neg_max < 1.0
                            or min_pos_support > 0.0 or min_neg_conflicts > 0.0):
                        strong_pos = target_valid & (target >= pos_min)
                        strong_neg = target_valid & (target <= neg_max)
                        if min_pos_support > 0.0:
                            strong_pos = strong_pos & (supports >= min_pos_support)
                        if min_neg_conflicts > 0.0:
                            strong_neg = strong_neg & (conflicts >= min_neg_conflicts)
                        target_valid = strong_pos | strong_neg
                        target = torch.where(
                            strong_pos,
                            target.new_ones(()),
                            target.new_zeros(()),
                        )
                if target_valid.any():
                    pred = prob[0, 0].clamp(1e-4, 1.0 - 1e-4)
                    target_v = target[target_valid]
                    loss_px = Fnn.binary_cross_entropy(
                        pred[target_valid], target_v, reduction="none"
                    )
                    pos_w = max(float(args.surface_confidence_positive_weight), 0.0)
                    neg_w = max(float(args.surface_confidence_negative_weight), 0.0)
                    w_loss = target_v * pos_w + (1.0 - target_v) * neg_w
                    loss = (loss_px * w_loss).sum() / w_loss.sum().clamp_min(1e-6)
                    surface_confidence_gt_terms.append(loss)
        return q

    def _apply_surface_refine(p: dict,
                              view_i: int,
                              frames: torch.Tensor,
                              fg: torch.Tensor,
                              depths: torch.Tensor | None,
                              target_frames: torch.Tensor | None,
                              K_all: torch.Tensor,
                              c2w_all: torch.Tensor,
                              radius: float,
                              mv_features: torch.Tensor | None) -> dict:
        if surface_refine_head is None or depths is None or view_i >= depths.shape[0]:
            return p
        depth_i = depths[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype)
        mv_i = None
        if mv_features is not None and view_i < mv_features.shape[0]:
            mv_i = mv_features[view_i]
        feat, depth_valid = _surface_confidence_features(
            frames[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            fg[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            depth_i,
            K_all[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            c2w_all[view_i].to(device=p["opacity"].device, dtype=p["opacity"].dtype),
            radius,
            mv_i,
        )
        if args.surface_refine_detach_inputs:
            feat = feat.detach()
        if (args.surface_refine_checkpoint
                and torch.is_grad_enabled()
                and surface_refine_head.training):
            from torch.utils.checkpoint import checkpoint

            raw = checkpoint(surface_refine_head, feat, use_reentrant=False)
        else:
            raw = surface_refine_head(feat)
        op_delta = (
            max(float(args.surface_refine_opacity_delta_scale), 0.0)
            * torch.tanh(raw[:, 0:1])
        )
        scale_delta = (
            max(float(args.surface_refine_scale_delta_scale), 0.0)
            * torch.tanh(raw[:, 1:2])
        )
        rgb_delta = (
            max(float(args.surface_refine_rgb_delta_scale), 0.0)
            * torch.tanh(raw[:, 2:5])
        )
        delta_all = torch.cat([op_delta, scale_delta, rgb_delta], dim=1)

        init = min(max(float(args.surface_refine_init), 1e-4), 1.0 - 1e-4)
        prior = math.log(init / (1.0 - init))
        prob = torch.sigmoid(op_delta + op_delta.new_tensor(prior))
        floor = min(max(float(args.surface_refine_opacity_floor), 0.0), 1.0)
        op_gate = (prob / init).clamp(min=floor, max=max(1.0 / init, 1.0))
        scale_floor = min(max(float(args.surface_refine_scale_floor), 1e-3), 1.0)
        scale_gate = torch.exp(scale_delta).clamp(min=scale_floor, max=1.0 / scale_floor)

        if op_gate.shape[-2:] != (model.map_h, model.map_w):
            op_gate_map = Fnn.interpolate(
                op_gate, size=(model.map_h, model.map_w),
                mode="bilinear", align_corners=False,
            )
            scale_gate_map = Fnn.interpolate(
                scale_gate, size=(model.map_h, model.map_w),
                mode="bilinear", align_corners=False,
            )
            rgb_delta_map = Fnn.interpolate(
                rgb_delta, size=(model.map_h, model.map_w),
                mode="bilinear", align_corners=False,
            )
            valid_map = Fnn.interpolate(
                depth_valid[None, None].to(dtype=op_gate.dtype),
                size=(model.map_h, model.map_w),
                mode="nearest",
            )
            prob_map = Fnn.interpolate(
                prob, size=(model.map_h, model.map_w),
                mode="bilinear", align_corners=False,
            )
        else:
            op_gate_map = op_gate
            scale_gate_map = scale_gate
            rgb_delta_map = rgb_delta
            valid_map = depth_valid[None, None].to(dtype=op_gate.dtype)
            prob_map = prob

        op_gate_map = op_gate_map * valid_map + (1.0 - valid_map)
        scale_gate_map = scale_gate_map * valid_map + (1.0 - valid_map)
        prob_map = prob_map * valid_map + prob_map.new_full((), init) * (1.0 - valid_map)
        valid_flat = valid_map[0, 0].reshape(-1, 1).to(device=p["opacity"].device)
        q = dict(p)
        q["opacity"] = (
            p["opacity"] * op_gate_map[0, 0].reshape(-1, 1).to(dtype=p["opacity"].dtype)
        )
        if "scale" in p:
            q["scale"] = (
                p["scale"]
                * scale_gate_map[0, 0].reshape(-1, 1).to(dtype=p["scale"].dtype)
            )
        if "rgb" in p:
            rgb_flat = rgb_delta_map[0].permute(1, 2, 0).reshape(-1, 3)
            q["rgb"] = (
                p["rgb"]
                + rgb_flat.to(device=p["rgb"].device, dtype=p["rgb"].dtype)
                * valid_flat.to(dtype=p["rgb"].dtype)
            ).clamp(0.0, 1.0)
        q["_surface_refine_prob"] = prob_map[0, 0].reshape(-1, 1).to(dtype=p["opacity"].dtype)

        if torch.is_grad_enabled():
            valid = depth_valid > 0.5
            if valid.any():
                surface_refine_delta_terms.append(
                    (delta_all.square() * depth_valid[None, None]).sum()
                    / (depth_valid.sum().clamp_min(1.0) * delta_all.shape[1])
                )
            if delta_all.shape[-1] > 1 and delta_all.shape[-2] > 1:
                valid_x = depth_valid[:, 1:] * depth_valid[:, :-1]
                valid_y = depth_valid[1:, :] * depth_valid[:-1, :]
                if valid_x.any():
                    tv_x = ((delta_all[:, :, :, 1:] - delta_all[:, :, :, :-1]).abs()[0]
                            * valid_x[None]).sum() / valid_x.sum().clamp_min(1.0)
                else:
                    tv_x = delta_all.new_zeros(())
                if valid_y.any():
                    tv_y = ((delta_all[:, :, 1:, :] - delta_all[:, :, :-1, :]).abs()[0]
                            * valid_y[None]).sum() / valid_y.sum().clamp_min(1.0)
                else:
                    tv_y = delta_all.new_zeros(())
                surface_refine_tv_terms.append(0.5 * (tv_x + tv_y))
            if (target_frames is not None
                    and (args.surface_refine_rgb_gt_weight > 0
                         or args.surface_refine_rgb_grad_gt_weight > 0)
                    and view_i < target_frames.shape[0]):
                tgt = target_frames[view_i].to(device=rgb_delta.device, dtype=rgb_delta.dtype)
                src = frames[view_i].to(device=rgb_delta.device, dtype=rgb_delta.dtype)
                mask = fg[view_i].to(device=rgb_delta.device, dtype=rgb_delta.dtype)[..., 0]
                alpha_min = min(max(float(args.surface_refine_gt_alpha_min), 0.0), 1.0)
                valid_rgb = (mask > alpha_min) & valid
                if valid_rgb.any():
                    refined = (src + rgb_delta[0].permute(1, 2, 0)).clamp(0.0, 1.0)
                    if args.surface_refine_rgb_gt_weight > 0:
                        loss_px = (refined - tgt).abs().mean(dim=-1)
                        surface_refine_rgb_gt_terms.append(loss_px[valid_rgb].mean())
                    if args.surface_refine_rgb_grad_gt_weight > 0:
                        grad_mask = valid_rgb.to(dtype=rgb_delta.dtype)[None, ..., None]
                        surface_refine_rgb_grad_gt_terms.append(
                            _fg_gradient_loss(refined[None], tgt[None], grad_mask)
                        )
        return q

    def _learned_fill_features(r_near: torch.Tensor, a_near: torch.Tensor,
                               r_static: torch.Tensor, a_static: torch.Tensor,
                               prior: torch.Tensor,
                               target_center: torch.Tensor,
                               anchor_center: torch.Tensor,
                               view_dist: torch.Tensor,
                               radius: float | None,
                               bg: float) -> torch.Tensor:
        h, w = a_near.shape[0], a_near.shape[1]
        dtype, device = r_near.dtype, r_near.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        radius_norm = max(float(radius) if radius is not None else 1.0, 1e-6)
        dist_map = (view_dist.to(device=device, dtype=dtype) / radius_norm).clamp(0.0, 8.0)
        cos = Fnn.cosine_similarity(
            target_center.to(device=device, dtype=dtype)[None],
            anchor_center.to(device=device, dtype=dtype)[None],
            dim=-1,
        ).clamp(-1.0, 1.0)[0]
        scalars = [
            prior[..., 0],
            torch.maximum(a_near[..., 0], a_static[..., 0]),
            xx,
            yy,
            dist_map.expand(h, w),
            cos.expand(h, w),
            torch.full((h, w), float(bg), device=device, dtype=dtype),
        ]
        feats = [
            r_near.permute(2, 0, 1),
            a_near.permute(2, 0, 1),
            r_static.permute(2, 0, 1),
            a_static.permute(2, 0, 1),
            (r_near - r_static).abs().permute(2, 0, 1),
            (a_static - a_near).permute(2, 0, 1),
            torch.stack(scalars, 0),
        ]
        out = torch.cat(feats, 0)[None]
        if out.shape[1] != LEARNED_FILL_FEATURE_CHANNELS:
            raise RuntimeError(f"learned-fill feature bug: got {out.shape[1]} channels")
        return out

    def _apply_learned_fill(r_near: torch.Tensor, a_near: torch.Tensor,
                            r_static: torch.Tensor, a_static: torch.Tensor,
                            prior: torch.Tensor,
                            target_center: torch.Tensor,
                            anchor_center: torch.Tensor,
                            view_dist: torch.Tensor,
                            radius: float | None,
                            bg: float) -> tuple[torch.Tensor, torch.Tensor]:
        if blend_head is None:
            raise ValueError("--anchor_render_mode learned_fill requires a blend head")
        if args.anchor_learned_fill_detach_inputs:
            r_n = r_near.detach()
            a_n = a_near.detach()
            r_s = r_static.detach()
            a_s = a_static.detach()
            p0 = prior.detach()
        else:
            r_n, a_n, r_s, a_s, p0 = r_near, a_near, r_static, a_static, prior
        feat = _learned_fill_features(
            r_n, a_n, r_s, a_s, p0, target_center, anchor_center,
            view_dist, radius, bg,
        )
        gate_delta, rgb_delta = blend_head(feat)
        delta_scale = max(float(args.anchor_learned_fill_delta_scale), 0.0)
        if delta_scale > 0:
            gate_delta = delta_scale * torch.tanh(gate_delta)
        if torch.is_grad_enabled():
            learned_fill_delta_terms.append(gate_delta.square().mean())
            if gate_delta.shape[-1] > 1 and gate_delta.shape[-2] > 1:
                tv_x = (gate_delta[:, :, :, 1:] - gate_delta[:, :, :, :-1]).abs().mean()
                tv_y = (gate_delta[:, :, 1:, :] - gate_delta[:, :, :-1, :]).abs().mean()
                learned_fill_tv_terms.append(0.5 * (tv_x + tv_y))
        prior_logit = torch.logit(p0.permute(2, 0, 1)[None].clamp(1e-4, 1.0 - 1e-4))
        near_w = torch.sigmoid(prior_logit + gate_delta).permute(0, 2, 3, 1)[0]
        rgb = r_n * near_w + r_s * (1.0 - near_w)
        alpha = a_n * near_w + a_s * (1.0 - near_w)
        if rgb_delta is not None:
            support = torch.maximum(a_n, a_s).clamp(0.0, 1.0)
            delta = rgb_delta.permute(0, 2, 3, 1)[0] * support
            rgb = (rgb + delta).clamp(0.0, 1.0)
        return rgb, alpha

    def _learned_iblend_features(obj_stack: torch.Tensor,
                                 a_stack: torch.Tensor,
                                 prior_w: torch.Tensor,
                                 support_stack: torch.Tensor,
                                 d_stack: torch.Tensor,
                                 view_w: torch.Tensor,
                                 r_anchor: torch.Tensor,
                                 a_anchor: torch.Tensor,
                                 r_static: torch.Tensor,
                                 a_static: torch.Tensor,
                                 fill_prior: torch.Tensor,
                                 radius: float | None,
                                 bg: float) -> torch.Tensor:
        k, h, w, _ = obj_stack.shape
        dtype, device = obj_stack.dtype, obj_stack.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        radius_norm = max(float(radius) if radius is not None else 1.0, 1e-6)
        depth = (d_stack.to(device=device, dtype=dtype) / radius_norm).clamp(0.0, 16.0)
        vw = view_w.to(device=device, dtype=dtype).view(k, 1, 1).expand(k, h, w)
        cand = torch.cat([
            obj_stack.permute(0, 3, 1, 2).reshape(k * 3, h, w),
            a_stack.permute(0, 3, 1, 2).reshape(k, h, w),
            prior_w.permute(0, 3, 1, 2).reshape(k, h, w),
            support_stack.permute(0, 3, 1, 2).reshape(k, h, w),
            depth,
            vw,
        ], dim=0)
        shared = torch.cat([
            r_anchor.permute(2, 0, 1),
            a_anchor.permute(2, 0, 1),
            r_static.permute(2, 0, 1),
            a_static.permute(2, 0, 1),
            (r_anchor - r_static).abs().permute(2, 0, 1),
            fill_prior.permute(2, 0, 1),
            xx[None],
            yy[None],
            torch.full((1, h, w), float(bg), device=device, dtype=dtype),
        ], dim=0)
        out = torch.cat([cand, shared], dim=0)[None]
        expected = _learned_iblend_feature_channels(k)
        if out.shape[1] != expected:
            raise RuntimeError(f"learned-iblend feature bug: got {out.shape[1]} channels")
        return out

    def _apply_learned_iblend_fill(obj_stack: torch.Tensor,
                                   a_stack: torch.Tensor,
                                   w_stack: torch.Tensor,
                                   support_stack: torch.Tensor,
                                   d_stack: torch.Tensor,
                                   view_w: torch.Tensor,
                                   r_static: torch.Tensor,
                                   a_static: torch.Tensor,
                                   fill_prior: torch.Tensor,
                                   radius: float | None,
                                   bg: float,
                                   target_rgb: torch.Tensor | None = None,
                                   target_alpha: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if blend_head is None or not hasattr(blend_head, "topk"):
            raise ValueError("--anchor_render_mode learned_iblend_fill requires an iblend blend head")
        if obj_stack.shape[0] != blend_head.topk:
            raise ValueError(
                f"learned_iblend_fill was initialized for topk={blend_head.topk}, "
                f"but selected {obj_stack.shape[0]} candidates"
            )
        denom = w_stack.sum(dim=0).clamp_min(1e-6)
        prior_w = w_stack / denom
        _, a_det, r_det = _compose_iblend_anchor(
            obj_stack, a_stack, prior_w, args.anchor_iblend_color_mode,
            args.anchor_iblend_alpha_mode, bg
        )
        if args.anchor_learned_fill_detach_inputs:
            obj_in = obj_stack.detach()
            a_in = a_stack.detach()
            prior_in = prior_w.detach()
            support_in = support_stack.detach()
            d_in = d_stack.detach()
            r_det_in = r_det.detach()
            a_det_in = a_det.detach()
            r_s_in = r_static.detach()
            a_s_in = a_static.detach()
            fill_in = fill_prior.detach()
        else:
            obj_in, a_in, prior_in, support_in, d_in = (
                obj_stack, a_stack, prior_w, support_stack, d_stack
            )
            r_det_in, a_det_in, r_s_in, a_s_in, fill_in = (
                r_det, a_det, r_static, a_static, fill_prior
            )
        feat = _learned_iblend_features(
            obj_in, a_in, prior_in, support_in, d_in, view_w,
            r_det_in, a_det_in, r_s_in, a_s_in, fill_in, radius, bg,
        )
        cand_delta, fill_delta, rgb_delta = blend_head(feat)
        delta_scale = max(float(args.anchor_learned_fill_delta_scale), 0.0)
        cand_delta_scale = float(args.anchor_learned_fill_candidate_delta_scale)
        if cand_delta_scale < 0:
            cand_delta_scale = delta_scale
        else:
            cand_delta_scale = max(cand_delta_scale, 0.0)
        if cand_delta_scale > 0:
            cand_delta = cand_delta_scale * torch.tanh(cand_delta)
        else:
            cand_delta = cand_delta * 0.0
        if delta_scale > 0:
            fill_delta = delta_scale * torch.tanh(fill_delta)
        else:
            fill_delta = fill_delta * 0.0
        if torch.is_grad_enabled():
            delta_all = torch.cat([cand_delta, fill_delta], dim=1)
            learned_fill_delta_terms.append(delta_all.square().mean())
            if delta_all.shape[-1] > 1 and delta_all.shape[-2] > 1:
                tv_x = (delta_all[:, :, :, 1:] - delta_all[:, :, :, :-1]).abs().mean()
                tv_y = (delta_all[:, :, 1:, :] - delta_all[:, :, :-1, :]).abs().mean()
                learned_fill_tv_terms.append(0.5 * (tv_x + tv_y))
        cand_logits = torch.log(prior_in.permute(3, 0, 1, 2).clamp_min(1e-6)) + cand_delta
        cand_w = torch.softmax(cand_logits, dim=1).permute(1, 2, 3, 0)
        obj, alpha_anchor, r_anchor = _compose_iblend_anchor(
            obj_in, a_in, cand_w, args.anchor_iblend_color_mode,
            args.anchor_iblend_alpha_mode, bg
        )
        fill_logit = torch.logit(fill_in.permute(2, 0, 1)[None].clamp(1e-4, 1.0 - 1e-4))
        near_w = torch.sigmoid(fill_logit + fill_delta).permute(0, 2, 3, 1)[0]
        rgb = r_anchor * near_w + r_s_in * (1.0 - near_w)
        alpha = alpha_anchor * near_w + a_s_in * (1.0 - near_w)
        if (torch.is_grad_enabled() and args.anchor_learned_fill_oracle_weight > 0
                and target_rgb is not None and target_alpha is not None):
            tgt_rgb = target_rgb.to(device=rgb.device, dtype=rgb.dtype)
            tgt_a = target_alpha.to(device=rgb.device, dtype=rgb.dtype).clamp(0.0, 1.0)
            if tgt_a.ndim == 2:
                tgt_a = tgt_a[..., None]
            if tgt_a.shape[-1] != 1:
                tgt_a = tgt_a[..., :1]
            temp_o = max(float(args.anchor_learned_fill_oracle_temp), 1e-4)
            mask_w = max(float(args.anchor_learned_fill_oracle_mask_weight), 0.0)
            alpha_min = max(float(args.anchor_learned_fill_oracle_alpha_min), 0.0)
            cand_rgb = obj_in.clamp(0.0, 1.0) * a_in + (1.0 - a_in) * bg
            cand_err = (cand_rgb - tgt_rgb[None]).abs().mean(dim=-1)
            cand_err = cand_err + mask_w * (a_in[..., 0] - tgt_a[..., 0][None]).abs()
            oracle_cand = torch.softmax(-cand_err / temp_o, dim=0).detach()
            cand_prob = cand_w[..., 0].clamp(1e-6, 1.0)
            cand_ce = -(oracle_cand * torch.log(cand_prob)).sum(dim=0)

            anchor_err = (r_anchor - tgt_rgb).abs().mean(dim=-1)
            anchor_err = anchor_err + mask_w * (alpha_anchor[..., 0] - tgt_a[..., 0]).abs()
            static_err = (r_s_in - tgt_rgb).abs().mean(dim=-1)
            static_err = static_err + mask_w * (a_s_in[..., 0] - tgt_a[..., 0]).abs()
            anchor_target = torch.sigmoid((static_err - anchor_err) / temp_o).detach()
            fill_prob = near_w[..., 0].clamp(1e-6, 1.0 - 1e-6)
            fill_bce = Fnn.binary_cross_entropy(
                fill_prob, anchor_target, reduction="none"
            )
            support = (
                (tgt_a[..., 0] > alpha_min)
                | (a_in[..., 0].amax(dim=0) > alpha_min)
                | (a_s_in[..., 0] > alpha_min)
            )
            if support.any():
                w_o = 1.0 + (args.fg_weight - 1.0) * tgt_a[..., 0]
                oracle_loss = ((cand_ce + fill_bce) * w_o)[support].sum() / (
                    w_o[support].sum().clamp_min(1e-6)
                )
                learned_fill_oracle_terms.append(oracle_loss)
        if rgb_delta is not None:
            support = torch.maximum(alpha_anchor, a_s_in).clamp(0.0, 1.0)
            delta = rgb_delta.permute(0, 2, 3, 1)[0] * support
            rgb = (rgb + delta).clamp(0.0, 1.0)
        return rgb, alpha

    def _with_fusion_sources(parts: list[dict], source_ids: list[int] | None = None) -> list[dict]:
        if args.fusion_voxel_size_frac <= 0:
            return parts
        out = []
        for src_i, p in enumerate(parts):
            source_id = int(source_ids[src_i]) if source_ids is not None else src_i
            q = dict(p)
            n = q["mean"].shape[0]
            q["_fusion_source"] = torch.full(
                (n, 1), source_id, dtype=torch.long, device=q["mean"].device
            )
            out.append(q)
        return out

    def _apply_fusion_candidate_gate(p: dict,
                                     radius: float | None,
                                     ref_count: int) -> dict:
        if fusion_candidate_head is None:
            return p
        if radius is None:
            raise ValueError("--fusion_candidate_gate requires radius")
        feat = _fusion_candidate_features(
            p,
            radius,
            ref_count,
            include_coords=bool(args.fusion_candidate_coord_features),
            include_rich=bool(args.fusion_candidate_rich_features),
            include_voxel=bool(args.fusion_candidate_voxel_features),
            include_neighbor=bool(args.fusion_candidate_neighbor_features),
            neighbor_radius=args.fusion_candidate_neighbor_radius,
            voxel_size=(
                args.fusion_voxel_size_frac * float(radius)
                if args.fusion_voxel_size_frac > 0 else None
            ),
        )
        if args.fusion_candidate_detach_inputs:
            feat = feat.detach()
        def _run_candidate_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if (args.fusion_candidate_checkpoint
                    and torch.is_grad_enabled()
                    and fusion_candidate_head.training):
                from torch.utils.checkpoint import checkpoint

                return checkpoint(fusion_candidate_head, x, use_reentrant=False)
            return fusion_candidate_head(x)

        chunk_size = max(int(args.fusion_candidate_chunk_size), 0)
        if chunk_size > 0 and feat.shape[0] > chunk_size:
            score_chunks: list[torch.Tensor] = []
            opacity_chunks: list[torch.Tensor] = []
            for start in range(0, feat.shape[0], chunk_size):
                raw_s, raw_o = _run_candidate_head(feat[start:start + chunk_size])
                score_chunks.append(raw_s)
                opacity_chunks.append(raw_o)
            raw_score_delta = torch.cat(score_chunks, dim=0)
            raw_opacity_delta = torch.cat(opacity_chunks, dim=0)
        else:
            raw_score_delta, raw_opacity_delta = _run_candidate_head(feat)
        score_scale = max(float(args.fusion_candidate_score_delta_scale), 0.0)
        opacity_scale = max(float(args.fusion_candidate_opacity_delta_scale), 0.0)
        score_delta = (
            score_scale * torch.tanh(raw_score_delta)
            if score_scale > 0 else raw_score_delta * 0.0
        )
        opacity_delta = (
            opacity_scale * torch.tanh(raw_opacity_delta)
            if opacity_scale > 0 else raw_opacity_delta * 0.0
        )
        init = min(max(float(args.fusion_candidate_opacity_init), 1e-4), 1.0 - 1e-4)
        prior = math.log(init / (1.0 - init))
        prob = torch.sigmoid(opacity_delta + opacity_delta.new_tensor(prior))
        floor = min(max(float(args.fusion_candidate_opacity_floor), 0.0), 1.0)
        gate = (prob / init).clamp(min=floor, max=max(1.0 / init, 1.0))

        q = dict(p)
        base_score = q.get("_fusion_score")
        if base_score is None:
            base_score = score_delta.new_zeros(score_delta.shape)
        q["_fusion_score"] = base_score.to(device=score_delta.device, dtype=score_delta.dtype) + score_delta
        q["opacity"] = p["opacity"] * gate.to(device=p["opacity"].device, dtype=p["opacity"].dtype)
        q["_fusion_candidate_prob"] = prob.to(device=p["opacity"].device, dtype=p["opacity"].dtype)
        q["_fusion_candidate_gate"] = gate.to(device=p["opacity"].device, dtype=p["opacity"].dtype)

        if torch.is_grad_enabled():
            fusion_candidate_delta_terms.append(
                0.5 * (score_delta.square().mean() + opacity_delta.square().mean())
            )
            support = p.get("_fusion_support")
            conflict = p.get("_fusion_conflict")
            if args.fusion_candidate_gt_source == "target_depth":
                support = p.get("_fusion_target_support", support)
                conflict = p.get("_fusion_target_conflict", conflict)
            if (support is not None and conflict is not None
                    and args.fusion_candidate_gt_weight > 0):
                support = support.reshape(-1, 1).to(device=prob.device, dtype=prob.dtype)
                conflict = conflict.reshape(-1, 1).to(device=prob.device, dtype=prob.dtype)
                denom = support + conflict
                valid = denom > 0
                target_soft = torch.where(
                    valid,
                    support / denom.clamp_min(1.0),
                    support.new_zeros(()),
                )
                pos_min = float(args.fusion_candidate_target_pos_min)
                neg_max = float(args.fusion_candidate_target_neg_max)
                if pos_min > 0.0 or neg_max < 1.0:
                    pos = valid & (target_soft >= pos_min)
                    neg = valid & (target_soft <= neg_max)
                    valid = pos | neg
                    target = torch.where(pos, target_soft.new_ones(()), target_soft.new_zeros(()))
                else:
                    target = target_soft
                if valid.any():
                    score_prob = torch.sigmoid(score_delta).clamp(1e-4, 1.0 - 1e-4)
                    gate_prob = prob.clamp(1e-4, 1.0 - 1e-4)
                    loss_px = 0.5 * (
                        Fnn.binary_cross_entropy(score_prob, target, reduction="none")
                        + Fnn.binary_cross_entropy(gate_prob, target, reduction="none")
                    )
                    pos_w = max(float(args.fusion_candidate_positive_weight), 0.0)
                    neg_w = max(float(args.fusion_candidate_negative_weight), 0.0)
                    weights = target * pos_w + (1.0 - target) * neg_w
                    fusion_candidate_gt_terms.append(
                        (loss_px[valid] * weights[valid]).sum()
                        / weights[valid].sum().clamp_min(1e-6)
                    )
        return q

    def _with_depth_consistency_score(p: dict,
                                      source_frames: torch.Tensor | None,
                                      source_fg: torch.Tensor | None,
                                      source_depths: torch.Tensor | None,
                                      source_confs: torch.Tensor | None,
                                      source_w2c: torch.Tensor | None,
                                      source_K: torch.Tensor | None,
                                      radius: float | None,
                                      target_depths_for_candidate: torch.Tensor | None = None,
                                      force_target_counts: bool = False) -> dict:
        if not args.fusion_voxel_score_depth:
            return p
        if radius is None:
            raise ValueError("--fusion_voxel_score_depth requires radius")
        if source_fg is None or source_depths is None or source_w2c is None or source_K is None:
            raise ValueError("--fusion_voxel_score_depth requires source RGBD conditioning")
        if args.fusion_voxel_score_color and source_frames is None:
            raise ValueError("--fusion_voxel_score_color requires source RGBD conditioning")
        if args.fusion_voxel_score_confidence and source_confs is None:
            raise ValueError("--fusion_voxel_score_confidence requires conditioning confidence maps")
        means = p["mean"].detach()
        score = torch.zeros(means.shape[0], dtype=means.dtype, device=means.device)
        support_count = torch.zeros_like(score)
        conflict_count = torch.zeros_like(score)
        coverage_count = torch.zeros_like(score)
        color_support = torch.zeros_like(score)
        depth_error_sum = torch.zeros_like(score)
        color_error_sum = torch.zeros_like(score)
        front_conflict_count = torch.zeros_like(score)
        silhouette_conflict_count = torch.zeros_like(score)
        use_target_candidate_gt = (
            target_depths_for_candidate is not None
            and (
                (
                    fusion_candidate_head is not None
                    and args.fusion_candidate_gt_source == "target_depth"
                )
                or (
                    sparse_voxel_fusion_head is not None
                    and args.sparse_voxel_target_vis_weight > 0
                )
                or force_target_counts
            )
        )
        target_support_count = torch.zeros_like(score) if use_target_candidate_gt else None
        target_conflict_count = torch.zeros_like(score) if use_target_candidate_gt else None
        h, w = source_fg.shape[1], source_fg.shape[2]
        tol_frac = args.fusion_voxel_score_depth_tol_frac
        tol = (tol_frac if tol_frac > 0 else args.fusion_depth_tol_frac) * float(radius)
        conflict_weight = max(float(args.fusion_voxel_score_conflict_weight), 0.0)
        color_weight = max(float(args.fusion_voxel_score_color_weight), 0.0)
        color_tol = max(float(args.fusion_voxel_score_color_tol), 1e-4)
        conf_floor = min(max(float(args.fusion_voxel_score_confidence_floor), 0.0), 1.0)
        conf_power = max(float(args.fusion_voxel_score_confidence_power), 1e-6)
        margin = max(args.fusion_bg_margin_px, 0)
        n_ref = min(source_depths.shape[0], source_w2c.shape[0], source_K.shape[0])
        source_ids = p.get("_fusion_source")
        if args.fusion_voxel_score_exclude_source_view and source_ids is not None:
            source_ids = source_ids.reshape(-1).to(device=means.device, dtype=torch.long)
        else:
            source_ids = None
        with torch.no_grad():
            for ref_view in range(n_ref):
                cam = means @ source_w2c[ref_view, :3, :3].T + source_w2c[ref_view, :3, 3]
                z = cam[:, 2]
                fx, fy = source_K[ref_view, 0, 0], source_K[ref_view, 1, 1]
                cx, cy = source_K[ref_view, 0, 2], source_K[ref_view, 1, 2]
                u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
                v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
                inb = (z > 1e-6) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
                if not inb.any():
                    continue
                fg_view = source_fg[ref_view, ..., 0].to(device=means.device) > 0.5
                if margin > 0:
                    fg_view = Fnn.max_pool2d(
                        fg_view.float()[None, None],
                        kernel_size=2 * margin + 1,
                        stride=1,
                        padding=margin,
                    )[0, 0] > 0.5
                z_view = source_depths[ref_view].to(device=means.device, dtype=means.dtype)
                valid_view = (z_view < 1e5) & fg_view
                conf_view = None
                if args.fusion_voxel_score_confidence:
                    c = source_confs[ref_view].to(device=means.device, dtype=means.dtype)
                    if args.fusion_voxel_score_confidence_normalize:
                        c_valid = torch.isfinite(c) & valid_view
                        c_norm = _normalize_confidence_map(c, c_valid)
                    else:
                        c_norm = torch.where(torch.isfinite(c), c.clamp(0.0, 1.0), c.new_zeros(c.shape))
                    conf_view = conf_floor + (1.0 - conf_floor) * c_norm.pow(conf_power)
                idx = torch.nonzero(inb, as_tuple=False).squeeze(1)
                coverage_count[idx] += 1.0
                ui = u[idx].round().long().clamp(0, w - 1)
                vi = v[idx].round().long().clamp(0, h - 1)
                sampled_conf = None
                if conf_view is not None:
                    sampled_conf = conf_view[vi, ui].clamp(0.0, 1.0)
                nearest_valid = valid_view[vi, ui]
                nearest_z = z_view[vi, ui]
                non_source = None
                if source_ids is not None:
                    non_source = source_ids[idx] != ref_view
                sampled_fg, depth_match, front_conflict, _ = _sample_depth_support_window(
                    fg_view,
                    z_view,
                    u[idx],
                    v[idx],
                    z[idx],
                    tol,
                    int(args.support_sample_radius_px),
                )
                depth_error_valid = sampled_fg & nearest_valid
                if non_source is not None:
                    depth_error_valid = depth_error_valid & non_source
                if depth_error_valid.any():
                    depth_err = (
                        (z[idx[depth_error_valid]] - nearest_z[depth_error_valid]).abs()
                        / max(float(tol), 1e-6)
                    ).clamp(0.0, 4.0)
                    if (sampled_conf is not None
                            and args.fusion_voxel_score_confidence_conflicts):
                        depth_err = depth_err * sampled_conf[depth_error_valid]
                    depth_error_sum[idx[depth_error_valid]] += depth_err
                if non_source is not None:
                    depth_match = depth_match & non_source
                if depth_match.any():
                    match_idx = idx[depth_match]
                    if (conf_view is not None
                            and args.fusion_voxel_score_confidence_supports):
                        match_conf = conf_view[vi[depth_match], ui[depth_match]]
                    else:
                        match_conf = score.new_ones(match_idx.shape)
                    score[match_idx] += match_conf
                    support_count[match_idx] += match_conf
                    if args.fusion_voxel_score_color and color_weight > 0:
                        sampled_rgb = source_frames[ref_view].to(
                            device=means.device, dtype=means.dtype
                        )[vi[depth_match], ui[depth_match]]
                        err = (p["rgb"].detach()[match_idx].to(means.dtype) - sampled_rgb).abs().mean(dim=-1)
                        color_bonus = match_conf * torch.exp(-err / color_tol)
                        color_support[match_idx] += color_bonus
                        score[match_idx] += color_weight * color_bonus
                if args.fusion_voxel_score_color and source_frames is not None:
                    color_valid = sampled_fg
                    if non_source is not None:
                        color_valid = color_valid & non_source
                    if color_valid.any():
                        color_idx = idx[color_valid]
                        sampled_rgb = source_frames[ref_view].to(
                            device=means.device, dtype=means.dtype
                        )[vi[color_valid], ui[color_valid]]
                        color_err = (
                            p["rgb"].detach()[color_idx].to(means.dtype) - sampled_rgb
                        ).abs().mean(dim=-1).clamp(0.0, 1.0)
                        color_error_sum[color_idx] += color_err
                if use_target_candidate_gt and ref_view < target_depths_for_candidate.shape[0]:
                    gt_z_view = target_depths_for_candidate[ref_view].to(
                        device=means.device, dtype=means.dtype
                    )
                    gt_valid = torch.isfinite(gt_z_view) & (gt_z_view > 1e-6) & (gt_z_view < 1e5)
                    gt_sampled_fg, gt_depth_match, gt_front_conflict, _ = _sample_depth_support_window(
                        fg_view & gt_valid,
                        gt_z_view,
                        u[idx],
                        v[idx],
                        z[idx],
                        tol,
                        int(args.support_sample_radius_px),
                    )
                    if gt_depth_match.any():
                        target_support_count[idx[gt_depth_match]] += 1.0
                    gt_conflict = (~gt_sampled_fg) | gt_front_conflict
                    if gt_conflict.any():
                        target_conflict_count[idx[gt_conflict]] += 1.0
                if conflict_weight > 0 or fusion_candidate_head is not None:
                    silhouette_conflict = ~sampled_fg
                    conflict = silhouette_conflict | front_conflict
                    if non_source is not None:
                        conflict = conflict & non_source
                        silhouette_conflict = silhouette_conflict & non_source
                        front_conflict = front_conflict & non_source
                    if conflict.any():
                        conflict_idx = idx[conflict]
                        if (sampled_conf is not None
                                and args.fusion_voxel_score_confidence_conflicts):
                            conf_w = torch.where(
                                silhouette_conflict,
                                sampled_conf.new_ones(sampled_conf.shape),
                                sampled_conf,
                            )
                            conflict_amount = conf_w[conflict]
                        else:
                            conflict_amount = score.new_ones(conflict_idx.shape)
                        conflict_count[conflict_idx] += conflict_amount
                        if conflict_weight > 0:
                            score[conflict_idx] -= conflict_weight * conflict_amount
                    if front_conflict.any():
                        if (sampled_conf is not None
                                and args.fusion_voxel_score_confidence_conflicts):
                            front_conflict_count[idx[front_conflict]] += sampled_conf[front_conflict]
                        else:
                            front_conflict_count[idx[front_conflict]] += 1.0
                    if silhouette_conflict.any():
                        silhouette_conflict_count[idx[silhouette_conflict]] += 1.0
        out = dict(p)
        if (args.surface_confidence_score_weight > 0
                and "_surface_confidence_prob" in p):
            conf = p["_surface_confidence_prob"].reshape(-1).to(
                device=score.device, dtype=score.dtype
            )
            init = min(max(float(args.surface_confidence_init), 1e-4), 1.0 - 1e-4)
            score = score + float(args.surface_confidence_score_weight) * (conf - init)
        out["_fusion_score"] = score[:, None]
        out["_fusion_support"] = support_count[:, None]
        out["_fusion_conflict"] = conflict_count[:, None]
        out["_fusion_coverage"] = coverage_count[:, None]
        out["_fusion_color_support"] = color_support[:, None]
        out["_fusion_depth_error"] = depth_error_sum[:, None]
        out["_fusion_color_error"] = color_error_sum[:, None]
        out["_fusion_front_conflict"] = front_conflict_count[:, None]
        out["_fusion_silhouette_conflict"] = silhouette_conflict_count[:, None]
        if target_support_count is not None and target_conflict_count is not None:
            out["_fusion_target_support"] = target_support_count[:, None]
            out["_fusion_target_conflict"] = target_conflict_count[:, None]
        return _apply_fusion_candidate_gate(out, radius, n_ref)

    def _maybe_voxel_fuse_params(p: dict, radius: float | None,
                                 apply_sparse_refine: bool = True) -> dict:
        if args.fusion_voxel_size_frac <= 0:
            return p
        if radius is None:
            raise ValueError("--fusion_voxel_size_frac requires an object radius")
        voxel_size = args.fusion_voxel_size_frac * float(radius)
        scale_floor = (
            voxel_size * args.fusion_voxel_scale_mult
            if args.fusion_voxel_scale_mult > 0 else None
        )
        fused, _ = voxel_fuse_params(
            p, voxel_size=voxel_size, min_count=args.fusion_voxel_min_count,
            max_per_voxel=args.fusion_voxel_max_per_cell,
            mode=args.fusion_voxel_mode,
            color_mode=args.fusion_voxel_color_mode,
            color_select_mix=args.fusion_voxel_color_select_mix,
            representative_mode=args.fusion_voxel_representative,
            score_softmax_temp=args.fusion_voxel_score_softmax_temp,
            score_soft_opacity_mix=args.fusion_voxel_score_soft_opacity_mix,
            score_soft_geometry_mix=args.fusion_voxel_score_soft_geometry_mix,
            scale_floor=scale_floor,
            scale_floor_z_mult=args.fusion_voxel_scale_floor_z_mult,
            low_support_scale_floor_mult=args.fusion_voxel_low_support_scale_mult,
            scale_floor_detail_key="_fusion_detail",
            scale_floor_detail_min=args.fusion_voxel_detail_scale_min,
            average_dist_decay=args.fusion_voxel_average_dist_decay,
            neighbor_support_radius=args.fusion_voxel_neighbor_radius,
            neighbor_support_min=args.fusion_voxel_neighbor_min,
            neighbor_opacity_decay=args.fusion_voxel_neighbor_opacity_decay,
            support_propagation_steps=args.fusion_voxel_support_propagation_steps,
            support_propagation_radius=args.fusion_voxel_support_propagation_radius,
            support_propagation_opacity_decay=args.fusion_voxel_support_propagation_opacity_decay,
            support_key="_fusion_source",
            low_support_opacity_decay=args.fusion_voxel_low_support_opacity_decay,
            coverage_opacity_mult=args.fusion_voxel_coverage_opacity_mult,
            coverage_scale_mult=args.fusion_voxel_coverage_scale_mult,
            pca_quat=bool(args.fusion_voxel_pca_quat),
        )
        if (args.fusion_voxel_score_opacity_norm > 0 and "_fusion_score" in fused
                and "opacity" in fused):
            score = fused["_fusion_score"].reshape(-1, 1).to(
                device=fused["opacity"].device, dtype=fused["opacity"].dtype
            )
            norm = max(float(args.fusion_voxel_score_opacity_norm), 1e-6)
            power = max(float(args.fusion_voxel_score_opacity_power), 1e-6)
            floor = min(max(float(args.fusion_voxel_score_opacity_floor), 0.0), 1.0)
            gate = (score.clamp_min(0.0) / norm).clamp(0.0, 1.0).pow(power)
            gate = floor + (1.0 - floor) * gate
            fused = dict(fused)
            fused["opacity"] = fused["opacity"] * gate
        if sparse_voxel_fusion_head is not None and apply_sparse_refine:
            fused = sparse_voxel_fusion_head.refine(fused, voxel_size, float(radius))
        return fused

    def _filter_depth_consistency(parts: list[dict], anchor_ids: list[int],
                                  w2c_all: torch.Tensor | None,
                                  K_all: torch.Tensor,
                                  depths: torch.Tensor | None,
                                  fg: torch.Tensor,
                                  radius: float) -> list[dict]:
        if not args.fusion_depth_filter or w2c_all is None or depths is None:
            return parts
        h, w = fg.shape[1], fg.shape[2]
        tol = args.fusion_depth_tol_frac * radius
        margin = max(args.fusion_bg_margin_px, 0)
        silhouette_weight = max(float(args.fusion_filter_silhouette_weight), 0.0)
        front_weight = max(float(args.fusion_filter_front_weight), 0.0)
        filtered = []
        with torch.no_grad():
            centers = None
            if args.fusion_filter_nearest_refs > 0:
                R = w2c_all[:, :3, :3]
                t = w2c_all[:, :3, 3]
                centers = -(R.transpose(1, 2) @ t[..., None]).squeeze(-1)
            for part, src_view in zip(parts, anchor_ids):
                means = part["mean"].detach()
                keep = torch.ones(means.shape[0], dtype=torch.bool, device=means.device)
                conflicts = torch.zeros(means.shape[0], dtype=means.dtype, device=means.device)
                supports = torch.zeros(means.shape[0], dtype=means.dtype, device=means.device)
                if args.fusion_filter_nearest_refs > 0 and centers is not None:
                    candidates = [v for v in anchor_ids if v != src_view and v < depths.shape[0]]
                    if candidates:
                        cand_t = torch.as_tensor(candidates, device=centers.device, dtype=torch.long)
                        d = torch.linalg.norm(centers[cand_t] - centers[src_view:src_view + 1], dim=1)
                        order = d.argsort()[:args.fusion_filter_nearest_refs].tolist()
                        ref_views = [candidates[i] for i in order]
                    else:
                        ref_views = []
                elif args.fusion_filter_all_views:
                    ref_views = list(range(min(depths.shape[0], K_all.shape[0])))
                else:
                    ref_views = anchor_ids
                support_views = ref_views
                if args.fusion_min_support > 0 and src_view < depths.shape[0]:
                    support_views = [src_view] + [v for v in ref_views if v != src_view]
                for ref_view in support_views:
                    if ref_view >= depths.shape[0]:
                        continue
                    cam = means @ w2c_all[ref_view, :3, :3].T + w2c_all[ref_view, :3, 3]
                    z = cam[:, 2]
                    fx, fy = K_all[ref_view, 0, 0], K_all[ref_view, 1, 1]
                    cx, cy = K_all[ref_view, 0, 2], K_all[ref_view, 1, 2]
                    u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
                    v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
                    inb = (z > 1e-6) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
                    if not inb.any():
                        continue

                    fg_view = fg[ref_view, ..., 0].to(device=means.device) > 0.5
                    if margin > 0:
                        fg_view = Fnn.max_pool2d(
                            fg_view.float()[None, None],
                            kernel_size=2 * margin + 1,
                            stride=1,
                            padding=margin,
                        )[0, 0] > 0.5
                    z_view = depths[ref_view].to(device=means.device, dtype=means.dtype)
                    valid_view = (z_view < 1e5) & fg_view

                    idx = torch.nonzero(inb, as_tuple=False).squeeze(1)
                    ui = u[idx].round().long().clamp(0, w - 1)
                    vi = v[idx].round().long().clamp(0, h - 1)
                    sampled_fg = fg_view[vi, ui]
                    sampled_valid = valid_view[vi, ui]
                    sampled_z = z_view[vi, ui]
                    depth_match = sampled_valid & ((z[idx] - sampled_z).abs() <= tol)
                    if args.fusion_min_support > 0 and depth_match.any():
                        supports[idx[depth_match]] += 1.0

                    # A real object point should never project outside another
                    # anchor's silhouette. If it projects onto foreground, prune
                    # only when it is in front of the known visible depth; points
                    # behind that surface may simply be occluded and should stay.
                    if ref_view == src_view:
                        continue
                    if args.fusion_depth_bidirectional:
                        depth_conflict = sampled_valid & ((z[idx] - sampled_z).abs() > tol)
                    else:
                        depth_conflict = sampled_valid & (z[idx] < sampled_z - tol)
                    conflict_score = conflicts.new_zeros(idx.shape)
                    if silhouette_weight > 0:
                        conflict_score = torch.maximum(
                            conflict_score,
                            (~sampled_fg).to(dtype=conflict_score.dtype) * silhouette_weight,
                        )
                    if front_weight > 0:
                        conflict_score = torch.maximum(
                            conflict_score,
                            depth_conflict.to(dtype=conflict_score.dtype) * front_weight,
                        )
                    conflict = conflict_score > 0
                    if conflict.any():
                        conflict_idx = idx[conflict]
                        keep[conflict_idx] = False
                        conflicts[conflict_idx] += conflict_score[conflict]
                if args.fusion_min_support > 0:
                    missing_support = (float(args.fusion_min_support) - supports).clamp_min(0.0)
                    keep = keep & (supports >= float(args.fusion_min_support))
                else:
                    missing_support = conflicts.new_zeros(conflicts.shape)
                if args.fusion_filter_mode == "opacity":
                    penalty = conflicts + missing_support
                    decay = torch.exp(-args.fusion_conflict_opacity_decay * penalty).unsqueeze(-1)
                    part_soft = dict(part)
                    part_soft["opacity"] = part["opacity"] * decay
                    filtered.append(part_soft)
                else:
                    filtered.append(_filter_param_keep(part, keep) if keep.any() else part)
        return filtered

    def _apply_oracle_depth(p: dict, K_i: torch.Tensor, c2w_i: torch.Tensor,
                            depth_i: torch.Tensor | None, mask_i: torch.Tensor,
                            radius: float) -> dict:
        if not args.oracle_anchor_depth or depth_i is None:
            return p
        from decoder.clean.geometry import ray_dirs_world

        target_t, valid_t = depth_target_on_grid(
            depth_i, mask_i[..., 0] > 0.5, K_i, model.map_h, model.map_w
        )
        if not valid_t.any():
            return p
        dirs = ray_dirs_world(K_i, c2w_i, model.map_h, model.map_w).to(
            device=p["mean"].device, dtype=p["mean"].dtype
        )
        origins = c2w_i[:3, 3].to(device=p["mean"].device, dtype=p["mean"].dtype).expand_as(dirs)
        mean = origins + target_t.to(device=p["mean"].device, dtype=p["mean"].dtype).unsqueeze(-1) * dirs
        valid = valid_t.to(device=p["mean"].device)
        p = {k: v.clone() for k, v in p.items()}
        p["mean"][valid] = mean[valid]
        p["mean_anchor"][valid] = mean[valid]
        p["depth"][valid] = target_t.to(device=p["depth"].device, dtype=p["depth"].dtype)[valid].unsqueeze(-1)
        return p

    def _apply_surface_token_view_gates(
        p: dict,
        view_gates: torch.Tensor | None,
        n_views: int,
    ) -> dict:
        if view_gates is None:
            return p
        base_n = int(n_views) * int(args.surface_token_grid_h) * int(args.surface_token_grid_w)
        if base_n <= 0 or "opacity" not in p or p["opacity"].shape[0] < base_n:
            return p
        total_n = int(p["opacity"].shape[0])
        if total_n % base_n != 0:
            return p
        gate = view_gates.reshape(int(n_views), 1, 1).expand(
            int(n_views), int(args.surface_token_grid_h) * int(args.surface_token_grid_w), 1
        ).reshape(base_n, 1)
        repeats = total_n // base_n
        if repeats > 1:
            gate = gate.repeat(repeats, 1)
        gate = gate.to(device=p["opacity"].device, dtype=p["opacity"].dtype)
        out = dict(p)
        out["opacity"] = (out["opacity"] * gate).clamp(0.0, 1.0)
        out["_surface_token_global_view_gate"] = gate
        return out

    def _predict_anchor_parts(latent: torch.Tensor, K_all: torch.Tensor, c2w_all: torch.Tensor,
                              frames: torch.Tensor, fg: torch.Tensor, radius: float,
                              w2c_all: torch.Tensor | None = None,
                              depths: torch.Tensor | None = None,
                              visibility_depths: torch.Tensor | None = None,
                              target_depths: torch.Tensor | None = None,
                              target_frames: torch.Tensor | None = None,
                              confs: torch.Tensor | None = None,
                              spread_anchors: bool = False,
                              disable_surface_token_new_capacity: bool = False) -> tuple[
                                  list[dict], list[int], torch.Tensor | None, torch.Tensor | None
                              ]:
        if spread_anchors:
            anchor_ids = _eval_anchor_indices(K_all.shape[0], args.anchor_views)
        else:
            anchor_ids = list(range(min(max(args.anchor_views, 1), K_all.shape[0])))
        image_depths = depths
        if args.image_hull_clamp_depth:
            if w2c_all is None:
                raise ValueError("--image_hull_clamp_depth requires w2c_all")
            image_depths = _hull_clamped_depths(
                image_depths, fg, K_all, c2w_all, w2c_all, radius
            )
        if args.image_guided_plane_sweep_depth:
            if w2c_all is None:
                raise ValueError("--image_guided_plane_sweep_depth requires w2c_all")
            image_depths = _guided_plane_sweep_depths(
                frames, fg, image_depths, K_all, c2w_all, w2c_all, radius
            )
        elif args.image_plane_sweep_depth:
            if w2c_all is None:
                raise ValueError("--image_plane_sweep_depth requires w2c_all")
            image_depths = _plane_sweep_depths(frames, fg, K_all, c2w_all, w2c_all, radius)
        elif args.image_voxel_hull_depth:
            if w2c_all is None:
                raise ValueError("--image_voxel_hull_depth requires w2c_all")
            image_depths = _voxel_hull_depths(fg, K_all, c2w_all, w2c_all, radius)
        elif args.image_visual_hull_depth:
            if w2c_all is None:
                raise ValueError("--image_visual_hull_depth requires w2c_all")
            image_depths = _visual_hull_depths(fg, K_all, c2w_all, w2c_all, radius)
        consistency_depths = visibility_depths if visibility_depths is not None else depths
        filter_depths = image_depths if (
            args.image_guided_plane_sweep_depth or args.image_plane_sweep_depth
            or args.image_voxel_hull_depth or args.image_visual_hull_depth
        ) else consistency_depths
        if canonical_voxel_decoder is not None:
            if image_depths is None:
                raise ValueError("--use_canonical_voxel_decoder requires conditioning depths")
            if not anchor_ids:
                raise ValueError("--use_canonical_voxel_decoder resolved no anchor views")
            ids = torch.as_tensor(anchor_ids, device=frames.device, dtype=torch.long)
            p_vox = canonical_voxel_decoder(
                latent,
                frames.index_select(0, ids),
                fg.index_select(0, ids),
                image_depths.index_select(0, ids),
                K_all.index_select(0, ids),
                c2w_all.index_select(0, ids),
                radius,
                w2c_all.index_select(0, ids) if w2c_all is not None else None,
            )
            if args.canonical_source_vis_gate or args.canonical_source_vis_learned_refine:
                if w2c_all is None:
                    raise ValueError("canonical source visibility refinement requires w2c cameras")
                src_confs = (
                    confs.index_select(0, ids)
                    if confs is not None and confs.shape[0] >= int(ids.max()) + 1
                    else None
                )
                p_scored = _with_depth_consistency_score(
                    p_vox,
                    frames.index_select(0, ids),
                    fg.index_select(0, ids),
                    image_depths.index_select(0, ids),
                    src_confs,
                    w2c_all.index_select(0, ids),
                    K_all.index_select(0, ids),
                    radius,
                )
                if args.canonical_source_vis_learned_refine:
                    p_scored = canonical_voxel_decoder.refine_source_consistency(
                        p_scored, radius
                    )
                if args.canonical_source_vis_gate:
                    support = p_scored.get("_fusion_support")
                    conflict = p_scored.get("_fusion_conflict")
                    if support is not None and conflict is not None:
                        score = support - float(args.canonical_source_vis_conflict_weight) * conflict
                        gate = torch.sigmoid(
                            (score - float(args.canonical_source_vis_min_support))
                            / max(float(args.canonical_source_vis_softness), 1e-6)
                        )
                        floor = min(max(float(args.canonical_source_vis_floor), 0.0), 1.0)
                        gate = floor + (1.0 - floor) * gate
                        p_scored["opacity"] = p_scored["opacity"] * gate.to(
                            device=p_scored["opacity"].device,
                            dtype=p_scored["opacity"].dtype,
                        )
                        p_scored["_canonical_source_vis_gate"] = gate
                p_vox = p_scored
            return [p_vox], [anchor_ids[0]], image_depths, filter_depths
        if surface_token_decoder is not None:
            if image_depths is None:
                raise ValueError("--use_surface_token_decoder requires conditioning depths")
            if not anchor_ids:
                raise ValueError("--use_surface_token_decoder resolved no anchor views")
            view_gates = None
            if (surface_token_view_selector is not None
                    and not disable_surface_token_new_capacity):
                prior_ids = torch.as_tensor(anchor_ids, device=frames.device, dtype=torch.long)
                selected = surface_token_view_selector.select(
                    latent,
                    frames,
                    fg,
                    image_depths,
                    K_all,
                    c2w_all,
                    radius,
                    k=args.anchor_views,
                    prior_ids=prior_ids,
                    train_noise=(
                        args.surface_token_view_selector_train_noise
                        if surface_token_view_selector.training else 0.0
                    ),
                )
                ids = selected["ids"].to(device=frames.device, dtype=torch.long)
                anchor_ids = [int(i) for i in ids.detach().cpu().tolist()]
                view_gates = selected["gates"]
            else:
                ids = torch.as_tensor(anchor_ids, device=frames.device, dtype=torch.long)
            p_tok = surface_token_decoder(
                latent,
                frames.index_select(0, ids),
                fg.index_select(0, ids),
                image_depths.index_select(0, ids),
                K_all.index_select(0, ids),
                c2w_all.index_select(0, ids),
                radius,
                disable_new_capacity=disable_surface_token_new_capacity,
            )
            p_tok = _apply_surface_token_view_gates(p_tok, view_gates, ids.numel())
            p_tok["_surface_token_selected_view_ids"] = ids.detach().to(
                device=p_tok["mean"].device
            )
            p_tok["_surface_token_candidate_view_count"] = p_tok["mean"].new_tensor(
                float(frames.shape[0])
            )
            return [p_tok], [anchor_ids[0]], image_depths, filter_depths
        surface_mv_features = None
        if ((surface_confidence_head is not None or surface_refine_head is not None)
                and image_depths is not None):
            with torch.no_grad():
                surface_mv_features = _depth_multiview_support_maps(
                    image_depths.detach(),
                    fg,
                    K_all,
                    c2w_all,
                    radius,
                    args.surface_confidence_multiview_tol_frac,
                    args.surface_confidence_multiview_refs,
                    args.surface_confidence_multiview_radius_px,
                )
        parts = []
        for ai in anchor_ids:
            out = model(latent, K_all[ai], c2w_all[ai], radius,
                        image_cond=_image_cond(frames, fg, ai, depths=image_depths,
                                               confs=confs,
                                               K_all=K_all, c2w_all=c2w_all,
                                               w2c_all=w2c_all,
                                               radius=radius))
            p_i = {k: v[0] for k, v in out.items()}
            detail_i = _fusion_detail_flat(frames[ai], fg[ai])
            if detail_i is not None:
                p_i["_fusion_detail"] = detail_i.to(
                    device=p_i["mean"].device, dtype=p_i["mean"].dtype
                )
            d_i = depths[ai] if depths is not None and ai < depths.shape[0] else None
            p_i = _apply_support_gate(
                p_i, ai, frames, fg, depths, target_depths, K_all, c2w_all, w2c_all, radius
            )
            p_i = _apply_surface_confidence_gate(
                p_i, ai, frames, fg, image_depths, target_depths, K_all, c2w_all,
                w2c_all, radius, surface_mv_features,
            )
            p_i = _apply_surface_refine(
                p_i, ai, frames, fg, image_depths, target_frames, K_all, c2w_all,
                radius, surface_mv_features,
            )
            parts.append(_apply_oracle_depth(p_i, K_all[ai], c2w_all[ai], d_i, fg[ai], radius))
        parts = _filter_depth_consistency(parts, anchor_ids, w2c_all, K_all, filter_depths, fg, radius)
        return parts, anchor_ids, image_depths, filter_depths

    def _predict_params(latent: torch.Tensor, K_all: torch.Tensor, c2w_all: torch.Tensor,
                        frames: torch.Tensor, fg: torch.Tensor, radius: float,
                        depths: torch.Tensor | None = None,
                        spread_anchors: bool = False) -> dict:
        parts, _, _, _ = _predict_anchor_parts(
            latent, K_all, c2w_all, frames, fg, radius,
            w2c_all=None,
            depths=depths,
            visibility_depths=depths,
            target_depths=depths,
            confs=None,
            spread_anchors=spread_anchors,
        )
        return parts[0] if len(parts) == 1 else _cat_params(parts)

    def _train_anchor_ids(n_views: int) -> list[int]:
        return list(range(min(max(args.anchor_views, 1), n_views)))

    def _nearest_anchor_indices(target_c2w: torch.Tensor, anchor_c2w: torch.Tensor) -> list[int]:
        if anchor_c2w.shape[0] == 1:
            return [0] * target_c2w.shape[0]
        centers = target_c2w[:, :3, 3]
        anchor_centers = anchor_c2w[:, :3, 3]
        dist = torch.cdist(centers, anchor_centers)
        return dist.argmin(dim=1).tolist()

    def _render_with_anchor_mode(parts: list[dict], anchor_ids: list[int],
                                 w2c_all: torch.Tensor, K_all: torch.Tensor,
                                 c2w_all: torch.Tensor, w: int, h: int,
                                 bg: float,
                                 anchor_c2w_all: torch.Tensor | None = None,
                                 radius: float | None = None,
                                 source_frames: torch.Tensor | None = None,
                                 source_fg: torch.Tensor | None = None,
                                 source_depths: torch.Tensor | None = None,
                                 source_visibility_depths: torch.Tensor | None = None,
                                 source_target_depths: torch.Tensor | None = None,
                                 source_confs: torch.Tensor | None = None,
                                 source_w2c: torch.Tensor | None = None,
                                 source_K: torch.Tensor | None = None,
                                 source_c2w: torch.Tensor | None = None,
                                 target_frames: torch.Tensor | None = None,
                                 target_fg: torch.Tensor | None = None,
                                 apply_sparse_voxel_refine: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        sh_degree = args.fusion_sh_degree if args.fusion_sh_degree > 0 else None
        source_support_depths = (
            source_visibility_depths if source_visibility_depths is not None else source_depths
        )

        def apply_output_hull(r_i: torch.Tensor, a_i: torch.Tensor,
                              view_i: int) -> tuple[torch.Tensor, torch.Tensor]:
            if not args.anchor_output_hull_mask:
                return r_i, a_i
            if (source_fg is None or source_K is None or source_w2c is None
                    or source_c2w is None or radius is None):
                raise ValueError("--anchor_output_hull_mask requires source masks/cameras")
            hull_mask = _target_visual_hull_masks(
                source_fg, source_K, source_c2w, source_w2c,
                K_all[view_i:view_i + 1],
                c2w_all[view_i:view_i + 1],
                w2c_all[view_i:view_i + 1],
                float(radius), h, w,
            )[0].to(device=r_i.device, dtype=r_i.dtype)
            r_i = r_i * hull_mask + r_i.new_full((), bg) * (1.0 - hull_mask)
            a_i = a_i * hull_mask
            return r_i, a_i

        def apply_output_alpha_cleanup(r_i: torch.Tensor,
                                       a_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Soft final alpha gate for view-conditioned fringe cleanup.

            This is feed-forward image-space compositing, not per-object
            optimization. It is intended for low-confidence splat debris that
            survives RGBD shell fusion around silhouettes.
            """
            min_alpha = max(float(args.output_alpha_cleanup_min), 0.0)
            erode_px = max(int(args.output_alpha_cleanup_erode_px), 0)
            dilate_px = max(int(args.output_alpha_cleanup_dilate_px), 0)
            if min_alpha <= 0 and erode_px <= 0 and dilate_px <= 0:
                return r_i, a_i
            gate = a_i[..., 0].clamp(0.0, 1.0)
            if min_alpha > 0:
                softness = max(float(args.output_alpha_cleanup_softness), 1e-6)
                gate = ((gate - min_alpha) / softness).clamp(0.0, 1.0)
                gate = gate * gate * (3.0 - 2.0 * gate)
            if erode_px > 0:
                gate_n = gate[None, None]
                gate = -Fnn.max_pool2d(
                    -gate_n, kernel_size=2 * erode_px + 1, stride=1, padding=erode_px
                )[0, 0]
            if dilate_px > 0:
                gate = Fnn.max_pool2d(
                    gate[None, None], kernel_size=2 * dilate_px + 1,
                    stride=1, padding=dilate_px,
                )[0, 0]
            gate = gate.clamp(0.0, 1.0)[..., None]
            r_i = r_i * gate + r_i.new_full((), bg) * (1.0 - gate)
            a_i = a_i * gate
            return r_i, a_i

        def apply_output_alpha_refine(r_i: torch.Tensor,
                                      a_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if output_alpha_refine_head is None:
                return r_i, a_i
            r_o, a_o, _, delta = apply_output_alpha_refiner(
                output_alpha_refine_head,
                r_i,
                a_i,
                bg,
                args.output_alpha_refine_delta_scale,
                init=args.output_alpha_refine_init,
                floor=args.output_alpha_refine_floor,
            )
            if torch.is_grad_enabled():
                output_alpha_refine_delta_terms.append(delta.square().mean())
                if delta.shape[-1] > 1 and delta.shape[-2] > 1:
                    tv_x = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
                    tv_y = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
                    output_alpha_refine_tv_terms.append(0.5 * (tv_x + tv_y))
            return r_o, a_o

        if args.anchor_render_mode == "tsdf":
            if radius is None:
                raise ValueError("--anchor_render_mode tsdf requires radius")
            if (source_frames is None or source_fg is None or source_depths is None
                    or source_w2c is None or source_K is None or source_c2w is None):
                raise ValueError("--anchor_render_mode tsdf requires source RGBD conditioning")
            p_tsdf, _ = rgbd_tsdf_fuse(
                source_frames, source_fg, source_depths, source_K, source_w2c,
                source_c2w,
                voxel_size=args.tsdf_voxel_size_frac * float(radius),
                trunc_mult=args.tsdf_trunc_mult,
                min_weight=args.tsdf_min_weight,
                surface_thresh=args.tsdf_surface_thresh,
                max_voxels=args.tsdf_max_voxels,
                max_points=args.tsdf_max_points,
                scale_mult=args.tsdf_scale_mult,
                normal_scale_mult=args.tsdf_normal_scale_mult,
                opacity=args.tsdf_opacity,
                color_mode=args.tsdf_color_mode,
                surface_mode=args.tsdf_surface_mode,
            )
            # Keep the training graph formally connected in one-step diagnostics.
            zero_dep = sum((p["mean"].sum() * 0.0 for p in parts), parts[0]["mean"].new_zeros(()))
            p_tsdf = {
                k: (v + zero_dep if torch.is_floating_point(v) else v)
                for k, v in p_tsdf.items()
            }
            if args.fusion_sh_degree > 0:
                p_tsdf, _ = rgbd_fit_sh_colors(
                    p_tsdf, source_frames, source_fg, source_depths, source_K,
                    source_w2c,
                    degree=args.fusion_sh_degree,
                    depth_tol=args.fusion_sh_depth_tol_frac * float(radius),
                    ridge=args.fusion_sh_ridge,
                    min_obs=args.fusion_sh_min_obs,
                    mix=args.fusion_sh_mix,
                )
            return _render_params(p_tsdf, w2c_all, K_all, w, h, bg=bg, sh_degree=sh_degree)
        target_mode = {
            "nearest_fill", "learned_fill", "zselect", "zselect_fill",
            "maxalpha", "maxalpha_fill", "iblend", "iblend_fill",
            "learned_iblend_fill", "target_rgbd", "target_rgbd_fill",
            "target_rgbd_splat", "target_rgbd_splat_fill",
            "iblend_surf_gate", "iblend_surf_gate_fill", "iblend_tsurf_fill",
            "blend", "nearest",
        }
        if args.anchor_render_mode == "concat" or (
            len(parts) == 1 and args.anchor_render_mode not in target_mode
        ):
            src_parts = _with_fusion_sources(parts, anchor_ids)
            p_all = src_parts[0] if len(src_parts) == 1 else _cat_params(src_parts)
            if args.fusion_tsdf_filter:
                if radius is None:
                    raise ValueError("--fusion_tsdf_filter requires radius")
                if (source_frames is None or source_fg is None or source_depths is None
                        or source_w2c is None or source_K is None or source_c2w is None):
                    raise ValueError("--fusion_tsdf_filter requires source RGBD conditioning")
                p_all, _ = rgbd_tsdf_filter_params(
                    p_all, source_frames, source_fg, source_depths, source_K,
                    source_w2c, source_c2w,
                    voxel_size=args.tsdf_voxel_size_frac * float(radius),
                    trunc_mult=args.tsdf_trunc_mult,
                    min_weight=args.tsdf_min_weight,
                    band=args.fusion_tsdf_band,
                    opacity_decay=args.fusion_tsdf_opacity_decay,
                    invalid_opacity_mult=args.fusion_tsdf_invalid_opacity_mult,
                    max_voxels=args.tsdf_max_voxels,
                )
            p_all = _with_depth_consistency_score(
                p_all, source_frames, source_fg, source_support_depths, source_confs,
                source_w2c, source_K, radius,
                target_depths_for_candidate=source_target_depths,
            )
            p_all = _maybe_voxel_fuse_params(
                p_all, radius,
                apply_sparse_refine=apply_sparse_voxel_refine,
            )
            if args.fusion_sh_degree > 0:
                if radius is None:
                    raise ValueError("--fusion_sh_degree requires radius")
                if (source_frames is None or source_fg is None or source_depths is None
                        or source_w2c is None or source_K is None):
                    raise ValueError("--fusion_sh_degree requires source RGBD conditioning")
                p_all, _ = rgbd_fit_sh_colors(
                    p_all, source_frames, source_fg, source_depths, source_K,
                    source_w2c,
                    degree=args.fusion_sh_degree,
                    depth_tol=args.fusion_sh_depth_tol_frac * float(radius),
                    ridge=args.fusion_sh_ridge,
                    min_obs=args.fusion_sh_min_obs,
                    mix=args.fusion_sh_mix,
                )
            return _render_params(p_all, w2c_all, K_all, w, h, bg=bg, sh_degree=sh_degree)
        static_fill = None
        if args.anchor_render_mode in {
            "nearest_fill", "learned_fill", "zselect_fill", "maxalpha_fill",
            "iblend_fill", "learned_iblend_fill", "target_rgbd_fill",
            "target_rgbd_splat_fill", "iblend_surf_gate_fill"
        }:
            src_parts = _with_fusion_sources(parts, anchor_ids)
            p_all = src_parts[0] if len(src_parts) == 1 else _cat_params(src_parts)
            if args.fusion_tsdf_filter:
                if radius is None:
                    raise ValueError("--fusion_tsdf_filter requires radius")
                if (source_frames is None or source_fg is None or source_depths is None
                        or source_w2c is None or source_K is None or source_c2w is None):
                    raise ValueError("--fusion_tsdf_filter requires source RGBD conditioning")
                p_all, _ = rgbd_tsdf_filter_params(
                    p_all, source_frames, source_fg, source_depths, source_K,
                    source_w2c, source_c2w,
                    voxel_size=args.tsdf_voxel_size_frac * float(radius),
                    trunc_mult=args.tsdf_trunc_mult,
                    min_weight=args.tsdf_min_weight,
                    band=args.fusion_tsdf_band,
                    opacity_decay=args.fusion_tsdf_opacity_decay,
                    invalid_opacity_mult=args.fusion_tsdf_invalid_opacity_mult,
                    max_voxels=args.tsdf_max_voxels,
                )
            p_all = _with_depth_consistency_score(
                p_all, source_frames, source_fg, source_support_depths, source_confs,
                source_w2c, source_K, radius,
                target_depths_for_candidate=source_target_depths,
            )
            p_all = _maybe_voxel_fuse_params(
                p_all, radius,
                apply_sparse_refine=apply_sparse_voxel_refine,
            )
            if args.fusion_sh_degree > 0:
                if radius is None:
                    raise ValueError("--fusion_sh_degree requires radius")
                if (source_frames is None or source_fg is None or source_depths is None
                        or source_w2c is None or source_K is None):
                    raise ValueError("--fusion_sh_degree requires source RGBD conditioning")
                p_all, _ = rgbd_fit_sh_colors(
                    p_all, source_frames, source_fg, source_depths, source_K,
                    source_w2c,
                    degree=args.fusion_sh_degree,
                    depth_tol=args.fusion_sh_depth_tol_frac * float(radius),
                    ridge=args.fusion_sh_ridge,
                    min_obs=args.fusion_sh_min_obs,
                    mix=args.fusion_sh_mix,
                )
            static_fill = p_all
        if args.anchor_render_mode in {
            "target_rgbd", "target_rgbd_fill", "target_rgbd_splat", "target_rgbd_splat_fill"
        }:
            if radius is None:
                raise ValueError("--anchor_render_mode target_rgbd requires radius")
            if (source_frames is None or source_fg is None or source_depths is None
                    or source_w2c is None or source_K is None or source_c2w is None):
                raise ValueError("--anchor_render_mode target_rgbd requires source RGBD conditioning")
            renders, alphas = [], []
            scale_frac = (
                args.target_surface_scale_frac
                if args.target_surface_scale_frac > 0 else args.image_scale_frac
            )
            normal_scale_frac = (
                args.target_surface_normal_scale_frac
                if args.target_surface_normal_scale_frac > 0
                else (args.image_normal_scale_frac if args.image_normal_scale_frac > 0 else scale_frac)
            )
            opacity = (
                args.target_surface_opacity
                if args.target_surface_opacity > 0 else args.image_opacity_fg
            )
            for view_i in range(w2c_all.shape[0]):
                surface_fn = (
                    rgbd_target_view_surface_splat
                    if args.anchor_render_mode in {"target_rgbd_splat", "target_rgbd_splat_fill"}
                    else rgbd_target_view_surface
                )
                kwargs = {}
                if surface_fn is rgbd_target_view_surface_splat:
                    kwargs["view_weight_temp"] = (
                        max(float(args.target_surface_view_weight_temp_frac), 0.0)
                        * float(radius)
                    )
                    kwargs["min_support"] = args.target_surface_min_support
                    support_tol_frac = (
                        args.target_surface_support_tol_frac
                        if args.target_surface_support_tol_frac > 0
                        else args.target_surface_depth_tol_frac
                    )
                    kwargs["support_depth_tol"] = support_tol_frac * float(radius)
                p_i, _ = surface_fn(
                    source_frames, source_fg, source_depths, source_K, source_w2c,
                    source_c2w, K_all[view_i], w2c_all[view_i], c2w_all[view_i],
                    w, h, float(radius), scale_frac, normal_scale_frac,
                    opacity=opacity,
                    depth_tol=args.target_surface_depth_tol_frac * float(radius),
                    **kwargs,
                )
                r_i, a_i = _render_params(
                    p_i, w2c_all[view_i:view_i + 1],
                    K_all[view_i:view_i + 1], w, h, bg=bg,
                    sh_degree=None,
                )
                r_i, a_i = r_i[0], a_i[0]
                if static_fill is not None:
                    r_s, a_s = _render_params(
                        static_fill, w2c_all[view_i:view_i + 1],
                        K_all[view_i:view_i + 1], w, h, bg=bg,
                        sh_degree=sh_degree,
                    )
                    r_i, a_i = _blend_static_fill(
                        r_i, a_i, r_s[0], a_s[0],
                        args.anchor_fill_alpha_power,
                        args.anchor_fill_static_alpha_min,
                        args.anchor_fill_static_alpha_softness,
                    )
                r_i, a_i = apply_output_hull(r_i, a_i, view_i)
                r_i, a_i = apply_output_alpha_cleanup(r_i, a_i)
                r_i, a_i = apply_output_alpha_refine(r_i, a_i)
                renders.append(r_i[None])
                alphas.append(a_i[None])
            return torch.cat(renders, 0), torch.cat(alphas, 0)

        def _render_target_surface_for_view(view_i: int, dtype: torch.dtype,
                                            device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
            if radius is None:
                raise ValueError("target surface rendering requires radius")
            if (source_frames is None or source_fg is None or source_depths is None
                    or source_w2c is None or source_K is None or source_c2w is None):
                raise ValueError("target surface rendering requires source RGBD conditioning")
            scale_frac = (
                args.target_surface_scale_frac
                if args.target_surface_scale_frac > 0 else args.image_scale_frac
            )
            normal_scale_frac = (
                args.target_surface_normal_scale_frac
                if args.target_surface_normal_scale_frac > 0
                else (args.image_normal_scale_frac if args.image_normal_scale_frac > 0 else scale_frac)
            )
            opacity = (
                args.target_surface_opacity
                if args.target_surface_opacity > 0 else args.image_opacity_fg
            )
            support_tol_frac = (
                args.target_surface_support_tol_frac
                if args.target_surface_support_tol_frac > 0
                else args.target_surface_depth_tol_frac
            )
            p_gate, _ = rgbd_target_view_surface_splat(
                source_frames, source_fg, source_depths, source_K, source_w2c,
                source_c2w, K_all[view_i], w2c_all[view_i], c2w_all[view_i],
                w, h, float(radius), scale_frac, normal_scale_frac,
                opacity=opacity,
                depth_tol=args.target_surface_depth_tol_frac * float(radius),
                view_weight_temp=max(float(args.target_surface_view_weight_temp_frac), 0.0) * float(radius),
                min_support=args.target_surface_min_support,
                support_depth_tol=support_tol_frac * float(radius),
            )
            r_t, a_t = _render_params(
                p_gate, w2c_all[view_i:view_i + 1],
                K_all[view_i:view_i + 1], w, h, bg=bg,
                sh_degree=None,
            )
            return (
                r_t[0].to(device=device, dtype=dtype),
                a_t[0].to(device=device, dtype=dtype),
                float(opacity),
            )

        def _target_surface_gate(view_i: int, dtype: torch.dtype,
                                 device: torch.device) -> torch.Tensor:
            _, a_gate, opacity = _render_target_surface_for_view(view_i, dtype, device)
            denom = max(float(opacity), 1e-6)
            gate = (a_gate / denom).clamp(0.0, 1.0)
            if args.target_surface_gate_dilate_px > 0:
                pad = int(args.target_surface_gate_dilate_px)
                gate_nchw = gate.permute(2, 0, 1)[None]
                gate = Fnn.max_pool2d(
                    gate_nchw, kernel_size=2 * pad + 1, stride=1, padding=pad
                )[0].permute(1, 2, 0)
            power = max(float(args.target_surface_gate_power), 1e-6)
            return gate.pow(power)

        def _candidate_source_support(view_i: int, depth_z: torch.Tensor,
                                      alpha: torch.Tensor) -> torch.Tensor:
            if args.anchor_iblend_support_weight <= 0:
                return alpha.new_ones(alpha.shape)
            if radius is None:
                raise ValueError("--anchor_iblend_support_weight requires radius")
            if (source_fg is None or source_support_depths is None or source_w2c is None
                    or source_K is None or source_c2w is None):
                raise ValueError("--anchor_iblend_support_weight requires source RGBD conditioning")
            with torch.no_grad():
                dtype, device = alpha.dtype, alpha.device
                depth = depth_z.to(device=device, dtype=dtype)
                valid = (
                    torch.isfinite(depth)
                    & (depth > 1e-6)
                    & (alpha[..., 0] > args.anchor_zselect_alpha_min)
                )
                n_ref = min(source_support_depths.shape[0], source_w2c.shape[0], source_K.shape[0])
                if n_ref <= 0:
                    return alpha.new_ones(alpha.shape)
                refs = torch.arange(n_ref, device=device, dtype=torch.long)
                max_refs = int(args.anchor_iblend_support_refs)
                if max_refs > 0 and n_ref > max_refs:
                    centers = source_c2w[:n_ref, :3, 3].to(device=device, dtype=dtype)
                    tgt_center = c2w_all[view_i, :3, 3].to(device=device, dtype=dtype)
                    order = torch.linalg.norm(centers - tgt_center[None], dim=1).argsort()[:max_refs]
                    refs = refs[order]

                yy, xx = torch.meshgrid(
                    torch.arange(h, device=device, dtype=dtype),
                    torch.arange(w, device=device, dtype=dtype),
                    indexing="ij",
                )
                K_t = K_all[view_i].to(device=device, dtype=dtype)
                z_flat = depth.reshape(-1)
                x = (xx.reshape(-1) - K_t[0, 2]) / K_t[0, 0].clamp_min(1e-6) * z_flat
                y = (yy.reshape(-1) - K_t[1, 2]) / K_t[1, 1].clamp_min(1e-6) * z_flat
                pts_cam = torch.stack([x, y, z_flat], dim=-1)
                c2w_t = c2w_all[view_i].to(device=device, dtype=dtype)
                pts = pts_cam @ c2w_t[:3, :3].T + c2w_t[:3, 3]

                valid_flat = valid.reshape(-1)
                supports = torch.zeros(h * w, device=device, dtype=dtype)
                conflicts = torch.zeros(h * w, device=device, dtype=dtype)
                tol_frac = (
                    args.anchor_iblend_support_tol_frac
                    if args.anchor_iblend_support_tol_frac > 0
                    else args.anchor_iblend_depth_tol_frac
                )
                tol = max(float(tol_frac) * float(radius), 1e-6)
                margin = max(args.fusion_bg_margin_px, 0)
                for ref_view_t in refs:
                    ref_view = int(ref_view_t.item())
                    cam = pts @ source_w2c[ref_view, :3, :3].to(device=device, dtype=dtype).T
                    cam = cam + source_w2c[ref_view, :3, 3].to(device=device, dtype=dtype)
                    z = cam[:, 2]
                    K_ref = source_K[ref_view].to(device=device, dtype=dtype)
                    u = K_ref[0, 0] * (cam[:, 0] / z.clamp_min(1e-6)) + K_ref[0, 2]
                    v = K_ref[1, 1] * (cam[:, 1] / z.clamp_min(1e-6)) + K_ref[1, 2]
                    inb = (
                        valid_flat
                        & (z > 1e-6)
                        & (u >= 0) & (u <= w - 1)
                        & (v >= 0) & (v <= h - 1)
                    )
                    if not inb.any():
                        continue
                    fg_view = source_fg[ref_view, ..., 0].to(device=device) > 0.5
                    if margin > 0:
                        fg_view = Fnn.max_pool2d(
                            fg_view.float()[None, None],
                            kernel_size=2 * margin + 1,
                            stride=1,
                            padding=margin,
                        )[0, 0] > 0.5
                    z_view = source_support_depths[ref_view].to(device=device, dtype=dtype)
                    valid_view = (z_view < 1e5) & fg_view
                    idx = torch.nonzero(inb, as_tuple=False).squeeze(1)
                    sampled_fg, depth_match, front_conflict, _ = _sample_depth_support_window(
                        fg_view,
                        z_view,
                        u[idx],
                        v[idx],
                        z[idx],
                        tol,
                        int(args.support_sample_radius_px),
                    )
                    if depth_match.any():
                        supports[idx[depth_match]] += 1.0
                    conflict = (~sampled_fg) | front_conflict
                    if conflict.any():
                        conflicts[idx[conflict]] += 1.0

                ref_count = max(int(refs.numel()), 1)
                support_frac = (supports / float(ref_count)).clamp(0.0, 1.0)
                floor = min(max(float(args.anchor_iblend_support_floor), 0.0), 1.0)
                support_score = floor + (1.0 - floor) * support_frac
                decay = max(float(args.anchor_iblend_support_decay), 0.0)
                if decay > 0:
                    support_score = support_score * torch.exp(-decay * conflicts)
                support_score = torch.where(valid_flat, support_score, torch.ones_like(support_score))
                return support_score.reshape(h, w, 1).to(dtype=alpha.dtype)

        renders, alphas = [], []
        if anchor_c2w_all is None:
            anchor_c2w = c2w_all[torch.as_tensor(anchor_ids, device=c2w_all.device)]
        else:
            anchor_c2w = anchor_c2w_all[torch.as_tensor(anchor_ids, device=anchor_c2w_all.device)]
        target_centers = c2w_all[:, :3, 3]
        anchor_centers = anchor_c2w[:, :3, 3]
        dist = torch.cdist(target_centers, anchor_centers)
        if args.anchor_render_mode in {"nearest", "nearest_fill", "learned_fill"}:
            selected = dist.argmin(dim=1).view(-1, 1)
            weights = torch.ones(selected.shape, device=dist.device, dtype=dist.dtype)
        elif args.anchor_render_mode in {
            "iblend", "iblend_fill", "learned_iblend_fill",
            "iblend_surf_gate", "iblend_surf_gate_fill", "iblend_tsurf_fill"
        }:
            k = min(max(args.anchor_blend_topk, 1), len(parts))
            vals, selected = dist.topk(k, dim=1, largest=False)
            if args.anchor_iblend_view_weight:
                temp = max(args.anchor_blend_temp, 1e-4)
                weights = torch.softmax(-vals / temp, dim=1)
            else:
                weights = torch.ones(selected.shape, device=dist.device, dtype=dist.dtype)
        elif args.anchor_render_mode in {"zselect", "zselect_fill", "maxalpha", "maxalpha_fill"}:
            k = min(max(args.anchor_blend_topk, 1), len(parts))
            _, selected = dist.topk(k, dim=1, largest=False)
            weights = torch.ones(selected.shape, device=dist.device, dtype=dist.dtype)
        else:
            k = min(max(args.anchor_blend_topk, 1), len(parts))
            vals, selected = dist.topk(k, dim=1, largest=False)
            temp = max(args.anchor_blend_temp, 1e-4)
            weights = torch.softmax(-vals / temp, dim=1)
        for view_i in range(w2c_all.shape[0]):
            if args.anchor_render_mode in {
                "iblend", "iblend_fill", "learned_iblend_fill",
                "iblend_surf_gate", "iblend_surf_gate_fill", "iblend_tsurf_fill"
            }:
                cand_obj, cand_a, cand_w, cand_d, cand_support = [], [], [], [], []
                alpha_power = max(float(args.anchor_iblend_alpha_power), 1e-6)
                depth_weight = max(float(args.anchor_iblend_depth_weight), 0.0)
                support_weight = max(float(args.anchor_iblend_support_weight), 0.0)
                if depth_weight > 0 and radius is None:
                    raise ValueError("--anchor_iblend_depth_weight requires radius")
                need_depth = (
                    depth_weight > 0
                    or support_weight > 0
                    or args.anchor_render_mode == "learned_iblend_fill"
                )
                for j in range(selected.shape[1]):
                    p_j = parts[int(selected[view_i, j])]
                    r_j, a_j = _render_params(
                        p_j, w2c_all[view_i:view_i + 1],
                        K_all[view_i:view_i + 1], w, h, bg=bg,
                        sh_degree=sh_degree,
                    )
                    r_j, a_j = r_j[0], a_j[0]
                    obj_j = (r_j - (1.0 - a_j) * bg) / a_j.clamp_min(1e-4)
                    pix_w = weights[view_i, j].to(dtype=r_j.dtype) * a_j.clamp(0.0, 1.0).pow(alpha_power)
                    cand_obj.append(obj_j)
                    cand_a.append(a_j)
                    if need_depth:
                        d_j = render_expected_depth(
                            p_j, w2c_all[view_i:view_i + 1],
                            K_all[view_i:view_i + 1], w, h, mode="ED"
                        )
                        d_map = d_j[0, ..., 0]
                        support_j = torch.ones_like(a_j)
                        if support_weight > 0:
                            support_j = _candidate_source_support(view_i, d_map, a_j)
                            pix_w = pix_w * support_j.pow(support_weight)
                        cand_support.append(support_j)
                        cand_d.append(d_map)
                    cand_w.append(pix_w)
                obj_stack = torch.stack(cand_obj, 0)
                a_stack = torch.stack(cand_a, 0)
                w_stack = torch.stack(cand_w, 0)
                agree_weight = max(float(args.anchor_iblend_agree_weight), 0.0)
                if agree_weight > 0 and obj_stack.shape[0] > 1:
                    sigma = max(float(args.anchor_iblend_agree_sigma), 1e-4)
                    consensus = obj_stack.median(dim=0).values
                    color_err = (obj_stack - consensus[None]).abs().mean(dim=-1, keepdim=True)
                    valid_color = a_stack > args.anchor_zselect_alpha_min
                    agree_gate = torch.exp(-agree_weight * color_err / sigma)
                    w_stack = w_stack * torch.where(valid_color, agree_gate, torch.ones_like(agree_gate))
                if depth_weight > 0 and cand_d:
                    d_stack = torch.stack(cand_d, 0)
                    valid_d = a_stack[..., 0] > args.anchor_zselect_alpha_min
                    front = torch.where(valid_d, d_stack, torch.full_like(d_stack, 1e10)).amin(dim=0)
                    tol = max(float(args.anchor_iblend_depth_tol_frac) * float(radius), 1e-6)
                    behind = (d_stack - front[None]).clamp_min(0.0)
                    depth_gate = torch.exp(-depth_weight * behind / tol)
                    w_stack = w_stack * torch.where(valid_d, depth_gate, torch.ones_like(depth_gate))[..., None]
                obj, a_i, r_i = _compose_iblend_anchor(
                    obj_stack, a_stack, w_stack, args.anchor_iblend_color_mode,
                    args.anchor_iblend_alpha_mode, bg
                )
                if args.anchor_render_mode == "iblend_tsurf_fill":
                    r_t, a_t, _ = _render_target_surface_for_view(view_i, r_i.dtype, r_i.device)
                    gate = a_i.clamp(0.0, 1.0).pow(max(float(args.anchor_fill_alpha_power), 1e-6))
                    r_i = r_i * gate + r_t * (1.0 - gate)
                    a_i = a_i * gate + a_t * (1.0 - gate)
                if static_fill is not None:
                    r_s, a_s = _render_params(
                        static_fill, w2c_all[view_i:view_i + 1],
                        K_all[view_i:view_i + 1], w, h, bg=bg,
                        sh_degree=sh_degree,
                    )
                    gate = a_i.clamp(0.0, 1.0).pow(max(float(args.anchor_fill_alpha_power), 1e-6))
                    if args.anchor_render_mode == "learned_iblend_fill":
                        d_feat = torch.stack(cand_d, 0) if cand_d else a_stack[..., 0].new_zeros(a_stack.shape[:-1])
                        support_feat = (
                            torch.stack(cand_support, 0)
                            if cand_support
                            else a_stack.new_ones(a_stack.shape)
                        )
                        target_rgb_i = target_frames[view_i] if target_frames is not None else None
                        target_alpha_i = target_fg[view_i] if target_fg is not None else None
                        r_i, a_i = _apply_learned_iblend_fill(
                            obj_stack, a_stack, w_stack, support_feat, d_feat,
                            weights[view_i, :selected.shape[1]],
                            r_s[0], a_s[0], gate,
                            radius, bg,
                            target_rgb=target_rgb_i,
                            target_alpha=target_alpha_i,
                        )
                        r_i, a_i = apply_output_hull(r_i, a_i, view_i)
                        r_i, a_i = apply_output_alpha_cleanup(r_i, a_i)
                        r_i, a_i = apply_output_alpha_refine(r_i, a_i)
                        renders.append(r_i[None])
                        alphas.append(a_i[None])
                        continue
                    r_blend, a_blend = _blend_static_fill(
                        r_i, a_i, r_s[0], a_s[0],
                        args.anchor_fill_alpha_power,
                        args.anchor_fill_static_alpha_min,
                        args.anchor_fill_static_alpha_softness,
                    )
                    fill_mask = None
                    if (args.anchor_fill_mask_alpha_min > 0
                            or args.anchor_fill_mask_dilate_px > 0):
                        fill_mask = (
                            a_i[..., 0] > float(args.anchor_fill_mask_alpha_min)
                        ).to(dtype=r_i.dtype)[None, None]
                        if args.anchor_fill_mask_dilate_px > 0:
                            pad = int(args.anchor_fill_mask_dilate_px)
                            fill_mask = Fnn.max_pool2d(
                                fill_mask, kernel_size=2 * pad + 1,
                                stride=1, padding=pad,
                            )
                        fill_mask = fill_mask[0, 0, ..., None].clamp(0.0, 1.0)
                    if args.anchor_fill_hull_mask:
                        if (source_fg is None or source_K is None or source_w2c is None
                                or source_c2w is None or radius is None):
                            raise ValueError("--anchor_fill_hull_mask requires source masks/cameras")
                        hull_mask = _target_visual_hull_masks(
                            source_fg, source_K, source_c2w, source_w2c,
                            K_all[view_i:view_i + 1],
                            c2w_all[view_i:view_i + 1],
                            w2c_all[view_i:view_i + 1],
                            float(radius), h, w,
                        )[0].to(device=r_i.device, dtype=r_i.dtype)
                        fill_mask = hull_mask if fill_mask is None else fill_mask * hull_mask
                    if args.anchor_fill_target_surface_mask:
                        surface_mask = _target_surface_gate(view_i, r_i.dtype, r_i.device)
                        fill_mask = surface_mask if fill_mask is None else fill_mask * surface_mask
                    if fill_mask is not None:
                        r_i = r_blend * fill_mask + r_i * (1.0 - fill_mask)
                        a_i = a_blend * fill_mask + a_i * (1.0 - fill_mask)
                    else:
                        r_i, a_i = r_blend, a_blend
                if args.anchor_render_mode in {"iblend_surf_gate", "iblend_surf_gate_fill"}:
                    gate_s = _target_surface_gate(view_i, r_i.dtype, r_i.device)
                    r_i = r_i * gate_s + r_i.new_full((), bg) * (1.0 - gate_s)
                    a_i = a_i * gate_s
            elif args.anchor_render_mode in {"zselect", "zselect_fill", "maxalpha", "maxalpha_fill"}:
                cand_r, cand_a, cand_d = [], [], []
                for j in range(selected.shape[1]):
                    p_j = parts[int(selected[view_i, j])]
                    r_j, a_j = _render_params(
                        p_j, w2c_all[view_i:view_i + 1],
                        K_all[view_i:view_i + 1], w, h, bg=bg,
                        sh_degree=sh_degree,
                    )
                    d_j = render_expected_depth(
                        p_j, w2c_all[view_i:view_i + 1],
                        K_all[view_i:view_i + 1], w, h, mode="ED"
                    )
                    cand_r.append(r_j[0])
                    cand_a.append(a_j[0])
                    cand_d.append(d_j[0, ..., 0])
                r_stack = torch.stack(cand_r, 0)
                a_stack = torch.stack(cand_a, 0)
                d_stack = torch.stack(cand_d, 0)
                valid = a_stack[..., 0] > args.anchor_zselect_alpha_min
                if args.anchor_render_mode in {"maxalpha", "maxalpha_fill"}:
                    alpha_score = torch.where(valid, a_stack[..., 0], torch.full_like(d_stack, -1.0))
                    pick = alpha_score.argmax(dim=0)
                else:
                    inf = torch.full_like(d_stack, 1e10)
                    d_score = torch.where(valid, d_stack, inf)
                    pick = d_score.argmin(dim=0)
                gather_rgb = pick[None, ..., None].expand(1, h, w, 3)
                gather_a = pick[None, ..., None].expand(1, h, w, 1)
                r_i = r_stack.gather(0, gather_rgb)[0]
                a_i = a_stack.gather(0, gather_a)[0]
                empty = ~valid.any(dim=0)
                if empty.any():
                    r_i = torch.where(empty[..., None], r_i.new_full((), bg), r_i)
                    a_i = torch.where(empty[..., None], a_i.new_zeros(()), a_i)
                if static_fill is not None:
                    r_s, a_s = _render_params(
                        static_fill, w2c_all[view_i:view_i + 1],
                        K_all[view_i:view_i + 1], w, h, bg=bg,
                        sh_degree=sh_degree,
                    )
                    r_i, a_i = _blend_static_fill(
                        r_i, a_i, r_s[0], a_s[0],
                        args.anchor_fill_alpha_power,
                        args.anchor_fill_static_alpha_min,
                        args.anchor_fill_static_alpha_softness,
                    )
            elif selected.shape[1] == 1:
                p_i = parts[int(selected[view_i, 0])]
                r_i, a_i = _render_params(
                    p_i, w2c_all[view_i:view_i + 1],
                    K_all[view_i:view_i + 1], w, h, bg=bg,
                    sh_degree=sh_degree,
                )
                r_i, a_i = r_i[0], a_i[0]
                if static_fill is not None:
                    r_s, a_s = _render_params(
                        static_fill, w2c_all[view_i:view_i + 1],
                        K_all[view_i:view_i + 1], w, h, bg=bg,
                        sh_degree=sh_degree,
                    )
                    r_blend, a_blend = _blend_static_fill(
                        r_i, a_i, r_s[0], a_s[0],
                        args.anchor_fill_alpha_power,
                        args.anchor_fill_static_alpha_min,
                        args.anchor_fill_static_alpha_softness,
                    )
                    if args.anchor_render_mode == "learned_fill":
                        sel_local = int(selected[view_i, 0])
                        r_i, a_i = _apply_learned_fill(
                            r_i, a_i, r_s[0], a_s[0], gate,
                            c2w_all[view_i, :3, 3],
                            anchor_c2w[sel_local, :3, 3],
                            dist[view_i, sel_local],
                            radius, bg,
                        )
                        r_i, a_i = apply_output_hull(r_i, a_i, view_i)
                        r_i, a_i = apply_output_alpha_cleanup(r_i, a_i)
                        r_i, a_i = apply_output_alpha_refine(r_i, a_i)
                        renders.append(r_i[None])
                        alphas.append(a_i[None])
                        continue
                    fill_mask = None
                    if (args.anchor_fill_mask_alpha_min > 0
                            or args.anchor_fill_mask_dilate_px > 0):
                        fill_mask = (
                            a_i[..., 0] > float(args.anchor_fill_mask_alpha_min)
                        ).to(dtype=r_i.dtype)[None, None]
                        if args.anchor_fill_mask_dilate_px > 0:
                            pad = int(args.anchor_fill_mask_dilate_px)
                            fill_mask = Fnn.max_pool2d(
                                fill_mask, kernel_size=2 * pad + 1,
                                stride=1, padding=pad,
                            )
                        fill_mask = fill_mask[0, 0, ..., None].clamp(0.0, 1.0)
                    if args.anchor_fill_hull_mask:
                        if (source_fg is None or source_K is None or source_w2c is None
                                or source_c2w is None or radius is None):
                            raise ValueError("--anchor_fill_hull_mask requires source masks/cameras")
                        hull_mask = _target_visual_hull_masks(
                            source_fg, source_K, source_c2w, source_w2c,
                            K_all[view_i:view_i + 1],
                            c2w_all[view_i:view_i + 1],
                            w2c_all[view_i:view_i + 1],
                            float(radius), h, w,
                        )[0].to(device=r_i.device, dtype=r_i.dtype)
                        fill_mask = hull_mask if fill_mask is None else fill_mask * hull_mask
                    if args.anchor_fill_target_surface_mask:
                        surface_mask = _target_surface_gate(view_i, r_i.dtype, r_i.device)
                        fill_mask = surface_mask if fill_mask is None else fill_mask * surface_mask
                    if fill_mask is not None:
                        r_i = r_blend * fill_mask + r_i * (1.0 - fill_mask)
                        a_i = a_blend * fill_mask + a_i * (1.0 - fill_mask)
                    else:
                        r_i, a_i = r_blend, a_blend
            else:
                weighted_parts = []
                for j in range(selected.shape[1]):
                    p_j = dict(parts[int(selected[view_i, j])])
                    p_j["opacity"] = p_j["opacity"] * weights[view_i, j].to(p_j["opacity"].dtype)
                    weighted_parts.append(p_j)
                p_i = _cat_params(weighted_parts)
                r_i, a_i = _render_params(
                    p_i, w2c_all[view_i:view_i + 1],
                    K_all[view_i:view_i + 1], w, h, bg=bg,
                    sh_degree=sh_degree,
                )
                r_i, a_i = r_i[0], a_i[0]
            r_i, a_i = apply_output_hull(r_i, a_i, view_i)
            r_i, a_i = apply_output_alpha_cleanup(r_i, a_i)
            r_i, a_i = apply_output_alpha_refine(r_i, a_i)
            renders.append(r_i[None])
            alphas.append(a_i[None])
        return torch.cat(renders, 0), torch.cat(alphas, 0)

    def train_loss(sample, step: int):
        latent = sample["latent"][None].to(dev)
        ref_K, ref_c2w = sample["ref_K"].to(dev), sample["ref_c2w"].to(dev)
        radius = float(sample["radius"]); w, h = sample["width"], sample["height"]
        w2c, K = sample["w2c"].to(dev), sample["K"].to(dev)
        c2w = sample["c2w_opengl"].to(dev)
        frames, fg = sample["frames"].to(dev), sample["masks"].to(dev)
        sample_depths = sample.get("depths")
        sample_depths = sample_depths.to(dev) if sample_depths is not None else None
        depth_refine_delta_terms.clear()
        depth_refine_tv_terms.clear()
        depth_refine_gt_terms.clear()
        depth_refine_metric_gt_terms.clear()
        support_gate_delta_terms.clear()
        support_gate_tv_terms.clear()
        support_gate_gt_terms.clear()
        surface_confidence_delta_terms.clear()
        surface_confidence_tv_terms.clear()
        surface_confidence_gt_terms.clear()
        surface_refine_delta_terms.clear()
        surface_refine_tv_terms.clear()
        surface_refine_rgb_gt_terms.clear()
        surface_refine_rgb_grad_gt_terms.clear()
        fusion_candidate_delta_terms.clear()
        fusion_candidate_gt_terms.clear()
        output_alpha_refine_delta_terms.clear()
        output_alpha_refine_tv_terms.clear()
        condition_rgb_refine_gt_terms.clear()
        condition_rgbd_refine_delta_terms.clear()
        condition_rgbd_refine_tv_terms.clear()
        condition_rgbd_refine_rgb_gt_terms.clear()
        condition_rgbd_refine_depth_gt_terms.clear()
        condition_pose_center_terms.clear()
        condition_pose_forward_terms.clear()
        condition_pose_dist_terms.clear()
        condition_depth_affine_delta_terms.clear()
        condition_depth_affine_gt_terms.clear()
        condition_depth_confidence_delta_terms.clear()
        condition_depth_confidence_tv_terms.clear()
        condition_depth_confidence_gt_terms.clear()
        cond = _condition_bundle(sample, frames, fg, w2c, K, c2w, sample_depths)
        with _train_precision_context():
            parts, anchor_ids, render_depths, support_depths = _predict_anchor_parts(
                latent, cond["K"], cond["c2w"], cond["frames"], cond["fg"], radius,
                w2c_all=cond["w2c"],
                depths=cond["depths"],
                visibility_depths=cond["visibility_depths"],
                target_depths=cond["target_depths"],
                target_frames=cond["target_frames"],
                confs=cond["confs"],
                spread_anchors=False,
            )
        parts = [_renderable_params(part) for part in parts]
        p = parts[0] if len(parts) == 1 else _cat_params(parts)
        n_anchor_gauss = model.map_h * model.map_w
        cur_bg = float(torch.rand(1))
        learned_fill_delta_terms.clear()
        learned_fill_tv_terms.clear()
        learned_fill_oracle_terms.clear()
        tgt = frames * fg + cur_bg * (1 - fg)
        render, alpha = _render_with_anchor_mode(
            parts, anchor_ids, w2c, K, c2w, w, h, bg=cur_bg,
            anchor_c2w_all=cond["c2w"],
            radius=radius,
            source_frames=cond["frames"],
            source_fg=cond["fg"],
            source_depths=render_depths,
            source_visibility_depths=support_depths,
            source_target_depths=cond["target_depths"],
            source_confs=cond["confs"],
            source_w2c=cond["w2c"],
            source_K=cond["K"],
            source_c2w=cond["c2w"],
            target_frames=tgt,
            target_fg=fg,
        )
        # L2/MSE loss option: aligns optimization with PSNR exactly
        # (PSNR ≡ −10·log10(MSE)).  L1 has a documented deletion bias at
        # silhouette edges that pushes anti-aliased alpha to 0 — even when
        # all other regularization is in place (v1, v1.1, v1.2 all failed).
        if getattr(args, "photometric_loss_type", "l1") == "l2":
            per_px = ((render - tgt) ** 2).mean(-1, keepdim=True)
        else:
            per_px = (render - tgt).abs().mean(-1, keepdim=True)
        wmap = 1.0 + (args.fg_weight - 1.0) * fg
        l1 = (per_px * wmap).sum() / (wmap.sum() + 1e-8)
        ssim = _ssim(render, tgt)
        mask = mask_alpha_l1(alpha, fg)
        scale_cap = args.scale_cap_frac * radius
        hinge = scale_hinge(p["scale"], s_min=0.005, s_max=scale_cap)
        if adaptive_loss is not None:
            loss = (
                _loss_weighted("photo", l1, 1.0)
                + _loss_weighted("ssim", 1 - ssim, 0.2)
                + _loss_weighted("mask", mask, args.mask_weight)
                + _loss_weighted("hinge", hinge, 0.01)
            )
        else:
            loss = l1 + 0.2 * (1 - ssim) + args.mask_weight * mask + 0.01 * hinge
        # Sparse-voxel identity regularization: keep vis ≈ 1 (preserve prior)
        # unless strong photometric signal pushes otherwise.  Counters the
        # documented sharpness-via-deletion failure mode.
        sv_identity = loss.new_zeros(())
        if (sparse_voxel_fusion_head is not None
                and args.sparse_voxel_identity_reg_weight > 0):
            sv_identity = (
                args.sparse_voxel_identity_reg_weight
                * sparse_voxel_fusion_head.vis_reg_loss()
            )
            loss = loss + sv_identity
        sv_support = loss.new_zeros(())
        if (sparse_voxel_fusion_head is not None
                and args.sparse_voxel_support_reg_weight > 0):
            sv_support = (
                args.sparse_voxel_support_reg_weight
                * sparse_voxel_fusion_head.support_reg_loss()
            )
            loss = loss + sv_support
        sv_target_vis = loss.new_zeros(())
        if (sparse_voxel_fusion_head is not None
                and args.sparse_voxel_target_vis_weight > 0):
            sv_target_vis = (
                args.sparse_voxel_target_vis_weight
                * sparse_voxel_fusion_head.target_vis_loss()
            )
            loss = loss + sv_target_vis
        canonical_target_vis = loss.new_zeros(())
        if (canonical_voxel_decoder is not None
                and args.canonical_target_vis_weight > 0
                and cond.get("target_depths") is not None):
            scored_p = _with_depth_consistency_score(
                p,
                cond["frames"],
                cond["fg"],
                support_depths if support_depths is not None else render_depths,
                cond["confs"],
                cond["w2c"],
                cond["K"],
                radius,
                target_depths_for_candidate=cond["target_depths"],
                force_target_counts=True,
            )
            support = scored_p.get("_fusion_target_support")
            conflict = scored_p.get("_fusion_target_conflict")
            if support is not None and conflict is not None:
                support = support.detach().to(device=p["opacity"].device, dtype=p["opacity"].dtype)
                conflict = conflict.detach().to(device=p["opacity"].device, dtype=p["opacity"].dtype)
                denom = support + conflict
                valid = denom > 0
                if valid.any():
                    target_soft = torch.where(
                        valid,
                        support / denom.clamp_min(1.0),
                        support.new_zeros(()),
                    )
                    pos = valid & (target_soft >= float(args.canonical_target_vis_pos_min))
                    neg = valid & (target_soft <= float(args.canonical_target_vis_neg_max))
                    valid = pos | neg
                    if valid.any():
                        target = torch.where(
                            pos,
                            target_soft.new_ones(()),
                            target_soft.new_zeros(()),
                        ).reshape(-1, 1)
                        pred = p["opacity"].clamp(1e-4, 1.0 - 1e-4)
                        loss_px = Fnn.binary_cross_entropy(pred, target, reduction="none")
                        weights = (
                            target * max(float(args.canonical_target_vis_positive_weight), 0.0)
                            + (1.0 - target)
                            * max(float(args.canonical_target_vis_negative_weight), 0.0)
                        )
                        valid_2d = valid.reshape(-1, 1)
                        canonical_target_vis = (
                            args.canonical_target_vis_weight
                            * (loss_px[valid_2d] * weights[valid_2d]).sum()
                            / weights[valid_2d].sum().clamp_min(1e-6)
                        )
                        loss = loss + canonical_target_vis
        canonical_source_vis_distill = loss.new_zeros(())
        if (canonical_voxel_decoder is not None
                and args.canonical_source_vis_distill_weight > 0):
            support = p.get("_fusion_support")
            conflict = p.get("_fusion_conflict")
            if support is not None and conflict is not None:
                support = support.detach().to(device=p["opacity"].device, dtype=p["opacity"].dtype)
                conflict = conflict.detach().to(device=p["opacity"].device, dtype=p["opacity"].dtype)
                score = support - float(args.canonical_source_vis_conflict_weight) * conflict
                target = torch.sigmoid(
                    (score - float(args.canonical_source_vis_min_support))
                    / max(float(args.canonical_source_vis_softness), 1e-6)
                )
                floor = min(max(float(args.canonical_source_vis_floor), 0.0), 1.0)
                target = (floor + (1.0 - floor) * target).clamp(1e-4, 1.0 - 1e-4)
                pred = p["opacity"].clamp(1e-4, 1.0 - 1e-4)
                coverage = p.get("_fusion_coverage")
                if coverage is not None:
                    coverage = coverage.detach().to(device=pred.device, dtype=pred.dtype)
                else:
                    coverage = torch.ones_like(pred)
                weights = (
                    1.0
                    + 0.25 * support.clamp(0.0, 4.0)
                    + 0.50 * conflict.clamp(0.0, 4.0)
                    + 0.10 * coverage.clamp(0.0, 8.0)
                )
                canonical_source_vis_distill = (
                    args.canonical_source_vis_distill_weight
                    * (Fnn.binary_cross_entropy(pred, target, reduction="none") * weights).sum()
                    / weights.sum().clamp_min(1e-6)
                )
                loss = loss + canonical_source_vis_distill
        fg_alpha = loss.new_zeros(())
        if args.fg_alpha_weight > 0 or _adapt_has("fg_alpha"):
            fg_area = fg.sum().clamp_min(1.0)
            fg_alpha_raw = (torch.relu(fg - alpha) * fg).sum() / fg_area
            fg_alpha = _loss_weighted("fg_alpha", fg_alpha_raw, args.fg_alpha_weight)
            loss = loss + fg_alpha
        op_det = p["opacity"].detach()
        scale_det = p["scale"].detach()
        scale_raw_det = p["scale_raw"].detach()
        sf = lambda x: float(x.detach())
        comp = {"l1": sf(l1), "ssim": sf(0.2 * (1 - ssim)),
                "mask": sf(args.mask_weight * mask), "hinge": sf(0.01 * hinge),
                "percep": 0.0, "percep_w": 0.0, "fg_color": 0.0,
                "anchor_rgb": 0.0, "anchor_opacity": 0.0, "anchor_scale": 0.0,
                "anchor_visibility": 0.0,
                "sv_identity": sf(sv_identity), "sv_support": sf(sv_support),
                "sv_target_vis": sf(sv_target_vis),
                "canonical_target_vis": sf(canonical_target_vis),
                "canonical_source_vis_distill": sf(canonical_source_vis_distill),
                "fg_alpha": sf(fg_alpha),
                "grad": 0.0, "grad_w": 0.0,
                "scaffold_detail": 0.0, "scaffold_detail_w": 0.0,
                "detail_teacher": 0.0, "detail_teacher_w": 0.0,
                "alpha_grad": 0.0, "alpha_grad_w": 0.0,
                "alpha_interior_smooth": 0.0,
                "alpha_interior_smooth_w": 0.0,
                "alpha_anti_lattice": 0.0,
                "alpha_anti_lattice_w": 0.0,
                "depth": 0.0, "depth_si": 0.0, "depth_abs": 0.0,
                "anchor_depth": 0.0, "depth_w": 0.0,
                "opreg": 0.0, "opent": 0.0, "bg_alpha": 0.0,
                "fill_prior": 0.0, "fill_tv": 0.0, "fill_delta": 0.0,
                "fill_oracle": 0.0,
                "depth_refine_delta": 0.0, "depth_refine_prior": 0.0,
                "depth_refine_tv": 0.0, "depth_refine_gt": 0.0,
                "depth_refine_metric_gt": 0.0,
                "support_gate_delta": 0.0, "support_gate_prior": 0.0,
                "support_gate_tv": 0.0, "support_gate_gt": 0.0,
                "surface_confidence_delta": 0.0, "surface_confidence_prior": 0.0,
                "surface_confidence_tv": 0.0, "surface_confidence_gt": 0.0,
                "surface_refine_delta": 0.0, "surface_refine_prior": 0.0,
                "surface_refine_tv": 0.0, "surface_refine_rgb_gt": 0.0,
                "surface_refine_rgb_grad_gt": 0.0,
                "surface_token_scale_reg": 0.0, "surface_token_mean_reg": 0.0,
                "surface_token_depth_normal_blend": 0.0,
                "surface_token_selected_view_mean": 0.0,
                "surface_token_selected_view_span": 0.0,
                "surface_token_selected_view_max": 0.0,
                "surface_token_candidate_view_count": 0.0,
                "surface_token_source_depth_abs_frac": 0.0,
                "surface_token_source_confidence_gate": 0.0,
                "surface_token_policy_depth_abs_frac": 0.0,
                "surface_token_policy_move_abs_frac": 0.0,
                "surface_token_policy_scale_mult": 0.0,
                "surface_token_policy_opacity_mult": 0.0,
                "surface_token_policy_view_gate": 0.0,
                "surface_token_policy_confidence_gate": 0.0,
                "surface_token_policy_keep_gate": 0.0,
                "surface_token_policy_coverage_mult": 0.0,
                "surface_token_policy_birth_gate": 0.0,
                "surface_token_projective_rgb": 0.0,
                "surface_token_projective_depth": 0.0,
                "surface_token_projective_opacity": 0.0,
                "surface_token_source_confidence": 0.0,
                "surface_token_source_depth": 0.0,
                "surface_token_source_support": 0.0,
                "surface_token_source_confidence_target": 0.0,
                "surface_token_source_depth_target_abs_frac": 0.0,
                "surface_token_proposal_cover": 0.0,
                "surface_token_proposal_surface": 0.0,
                "surface_token_proposal_opacity": 0.0,
                "surface_token_proposal_rgb": 0.0,
                "surface_token_proposal_detail": 0.0,
                "surface_token_proposal_support": 0.0,
                "surface_token_proposal_opacity_mean": 0.0,
                "surface_token_proposal_detail_mean": 0.0,
                "surface_token_proposal_coverage_mult": 0.0,
                "surface_token_proposal_anchor_mix": 0.0,
                "surface_token_proposal_anchor_entropy": 0.0,
                "surface_token_proposal_anchor_entropy_loss": 0.0,
                "surface_token_proposal_anchor_usage_loss": 0.0,
                "surface_token_proposal_anchor_usage_perplexity": 0.0,
                "surface_token_proposal_anchor_unique_frac": 0.0,
                "surface_token_proposal_anchor_collision_loss": 0.0,
                "surface_token_proposal_anchor_collision_frac": 0.0,
                "surface_token_proposal_anchor_even_prior": 0.0,
                "surface_token_proposal_policy_keep_gate": 0.0,
                "surface_token_proposal_policy_confidence_gate": 0.0,
                "surface_token_proposal_policy_coverage_mult": 0.0,
                "surface_token_proposal_policy_keep": 0.0,
                "surface_token_proposal_policy_confidence": 0.0,
                "surface_token_proposal_policy_coverage": 0.0,
                "surface_token_proposal_policy_keep_target": 0.0,
                "surface_token_proposal_policy_confidence_target": 0.0,
                "surface_token_proposal_policy_coverage_target": 0.0,
                "surface_token_scaffold_rgb": 0.0,
                "surface_token_scaffold_alpha": 0.0,
                "surface_token_scaffold_detail": 0.0,
                "canonical_scale_reg": 0.0, "canonical_scale_reg_w": 0.0,
                "fusion_candidate_delta": 0.0, "fusion_candidate_prior": 0.0,
                "fusion_candidate_gt": 0.0,
                "output_alpha_refine_delta": 0.0, "output_alpha_refine_prior": 0.0,
                "output_alpha_refine_tv": 0.0,
                "rgb_refine_gt": 0.0,
                "rgbd_refine_delta": 0.0, "rgbd_refine_prior": 0.0,
                "rgbd_refine_tv": 0.0, "rgbd_refine_rgb_gt": 0.0,
                "rgbd_refine_depth_gt": 0.0,
                "pose": 0.0, "pose_center": 0.0,
                "pose_forward": 0.0, "pose_dist": 0.0,
                "depth_affine_delta": 0.0, "depth_affine_prior": 0.0,
                "depth_affine_gt": 0.0,
                "depth_conf_delta": 0.0, "depth_conf_prior": 0.0,
                "depth_conf_tv": 0.0, "depth_conf_gt": 0.0,
                "resid_rgb": 0.0, "resid_geom": 0.0, "resid_depth": 0.0,
                "resid_opacity": 0.0, "resid_offset": 0.0,
                "resid_rgb_l2": 0.0, "resid_geom_l2": 0.0, "resid_depth_l2": 0.0,
                "resid_opacity_l2": 0.0, "resid_offset_l2": 0.0,
                "op_mean": float(op_det.mean()),
                "op_p99": float(op_det.reshape(-1).quantile(0.99)),
                "op_frac01": float((op_det > 0.1).float().mean()),
                "scale_mean": float(scale_det.mean()),
                "scale_frac98cap": float((scale_det > 0.98 * scale_cap).float().mean()),
                "scale_raw_over_cap": float((scale_raw_det > scale_cap).float().mean())}
        if adaptive_loss is not None:
            comp.update(adaptive_loss.values())
        st_scaffold_rgb_w = max(float(args.surface_token_scaffold_rgb_weight), 0.0)
        st_scaffold_alpha_w = max(float(args.surface_token_scaffold_alpha_weight), 0.0)
        st_scaffold_detail_w = max(float(args.surface_token_scaffold_detail_weight), 0.0)
        if (surface_token_decoder is not None
                and (st_scaffold_rgb_w > 0
                     or st_scaffold_alpha_w > 0
                     or st_scaffold_detail_w > 0
                     or _adapt_has("surface_token_scaffold_rgb")
                     or _adapt_has("surface_token_scaffold_alpha")
                     or _adapt_has("surface_token_scaffold_detail"))):
            was_training = surface_token_decoder.training
            try:
                surface_token_decoder.eval()
                with torch.no_grad():
                    with _train_precision_context():
                        teacher_parts, teacher_anchor_ids, teacher_render_depths, teacher_support_depths = (
                            _predict_anchor_parts(
                                latent, cond["K"], cond["c2w"], cond["frames"], cond["fg"], radius,
                                w2c_all=cond["w2c"],
                                depths=cond["depths"],
                                visibility_depths=cond["visibility_depths"],
                                target_depths=cond["target_depths"],
                                target_frames=cond["target_frames"],
                                confs=cond["confs"],
                                spread_anchors=False,
                                disable_surface_token_new_capacity=True,
                            )
                        )
                    teacher_parts = [_renderable_params(part) for part in teacher_parts]
                    teacher_render, teacher_alpha = _render_with_anchor_mode(
                        teacher_parts, teacher_anchor_ids, w2c, K, c2w, w, h, bg=cur_bg,
                        anchor_c2w_all=cond["c2w"],
                        radius=radius,
                        source_frames=cond["frames"],
                        source_fg=cond["fg"],
                        source_depths=teacher_render_depths,
                        source_visibility_depths=teacher_support_depths,
                        source_target_depths=cond["target_depths"],
                        source_confs=cond["confs"],
                        source_w2c=cond["w2c"],
                        source_K=cond["K"],
                        source_c2w=cond["c2w"],
                        target_frames=tgt,
                        target_fg=fg,
                    )
            finally:
                if was_training:
                    surface_token_decoder.train()
            margin = max(float(args.surface_token_scaffold_margin), 0.0)
            learned_err = (render.detach() - tgt).abs().mean(-1, keepdim=True)
            teacher_err = (teacher_render.detach() - tgt).abs().mean(-1, keepdim=True)
            worse = (learned_err > teacher_err + margin).to(dtype=render.dtype)
            if bool((worse > 0).any()):
                preserve_w = worse * (1.0 + (args.fg_weight - 1.0) * fg)
                preserve_den = preserve_w.sum().clamp_min(1.0)
                if st_scaffold_rgb_w > 0 or _adapt_has("surface_token_scaffold_rgb"):
                    scaffold_rgb_raw = (
                        (render - teacher_render.detach()).abs().mean(-1, keepdim=True)
                        * preserve_w
                    ).sum() / preserve_den
                    scaffold_rgb = _loss_weighted(
                        "surface_token_scaffold_rgb", scaffold_rgb_raw, st_scaffold_rgb_w
                    )
                    loss = loss + scaffold_rgb
                    comp["surface_token_scaffold_rgb"] = sf(scaffold_rgb)
                if st_scaffold_alpha_w > 0 or _adapt_has("surface_token_scaffold_alpha"):
                    scaffold_alpha_raw = (
                        (alpha - teacher_alpha.detach()).abs() * preserve_w
                    ).sum() / preserve_den
                    scaffold_alpha = _loss_weighted(
                        "surface_token_scaffold_alpha", scaffold_alpha_raw, st_scaffold_alpha_w
                    )
                    loss = loss + scaffold_alpha
                    comp["surface_token_scaffold_alpha"] = sf(scaffold_alpha)
            if st_scaffold_detail_w > 0 or _adapt_has("surface_token_scaffold_detail"):
                detail_fg = fg * (
                    teacher_alpha.detach()
                    > float(args.surface_token_scaffold_detail_alpha_min)
                ).to(dtype=fg.dtype)
                scaffold_detail_raw = _fg_gradient_loss(
                    render,
                    teacher_render.detach(),
                    detail_fg,
                )
                scaffold_detail = _loss_weighted(
                    "surface_token_scaffold_detail", scaffold_detail_raw, st_scaffold_detail_w
                )
                loss = loss + scaffold_detail
                comp["surface_token_scaffold_detail"] = sf(scaffold_detail)
        for name, key, weight in [
            ("resid_rgb", "_raw_residual_l2_rgb", args.residual_rgb_weight),
            ("resid_geom", "_raw_residual_l2_geom", args.residual_geom_weight),
            ("resid_depth", "_raw_residual_l2_depth", args.residual_depth_weight),
            ("resid_opacity", "_raw_residual_l2_opacity", args.residual_opacity_weight),
            ("resid_offset", "_raw_residual_l2_offset", args.residual_offset_weight),
        ]:
            term = _param_scalar_mean(p, key)
            if term is None:
                continue
            comp[f"{name}_l2"] = sf(term)
            if weight > 0:
                reg = weight * term
                loss = loss + reg
                comp[name] = sf(reg)
        if learned_fill_delta_terms:
            fill_delta = torch.stack(learned_fill_delta_terms).mean()
            comp["fill_delta"] = sf(fill_delta)
            if args.anchor_learned_fill_prior_weight > 0:
                fill_prior = args.anchor_learned_fill_prior_weight * fill_delta
                loss = loss + fill_prior
                comp["fill_prior"] = sf(fill_prior)
        if learned_fill_tv_terms and args.anchor_learned_fill_tv_weight > 0:
            fill_tv_raw = torch.stack(learned_fill_tv_terms).mean()
            fill_tv = args.anchor_learned_fill_tv_weight * fill_tv_raw
            loss = loss + fill_tv
            comp["fill_tv"] = sf(fill_tv)
        if learned_fill_oracle_terms and args.anchor_learned_fill_oracle_weight > 0:
            fill_oracle_raw = torch.stack(learned_fill_oracle_terms).mean()
            fill_oracle = args.anchor_learned_fill_oracle_weight * fill_oracle_raw
            loss = loss + fill_oracle
            comp["fill_oracle"] = sf(fill_oracle)
        if depth_refine_delta_terms:
            depth_refine_delta = torch.stack(depth_refine_delta_terms).mean()
            comp["depth_refine_delta"] = sf(depth_refine_delta)
            if args.depth_refine_prior_weight > 0:
                depth_refine_prior = args.depth_refine_prior_weight * depth_refine_delta
                loss = loss + depth_refine_prior
                comp["depth_refine_prior"] = sf(depth_refine_prior)
        if depth_refine_tv_terms and args.depth_refine_tv_weight > 0:
            depth_refine_tv_raw = torch.stack(depth_refine_tv_terms).mean()
            depth_refine_tv = args.depth_refine_tv_weight * depth_refine_tv_raw
            loss = loss + depth_refine_tv
            comp["depth_refine_tv"] = sf(depth_refine_tv)
        if depth_refine_gt_terms and args.depth_refine_gt_weight > 0:
            depth_refine_gt_raw = torch.stack(depth_refine_gt_terms).mean()
            depth_refine_gt = args.depth_refine_gt_weight * depth_refine_gt_raw
            loss = loss + depth_refine_gt
            comp["depth_refine_gt"] = sf(depth_refine_gt)
        if depth_refine_metric_gt_terms and args.depth_refine_metric_gt_weight > 0:
            depth_refine_metric_gt_raw = torch.stack(depth_refine_metric_gt_terms).mean()
            depth_refine_metric_gt = args.depth_refine_metric_gt_weight * depth_refine_metric_gt_raw
            loss = loss + depth_refine_metric_gt
            comp["depth_refine_metric_gt"] = sf(depth_refine_metric_gt)
        if support_gate_delta_terms:
            support_gate_delta = torch.stack(support_gate_delta_terms).mean()
            comp["support_gate_delta"] = sf(support_gate_delta)
            if args.support_gate_prior_weight > 0:
                support_gate_prior = args.support_gate_prior_weight * support_gate_delta
                loss = loss + support_gate_prior
                comp["support_gate_prior"] = sf(support_gate_prior)
        if support_gate_tv_terms and args.support_gate_tv_weight > 0:
            support_gate_tv_raw = torch.stack(support_gate_tv_terms).mean()
            support_gate_tv = args.support_gate_tv_weight * support_gate_tv_raw
            loss = loss + support_gate_tv
            comp["support_gate_tv"] = sf(support_gate_tv)
        if support_gate_gt_terms and args.support_gate_gt_weight > 0:
            support_gate_gt_raw = torch.stack(support_gate_gt_terms).mean()
            support_gate_gt = args.support_gate_gt_weight * support_gate_gt_raw
            loss = loss + support_gate_gt
            comp["support_gate_gt"] = sf(support_gate_gt)
        if surface_confidence_delta_terms:
            surface_conf_delta = torch.stack(surface_confidence_delta_terms).mean()
            comp["surface_confidence_delta"] = sf(surface_conf_delta)
            if args.surface_confidence_prior_weight > 0:
                surface_conf_prior = args.surface_confidence_prior_weight * surface_conf_delta
                loss = loss + surface_conf_prior
                comp["surface_confidence_prior"] = sf(surface_conf_prior)
        if surface_confidence_tv_terms and args.surface_confidence_tv_weight > 0:
            surface_conf_tv_raw = torch.stack(surface_confidence_tv_terms).mean()
            surface_conf_tv = args.surface_confidence_tv_weight * surface_conf_tv_raw
            loss = loss + surface_conf_tv
            comp["surface_confidence_tv"] = sf(surface_conf_tv)
        if surface_confidence_gt_terms and args.surface_confidence_gt_weight > 0:
            surface_conf_gt_raw = torch.stack(surface_confidence_gt_terms).mean()
            surface_conf_gt = args.surface_confidence_gt_weight * surface_conf_gt_raw
            loss = loss + surface_conf_gt
            comp["surface_confidence_gt"] = sf(surface_conf_gt)
        if surface_refine_delta_terms:
            surface_refine_delta = torch.stack(surface_refine_delta_terms).mean()
            comp["surface_refine_delta"] = sf(surface_refine_delta)
            if args.surface_refine_prior_weight > 0:
                surface_refine_prior = args.surface_refine_prior_weight * surface_refine_delta
                loss = loss + surface_refine_prior
                comp["surface_refine_prior"] = sf(surface_refine_prior)
        if surface_refine_tv_terms and args.surface_refine_tv_weight > 0:
            surface_refine_tv_raw = torch.stack(surface_refine_tv_terms).mean()
            surface_refine_tv = args.surface_refine_tv_weight * surface_refine_tv_raw
            loss = loss + surface_refine_tv
            comp["surface_refine_tv"] = sf(surface_refine_tv)
        if surface_refine_rgb_gt_terms and args.surface_refine_rgb_gt_weight > 0:
            surface_refine_rgb_gt_raw = torch.stack(surface_refine_rgb_gt_terms).mean()
            surface_refine_rgb_gt = args.surface_refine_rgb_gt_weight * surface_refine_rgb_gt_raw
            loss = loss + surface_refine_rgb_gt
            comp["surface_refine_rgb_gt"] = sf(surface_refine_rgb_gt)
        if (surface_refine_rgb_grad_gt_terms
                and args.surface_refine_rgb_grad_gt_weight > 0):
            surface_refine_rgb_grad_raw = torch.stack(surface_refine_rgb_grad_gt_terms).mean()
            surface_refine_rgb_grad = (
                args.surface_refine_rgb_grad_gt_weight * surface_refine_rgb_grad_raw
            )
            loss = loss + surface_refine_rgb_grad
            comp["surface_refine_rgb_grad_gt"] = sf(surface_refine_rgb_grad)
        if surface_token_decoder is not None:
            selected_view_ids = p.get("_surface_token_selected_view_ids")
            if selected_view_ids is not None and selected_view_ids.numel() > 0:
                selected_view_ids_f = selected_view_ids.detach().to(
                    device=p["scale"].device, dtype=p["scale"].dtype
                ).reshape(-1)
                comp["surface_token_selected_view_mean"] = sf(
                    selected_view_ids_f.mean()
                )
                comp["surface_token_selected_view_span"] = sf(
                    selected_view_ids_f.max() - selected_view_ids_f.min()
                )
                comp["surface_token_selected_view_max"] = sf(
                    selected_view_ids_f.max()
                )
            candidate_view_count = p.get("_surface_token_candidate_view_count")
            if candidate_view_count is not None:
                comp["surface_token_candidate_view_count"] = sf(
                    candidate_view_count.reshape(())
                )
            valid_st = p.get("_surface_token_valid")
            if valid_st is None:
                valid_st = p["scale"].new_ones((p["scale"].shape[0], 1))
            valid_st = valid_st.to(device=p["scale"].device, dtype=p["scale"].dtype).clamp(0.0, 1.0)
            valid_den = valid_st.sum().clamp_min(1.0)
            normal_blend_diag = p.get("_surface_token_depth_normal_blend")
            if normal_blend_diag is not None:
                comp["surface_token_depth_normal_blend"] = sf(
                    (normal_blend_diag.to(device=valid_st.device, dtype=valid_st.dtype)
                     * valid_st).sum() / valid_den
                )
            source_depth_diag = p.get("_surface_token_source_depth_res")
            if source_depth_diag is not None:
                source_depth_diag = source_depth_diag.to(
                    device=valid_st.device, dtype=valid_st.dtype
                )
                diag_valid = valid_st[:source_depth_diag.shape[0]]
                prop_diag = p.get("_surface_token_proposal")
                if prop_diag is not None:
                    diag_valid = diag_valid * (
                        prop_diag[:source_depth_diag.shape[0]].to(
                            device=valid_st.device, dtype=valid_st.dtype
                        ) <= 0.5
                    ).to(dtype=valid_st.dtype)
                detail_diag = p.get("_surface_token_detail")
                if detail_diag is not None:
                    diag_valid = diag_valid * (
                        detail_diag[:source_depth_diag.shape[0]].to(
                            device=valid_st.device, dtype=valid_st.dtype
                        ) <= 0.5
                    ).to(dtype=valid_st.dtype)
                diag_den = diag_valid.sum().clamp_min(1.0)
                radius_t_diag = p["scale"].new_tensor(max(float(radius), 1e-6))
                comp["surface_token_source_depth_abs_frac"] = sf(
                    (source_depth_diag.abs() * diag_valid).sum()
                    / (diag_den * radius_t_diag.clamp_min(1e-8))
                )
            source_conf_diag = p.get("_surface_token_source_confidence_gate")
            if source_conf_diag is not None:
                source_conf_diag = source_conf_diag.to(
                    device=valid_st.device, dtype=valid_st.dtype
                )
                diag_valid = valid_st[:source_conf_diag.shape[0]]
                prop_diag = p.get("_surface_token_proposal")
                if prop_diag is not None:
                    diag_valid = diag_valid * (
                        prop_diag[:source_conf_diag.shape[0]].to(
                            device=valid_st.device, dtype=valid_st.dtype
                        ) <= 0.5
                    ).to(dtype=valid_st.dtype)
                detail_diag = p.get("_surface_token_detail")
                if detail_diag is not None:
                    diag_valid = diag_valid * (
                        detail_diag[:source_conf_diag.shape[0]].to(
                            device=valid_st.device, dtype=valid_st.dtype
                        ) <= 0.5
                    ).to(dtype=valid_st.dtype)
                diag_den = diag_valid.sum().clamp_min(1.0)
                comp["surface_token_source_confidence_gate"] = sf(
                    (source_conf_diag * diag_valid).sum() / diag_den
                )
            policy_diag_names = {
                "surface_token_policy_depth_abs_frac": (
                    "_surface_token_policy_depth_res", "abs_frac"
                ),
                "surface_token_policy_move_abs_frac": (
                    "_surface_token_policy_move_res", "abs_frac"
                ),
                "surface_token_policy_scale_mult": (
                    "_surface_token_policy_scale_mult", "mean"
                ),
                "surface_token_policy_opacity_mult": (
                    "_surface_token_policy_opacity_mult", "mean"
                ),
                "surface_token_policy_view_gate": (
                    "_surface_token_policy_view_gate", "mean"
                ),
                "surface_token_policy_confidence_gate": (
                    "_surface_token_policy_confidence_gate", "mean"
                ),
                "surface_token_policy_keep_gate": (
                    "_surface_token_policy_keep_gate", "mean"
                ),
                "surface_token_policy_coverage_mult": (
                    "_surface_token_policy_coverage_mult", "mean"
                ),
                "surface_token_policy_birth_gate": (
                    "_surface_token_policy_birth_gate", "mean"
                ),
            }
            radius_t_diag = p["scale"].new_tensor(max(float(radius), 1e-6))
            for comp_key, (param_key, mode) in policy_diag_names.items():
                diag = p.get(param_key)
                if diag is None:
                    continue
                diag = diag.to(device=valid_st.device, dtype=valid_st.dtype)
                diag_valid = valid_st[:diag.shape[0]]
                diag_den = diag_valid.sum().clamp_min(1.0)
                if mode == "abs_frac":
                    diag_val = diag.abs()
                    if diag_val.shape[-1] > 1:
                        diag_val = diag_val.norm(dim=-1, keepdim=True)
                    diag_val = diag_val / radius_t_diag.clamp_min(1e-8)
                else:
                    diag_val = diag
                comp[comp_key] = sf((diag_val * diag_valid).sum() / (diag_den * diag_val.shape[-1]))
            if args.surface_token_scale_reg_weight > 0:
                tangent = max(float(args.surface_token_tangent_scale_max_frac), 1e-6) * radius
                normal = max(float(args.surface_token_normal_scale_max_frac), 1e-6) * radius
                target_scale = p["scale"].new_tensor([tangent, tangent, normal]).reshape(1, 3)
                over = torch.relu(p["scale"] - target_scale) / target_scale.clamp_min(1e-8)
                scale_raw = (over.square() * valid_st).sum() / (valid_den * p["scale"].shape[-1])
                scale_reg = args.surface_token_scale_reg_weight * scale_raw
                loss = loss + scale_reg
                comp["surface_token_scale_reg"] = sf(scale_reg)
            if args.surface_token_mean_reg_weight > 0 and "mean_offset" in p:
                radius_t = p["mean_offset"].new_tensor(max(float(radius), 1e-6))
                offset_raw = (
                    (p["mean_offset"] / radius_t).square() * valid_st
                ).sum() / (valid_den * p["mean_offset"].shape[-1])
                mean_reg = args.surface_token_mean_reg_weight * offset_raw
                loss = loss + mean_reg
                comp["surface_token_mean_reg"] = sf(mean_reg)
            proj_rgb_w = max(float(args.surface_token_projective_rgb_weight), 0.0)
            proj_depth_w = max(float(args.surface_token_projective_depth_weight), 0.0)
            proj_opacity_w = max(float(args.surface_token_projective_opacity_weight), 0.0)
            if ((proj_rgb_w > 0 or proj_depth_w > 0 or proj_opacity_w > 0
                 or _adapt_has("surface_token_projective_rgb")
                 or _adapt_has("surface_token_projective_depth")
                 or _adapt_has("surface_token_projective_opacity"))
                    and sample_depths is not None):
                valid_flat = valid_st.reshape(-1) > 0.5
                idx = valid_flat.nonzero(as_tuple=False).reshape(-1)
                max_points = int(args.surface_token_projective_max_points)
                if max_points > 0 and idx.numel() > max_points:
                    perm = torch.randperm(idx.numel(), device=idx.device)[:max_points]
                    idx = idx.index_select(0, perm)
                if idx.numel() > 0:
                    means_q = p["mean"].index_select(0, idx)
                    rgb_q = p["rgb"].index_select(0, idx)
                    op_q = p["opacity"].index_select(0, idx).reshape(-1).clamp(1e-4, 1.0 - 1e-4)
                    rgb_terms = []
                    depth_terms = []
                    opacity_terms = []
                    tol = max(float(args.surface_token_projective_depth_tol_frac), 1e-6) * float(radius)
                    fg_thr = min(max(float(args.surface_token_projective_fg_threshold), 0.0), 1.0)
                    radius_t = means_q.new_tensor(max(float(radius), 1e-6))
                    height, width = int(frames.shape[1]), int(frames.shape[2])
                    for view_i in range(frames.shape[0]):
                        cam = (
                            means_q @ w2c[view_i, :3, :3].T
                            + w2c[view_i, :3, 3]
                        )
                        z = cam[:, 2]
                        fx, fy = K[view_i, 0, 0], K[view_i, 1, 1]
                        cx, cy = K[view_i, 0, 2], K[view_i, 1, 2]
                        u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
                        v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
                        inb = (
                            (z > 1e-6)
                            & (u >= 0)
                            & (u <= width - 1)
                            & (v >= 0)
                            & (v <= height - 1)
                        )
                        if not bool(inb.any()):
                            continue
                        grid_x = (u / max(width - 1, 1)) * 2.0 - 1.0
                        grid_y = (v / max(height - 1, 1)) * 2.0 - 1.0
                        grid = torch.stack([grid_x, grid_y], -1).view(1, -1, 1, 2)
                        target_rgb = Fnn.grid_sample(
                            frames[view_i].permute(2, 0, 1)[None],
                            grid,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=True,
                        ).view(3, -1).T
                        target_mask = Fnn.grid_sample(
                            fg[view_i].permute(2, 0, 1)[None],
                            grid,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=True,
                        ).view(-1)
                        target_depth = Fnn.grid_sample(
                            sample_depths[view_i][None, None],
                            grid,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=True,
                        ).view(-1)
                        depth_valid = (
                            torch.isfinite(target_depth)
                            & (target_depth > 1e-6)
                            & (target_depth < 1e5)
                        )
                        fg_proj = inb & depth_valid & (target_mask > fg_thr)
                        if not bool(fg_proj.any()):
                            continue
                        depth_abs = (z - target_depth).abs()
                        depth_match = fg_proj & (depth_abs <= tol)
                        if proj_depth_w > 0 or _adapt_has("surface_token_projective_depth"):
                            depth_terms.append(
                                (depth_abs[fg_proj] / radius_t).clamp(max=1.0).mean()
                            )
                        if ((proj_rgb_w > 0 or _adapt_has("surface_token_projective_rgb"))
                                and bool(depth_match.any())):
                            rgb_terms.append(
                                (rgb_q[depth_match] - target_rgb[depth_match]).abs()
                                .mean(dim=-1)
                                .mean()
                            )
                        if ((proj_opacity_w > 0
                             or _adapt_has("surface_token_projective_opacity"))
                                and bool(depth_match.any())):
                            opacity_terms.append((-torch.log(op_q[depth_match])).mean())
                    if rgb_terms:
                        proj_rgb = _loss_weighted(
                            "surface_token_projective_rgb",
                            torch.stack(rgb_terms).mean(),
                            proj_rgb_w,
                        )
                        loss = loss + proj_rgb
                        comp["surface_token_projective_rgb"] = sf(proj_rgb)
                    if depth_terms:
                        proj_depth = _loss_weighted(
                            "surface_token_projective_depth",
                            torch.stack(depth_terms).mean(),
                            proj_depth_w,
                        )
                        loss = loss + proj_depth
                        comp["surface_token_projective_depth"] = sf(proj_depth)
                    if opacity_terms:
                        proj_opacity = _loss_weighted(
                            "surface_token_projective_opacity",
                            torch.stack(opacity_terms).mean(),
                            proj_opacity_w,
                        )
                        loss = loss + proj_opacity
                        comp["surface_token_projective_opacity"] = sf(proj_opacity)
            source_conf_w = max(float(args.surface_token_source_confidence_weight), 0.0)
            source_depth_w = max(float(args.surface_token_source_depth_weight), 0.0)
            if ((source_conf_w > 0
                 or source_depth_w > 0
                 or _adapt_has("surface_token_source_confidence")
                 or _adapt_has("surface_token_source_depth"))
                    and sample_depths is not None):
                source_losses = _surface_token_source_policy_losses(
                    p,
                    frames,
                    fg,
                    sample_depths,
                    K,
                    c2w,
                    radius=radius,
                    source_points=int(args.surface_token_source_policy_points),
                    target_points=int(args.surface_token_source_policy_target_points),
                    depth_tol_frac=float(args.surface_token_source_policy_depth_tol_frac),
                    fg_threshold=float(args.surface_token_source_policy_fg_threshold),
                    confidence_positive_weight=float(
                        args.surface_token_source_policy_positive_weight
                    ),
                    confidence_negative_weight=float(
                        args.surface_token_source_policy_negative_weight
                    ),
                    confidence_target_scale=(
                        float(args.surface_token_source_policy_confidence_target_scale)
                        if float(args.surface_token_source_policy_confidence_target_scale) >= 0
                        else float(args.surface_token_source_confidence_res_scale)
                    ),
                    support_mode=args.surface_token_source_policy_support_mode,
                    target_mode=args.surface_token_source_policy_target_mode,
                )
                if source_conf_w > 0 or _adapt_has("surface_token_source_confidence"):
                    source_conf = _loss_weighted(
                        "surface_token_source_confidence",
                        source_losses["confidence"],
                        source_conf_w,
                    )
                    loss = loss + source_conf
                    comp["surface_token_source_confidence"] = sf(source_conf)
                if source_depth_w > 0 or _adapt_has("surface_token_source_depth"):
                    source_depth = _loss_weighted(
                        "surface_token_source_depth",
                        source_losses["depth"],
                        source_depth_w,
                    )
                    loss = loss + source_depth
                    comp["surface_token_source_depth"] = sf(source_depth)
                comp["surface_token_source_support"] = sf(source_losses["support_mean"])
                comp["surface_token_source_confidence_target"] = sf(
                    source_losses["confidence_target_mean"]
                )
                comp["surface_token_source_depth_target_abs_frac"] = sf(
                    source_losses["depth_target_abs_frac"]
                )
            prop_cover_w = max(float(args.surface_token_proposal_cover_weight), 0.0)
            prop_surface_w = max(float(args.surface_token_proposal_surface_weight), 0.0)
            prop_opacity_w = max(float(args.surface_token_proposal_opacity_weight), 0.0)
            prop_rgb_w = max(float(args.surface_token_proposal_rgb_weight), 0.0)
            prop_detail_w = max(float(args.surface_token_proposal_detail_weight), 0.0)
            prop_anchor_entropy_w = max(float(args.surface_token_proposal_anchor_entropy_weight), 0.0)
            prop_anchor_usage_w = max(float(args.surface_token_proposal_anchor_usage_weight), 0.0)
            prop_anchor_collision_w = max(
                float(args.surface_token_proposal_anchor_collision_weight), 0.0
            )
            prop_policy_keep_w = max(float(args.surface_token_proposal_policy_keep_weight), 0.0)
            prop_policy_confidence_w = max(
                float(args.surface_token_proposal_policy_confidence_weight), 0.0
            )
            prop_policy_coverage_w = max(
                float(args.surface_token_proposal_policy_coverage_weight), 0.0
            )
            if ((prop_cover_w > 0
                 or prop_surface_w > 0
                 or prop_opacity_w > 0
                 or prop_rgb_w > 0
                 or prop_detail_w > 0
                 or prop_policy_keep_w > 0
                 or prop_policy_confidence_w > 0
                 or prop_policy_coverage_w > 0
                 or _adapt_has("surface_token_proposal_cover")
                 or _adapt_has("surface_token_proposal_surface")
                 or _adapt_has("surface_token_proposal_opacity")
                 or _adapt_has("surface_token_proposal_rgb")
                 or _adapt_has("surface_token_proposal_detail")
                 or _adapt_has("surface_token_proposal_policy_keep")
                 or _adapt_has("surface_token_proposal_policy_confidence")
                 or _adapt_has("surface_token_proposal_policy_coverage"))
                    and sample_depths is not None):
                prop_losses = _surface_token_proposal_losses(
                    p,
                    frames,
                    fg,
                    sample_depths,
                    K,
                    c2w,
                    radius=radius,
                    cover_points=int(args.surface_token_proposal_cover_points),
                    depth_tol_frac=float(args.surface_token_proposal_depth_tol_frac),
                    fg_threshold=float(args.surface_token_proposal_fg_threshold),
                    opacity_positive_weight=float(args.surface_token_proposal_opacity_positive_weight),
                    opacity_negative_weight=float(args.surface_token_proposal_opacity_negative_weight),
                    policy_target_mode=args.surface_token_proposal_policy_target_mode,
                    detail_edge_thresh=float(args.surface_token_proposal_detail_edge_thresh),
                )
                if prop_cover_w > 0 or _adapt_has("surface_token_proposal_cover"):
                    prop_cover = _loss_weighted(
                        "surface_token_proposal_cover",
                        prop_losses["cover"],
                        prop_cover_w,
                    )
                    loss = loss + prop_cover
                    comp["surface_token_proposal_cover"] = sf(prop_cover)
                if prop_surface_w > 0 or _adapt_has("surface_token_proposal_surface"):
                    prop_surface = _loss_weighted(
                        "surface_token_proposal_surface",
                        prop_losses["surface"],
                        prop_surface_w,
                    )
                    loss = loss + prop_surface
                    comp["surface_token_proposal_surface"] = sf(prop_surface)
                if prop_opacity_w > 0 or _adapt_has("surface_token_proposal_opacity"):
                    prop_opacity = _loss_weighted(
                        "surface_token_proposal_opacity",
                        prop_losses["opacity"],
                        prop_opacity_w,
                    )
                    loss = loss + prop_opacity
                    comp["surface_token_proposal_opacity"] = sf(prop_opacity)
                if prop_rgb_w > 0 or _adapt_has("surface_token_proposal_rgb"):
                    prop_rgb = _loss_weighted(
                        "surface_token_proposal_rgb",
                        prop_losses["rgb"],
                        prop_rgb_w,
                    )
                    loss = loss + prop_rgb
                    comp["surface_token_proposal_rgb"] = sf(prop_rgb)
                if prop_detail_w > 0 or _adapt_has("surface_token_proposal_detail"):
                    prop_detail = _loss_weighted(
                        "surface_token_proposal_detail",
                        prop_losses["detail_cover"],
                        prop_detail_w,
                    )
                    loss = loss + prop_detail
                    comp["surface_token_proposal_detail"] = sf(prop_detail)
                if prop_policy_keep_w > 0 or _adapt_has("surface_token_proposal_policy_keep"):
                    prop_policy_keep = _loss_weighted(
                        "surface_token_proposal_policy_keep",
                        prop_losses["policy_keep"],
                        prop_policy_keep_w,
                    )
                    loss = loss + prop_policy_keep
                    comp["surface_token_proposal_policy_keep"] = sf(prop_policy_keep)
                if (prop_policy_confidence_w > 0
                        or _adapt_has("surface_token_proposal_policy_confidence")):
                    prop_policy_confidence = _loss_weighted(
                        "surface_token_proposal_policy_confidence",
                        prop_losses["policy_confidence"],
                        prop_policy_confidence_w,
                    )
                    loss = loss + prop_policy_confidence
                    comp["surface_token_proposal_policy_confidence"] = sf(
                        prop_policy_confidence
                    )
                if (prop_policy_coverage_w > 0
                        or _adapt_has("surface_token_proposal_policy_coverage")):
                    prop_policy_coverage = _loss_weighted(
                        "surface_token_proposal_policy_coverage",
                        prop_losses["policy_coverage"],
                        prop_policy_coverage_w,
                    )
                    loss = loss + prop_policy_coverage
                    comp["surface_token_proposal_policy_coverage"] = sf(
                        prop_policy_coverage
                    )
                comp["surface_token_proposal_support"] = sf(prop_losses["support_mean"])
                comp["surface_token_proposal_opacity_mean"] = sf(prop_losses["opacity_mean"])
                comp["surface_token_proposal_detail_mean"] = sf(prop_losses["detail_mean"])
                comp["surface_token_proposal_policy_keep_target"] = sf(
                    prop_losses["policy_keep_target_mean"]
                )
                comp["surface_token_proposal_policy_confidence_target"] = sf(
                    prop_losses["policy_confidence_target_mean"]
                )
                comp["surface_token_proposal_policy_coverage_target"] = sf(
                    prop_losses["policy_coverage_target_mean"]
                )
            prop_mask_diag = p.get("_surface_token_proposal")
            if prop_mask_diag is not None:
                prop_mask_diag = prop_mask_diag.reshape(-1) > 0.5
                if bool(prop_mask_diag.any()):
                    mix_diag = p.get("_surface_token_proposal_anchor_mix")
                    ent_diag = p.get("_surface_token_proposal_anchor_entropy")
                    cov_diag = p.get("_surface_token_proposal_coverage_mult")
                    keep_diag = p.get("_surface_token_proposal_policy_keep_gate")
                    conf_diag = p.get("_surface_token_proposal_policy_confidence_gate")
                    covp_diag = p.get("_surface_token_proposal_policy_coverage_mult")
                    if mix_diag is not None:
                        comp["surface_token_proposal_anchor_mix"] = sf(
                            mix_diag.reshape(-1)[prop_mask_diag].float().mean()
                        )
                    if ent_diag is not None:
                        comp["surface_token_proposal_anchor_entropy"] = sf(
                            ent_diag.reshape(-1)[prop_mask_diag].float().mean()
                        )
                    if cov_diag is not None:
                        comp["surface_token_proposal_coverage_mult"] = sf(
                            cov_diag.reshape(-1)[prop_mask_diag].float().mean()
                        )
                    if keep_diag is not None:
                        comp["surface_token_proposal_policy_keep_gate"] = sf(
                            keep_diag.reshape(-1)[prop_mask_diag].float().mean()
                        )
                    if conf_diag is not None:
                        comp["surface_token_proposal_policy_confidence_gate"] = sf(
                            conf_diag.reshape(-1)[prop_mask_diag].float().mean()
                        )
                    if covp_diag is not None:
                        comp["surface_token_proposal_policy_coverage_mult"] = sf(
                            covp_diag.reshape(-1)[prop_mask_diag].float().mean()
                        )
                    ent_loss = _param_scalar_mean(
                        p, "_surface_token_proposal_anchor_entropy_loss"
                    )
                    usage_loss = _param_scalar_mean(
                        p, "_surface_token_proposal_anchor_usage_loss"
                    )
                    usage_ppl = _param_scalar_mean(
                        p, "_surface_token_proposal_anchor_usage_perplexity"
                    )
                    unique_frac = _param_scalar_mean(
                        p, "_surface_token_proposal_anchor_unique_frac"
                    )
                    collision_loss = _param_scalar_mean(
                        p, "_surface_token_proposal_anchor_collision_loss"
                    )
                    collision_frac = _param_scalar_mean(
                        p, "_surface_token_proposal_anchor_collision_frac"
                    )
                    if ent_loss is not None:
                        ent_term = _loss_weighted(
                            "surface_token_proposal_anchor_entropy",
                            ent_loss,
                            prop_anchor_entropy_w,
                        )
                        if prop_anchor_entropy_w > 0:
                            loss = loss + ent_term
                        comp["surface_token_proposal_anchor_entropy_loss"] = sf(ent_term)
                    if usage_loss is not None:
                        usage_term = _loss_weighted(
                            "surface_token_proposal_anchor_usage",
                            usage_loss,
                            prop_anchor_usage_w,
                        )
                        if prop_anchor_usage_w > 0:
                            loss = loss + usage_term
                        comp["surface_token_proposal_anchor_usage_loss"] = sf(usage_term)
                    if usage_ppl is not None:
                        comp["surface_token_proposal_anchor_usage_perplexity"] = sf(usage_ppl)
                    if unique_frac is not None:
                        comp["surface_token_proposal_anchor_unique_frac"] = sf(unique_frac)
                    if collision_loss is not None:
                        collision_term = _loss_weighted(
                            "surface_token_proposal_anchor_collision",
                            collision_loss,
                            prop_anchor_collision_w,
                        )
                        if prop_anchor_collision_w > 0:
                            loss = loss + collision_term
                        comp["surface_token_proposal_anchor_collision_loss"] = sf(
                            collision_term
                        )
                    if collision_frac is not None:
                        comp["surface_token_proposal_anchor_collision_frac"] = sf(
                            collision_frac
                        )
        if canonical_voxel_decoder is not None:
            valid_cv = p.get("_canonical_voxel_valid")
            if valid_cv is None:
                valid_cv = p["scale"].new_ones((p["scale"].shape[0], 1))
            valid_cv = valid_cv.to(device=p["scale"].device, dtype=p["scale"].dtype).clamp(0.0, 1.0)
            valid_den = valid_cv.sum().clamp_min(1.0)
            canonical_scale_w = _ramped_weight(
                args.canonical_scale_reg_weight,
                args.canonical_scale_reg_start,
                args.canonical_scale_reg_ramp,
                step,
            )
            comp["canonical_scale_reg_w"] = canonical_scale_w
            if canonical_scale_w > 0:
                tangent = max(float(args.canonical_tangent_scale_max_frac), 1e-8) * radius
                normal = max(float(args.canonical_normal_scale_max_frac), 1e-8) * radius
                target_scale = p["scale"].new_tensor([tangent, tangent, normal]).reshape(1, 3)
                over = torch.relu(p["scale"] - target_scale) / target_scale.clamp_min(1e-8)
                scale_raw = (over.square() * valid_cv).sum() / (valid_den * p["scale"].shape[-1])
                canonical_scale_reg = canonical_scale_w * scale_raw
                loss = loss + canonical_scale_reg
                comp["canonical_scale_reg"] = sf(canonical_scale_reg)
        if fusion_candidate_delta_terms:
            fusion_candidate_delta = torch.stack(fusion_candidate_delta_terms).mean()
            comp["fusion_candidate_delta"] = sf(fusion_candidate_delta)
            if args.fusion_candidate_prior_weight > 0:
                fusion_candidate_prior = args.fusion_candidate_prior_weight * fusion_candidate_delta
                loss = loss + fusion_candidate_prior
                comp["fusion_candidate_prior"] = sf(fusion_candidate_prior)
        if fusion_candidate_gt_terms and args.fusion_candidate_gt_weight > 0:
            fusion_candidate_gt_raw = torch.stack(fusion_candidate_gt_terms).mean()
            fusion_candidate_gt = args.fusion_candidate_gt_weight * fusion_candidate_gt_raw
            loss = loss + fusion_candidate_gt
            comp["fusion_candidate_gt"] = sf(fusion_candidate_gt)
        if output_alpha_refine_delta_terms:
            output_delta = torch.stack(output_alpha_refine_delta_terms).mean()
            comp["output_alpha_refine_delta"] = sf(output_delta)
            if args.output_alpha_refine_prior_weight > 0:
                output_prior = args.output_alpha_refine_prior_weight * output_delta
                loss = loss + output_prior
                comp["output_alpha_refine_prior"] = sf(output_prior)
        if output_alpha_refine_tv_terms and args.output_alpha_refine_tv_weight > 0:
            output_tv_raw = torch.stack(output_alpha_refine_tv_terms).mean()
            output_tv = args.output_alpha_refine_tv_weight * output_tv_raw
            loss = loss + output_tv
            comp["output_alpha_refine_tv"] = sf(output_tv)
        if condition_rgb_refine_gt_terms and args.condition_rgb_refine_gt_weight > 0:
            rgb_refine_gt_raw = torch.stack(condition_rgb_refine_gt_terms).mean()
            rgb_refine_gt = args.condition_rgb_refine_gt_weight * rgb_refine_gt_raw
            loss = loss + rgb_refine_gt
            comp["rgb_refine_gt"] = sf(rgb_refine_gt)
        if condition_rgbd_refine_delta_terms:
            rgbd_delta = torch.stack(condition_rgbd_refine_delta_terms).mean()
            comp["rgbd_refine_delta"] = sf(rgbd_delta)
            if args.condition_rgbd_refine_prior_weight > 0:
                rgbd_prior = args.condition_rgbd_refine_prior_weight * rgbd_delta
                loss = loss + rgbd_prior
                comp["rgbd_refine_prior"] = sf(rgbd_prior)
        if condition_rgbd_refine_tv_terms and args.condition_rgbd_refine_tv_weight > 0:
            rgbd_tv_raw = torch.stack(condition_rgbd_refine_tv_terms).mean()
            rgbd_tv = args.condition_rgbd_refine_tv_weight * rgbd_tv_raw
            loss = loss + rgbd_tv
            comp["rgbd_refine_tv"] = sf(rgbd_tv)
        if (condition_rgbd_refine_rgb_gt_terms
                and args.condition_rgbd_refine_rgb_gt_weight > 0):
            rgbd_rgb_gt_raw = torch.stack(condition_rgbd_refine_rgb_gt_terms).mean()
            rgbd_rgb_gt = args.condition_rgbd_refine_rgb_gt_weight * rgbd_rgb_gt_raw
            loss = loss + rgbd_rgb_gt
            comp["rgbd_refine_rgb_gt"] = sf(rgbd_rgb_gt)
        if (condition_rgbd_refine_depth_gt_terms
                and args.condition_rgbd_refine_depth_gt_weight > 0):
            rgbd_depth_gt_raw = torch.stack(condition_rgbd_refine_depth_gt_terms).mean()
            rgbd_depth_gt = args.condition_rgbd_refine_depth_gt_weight * rgbd_depth_gt_raw
            loss = loss + rgbd_depth_gt
            comp["rgbd_refine_depth_gt"] = sf(rgbd_depth_gt)
        if (condition_pose_center_terms
                and condition_pose_forward_terms
                and condition_pose_dist_terms
                and args.condition_pose_weight > 0):
            pose_center = torch.stack(condition_pose_center_terms).mean()
            pose_forward = torch.stack(condition_pose_forward_terms).mean()
            pose_dist = torch.stack(condition_pose_dist_terms).mean()
            pose_raw = (
                args.condition_pose_center_weight * pose_center
                + args.condition_pose_forward_weight * pose_forward
                + args.condition_pose_dist_weight * pose_dist
            )
            pose_loss = args.condition_pose_weight * pose_raw
            loss = loss + pose_loss
            comp["pose"] = sf(pose_loss)
            comp["pose_center"] = sf(pose_center)
            comp["pose_forward"] = sf(pose_forward)
            comp["pose_dist"] = sf(pose_dist)
        if condition_depth_affine_delta_terms:
            depth_affine_delta = torch.stack(condition_depth_affine_delta_terms).mean()
            comp["depth_affine_delta"] = sf(depth_affine_delta)
            if args.condition_depth_affine_prior_weight > 0:
                depth_affine_prior = args.condition_depth_affine_prior_weight * depth_affine_delta
                loss = loss + depth_affine_prior
                comp["depth_affine_prior"] = sf(depth_affine_prior)
        if (condition_depth_affine_gt_terms
                and args.condition_depth_affine_gt_weight > 0):
            depth_affine_gt_raw = torch.stack(condition_depth_affine_gt_terms).mean()
            depth_affine_gt = args.condition_depth_affine_gt_weight * depth_affine_gt_raw
            loss = loss + depth_affine_gt
            comp["depth_affine_gt"] = sf(depth_affine_gt)
        if condition_depth_confidence_delta_terms:
            depth_conf_delta = torch.stack(condition_depth_confidence_delta_terms).mean()
            comp["depth_conf_delta"] = sf(depth_conf_delta)
            if args.condition_depth_confidence_prior_weight > 0:
                depth_conf_prior = args.condition_depth_confidence_prior_weight * depth_conf_delta
                loss = loss + depth_conf_prior
                comp["depth_conf_prior"] = sf(depth_conf_prior)
        if (condition_depth_confidence_tv_terms
                and args.condition_depth_confidence_tv_weight > 0):
            depth_conf_tv_raw = torch.stack(condition_depth_confidence_tv_terms).mean()
            depth_conf_tv = args.condition_depth_confidence_tv_weight * depth_conf_tv_raw
            loss = loss + depth_conf_tv
            comp["depth_conf_tv"] = sf(depth_conf_tv)
        if (condition_depth_confidence_gt_terms
                and args.condition_depth_confidence_gt_weight > 0):
            depth_conf_gt_raw = torch.stack(condition_depth_confidence_gt_terms).mean()
            depth_conf_gt = args.condition_depth_confidence_gt_weight * depth_conf_gt_raw
            loss = loss + depth_conf_gt
            comp["depth_conf_gt"] = sf(depth_conf_gt)
        if percep is not None:
            pw = _ramped_weight(args.perceptual_weight, args.perceptual_start,
                                args.perceptual_ramp, step)
            comp["percep_w"] = pw
            if pw > 0:
                pl = pw * percep(render, tgt)
                loss = loss + pl; comp["percep"] = sf(pl)
        grad_w = _ramped_weight(args.grad_weight, args.grad_start, args.grad_ramp, step)
        comp["grad_w"] = grad_w
        if grad_w > 0 or _adapt_has("grad"):
            gl_raw = _fg_gradient_loss(render, tgt, fg)
            gl = _loss_weighted("grad", gl_raw, grad_w)
            loss = loss + gl
            comp["grad"] = sf(gl)
        scaffold_w = _ramped_weight(
            args.scaffold_detail_weight,
            args.scaffold_detail_start,
            args.scaffold_detail_ramp,
            step,
        )
        comp["scaffold_detail_w"] = scaffold_w
        if scaffold_w > 0 and sparse_voxel_fusion_head is not None:
            with torch.no_grad():
                scaffold_render, scaffold_alpha = _render_with_anchor_mode(
                    parts, anchor_ids, w2c, K, c2w, w, h, bg=cur_bg,
                    anchor_c2w_all=cond["c2w"],
                    radius=radius,
                    source_frames=cond["frames"],
                    source_fg=cond["fg"],
                    source_depths=render_depths,
                    source_visibility_depths=support_depths,
                    source_target_depths=cond["target_depths"],
                    source_confs=cond["confs"],
                    source_w2c=cond["w2c"],
                    source_K=cond["K"],
                    source_c2w=cond["c2w"],
                    target_frames=tgt,
                    target_fg=fg,
                    apply_sparse_voxel_refine=False,
                )
            detail_fg = fg * (
                scaffold_alpha.detach() > float(args.scaffold_detail_alpha_min)
            ).to(dtype=fg.dtype)
            sdl = scaffold_w * _fg_gradient_loss(
                render,
                scaffold_render.detach(),
                detail_fg,
            )
            loss = loss + sdl
            comp["scaffold_detail"] = sf(sdl)
        detail_teacher_w = _ramped_weight(
            args.detail_teacher_weight,
            args.detail_teacher_start,
            args.detail_teacher_ramp,
            step,
        )
        comp["detail_teacher_w"] = detail_teacher_w
        if detail_teacher_w > 0 or _adapt_has("detail_teacher"):
            dtl_raw = _detail_teacher_loss(
                render,
                tgt,
                alpha,
                fg,
                edge_thresh=args.detail_teacher_edge_thresh,
                alpha_min=args.detail_teacher_alpha_min,
                artifact_weight=args.detail_teacher_artifact_weight,
            )
            dtl = _loss_weighted("detail_teacher", dtl_raw, detail_teacher_w)
            loss = loss + dtl
            comp["detail_teacher"] = sf(dtl)
        alpha_grad_w = _ramped_weight(
            args.alpha_grad_weight,
            args.alpha_grad_start,
            args.alpha_grad_ramp,
            step,
        )
        comp["alpha_grad_w"] = alpha_grad_w
        if alpha_grad_w > 0 or _adapt_has("alpha_grad"):
            agl_raw = _alpha_gradient_loss(
                alpha,
                fg,
                band_px=args.alpha_grad_band_px,
            )
            agl = _loss_weighted("alpha_grad", agl_raw, alpha_grad_w)
            loss = loss + agl
            comp["alpha_grad"] = sf(agl)
        alpha_smooth_w = _ramped_weight(
            args.alpha_interior_smooth_weight,
            args.alpha_interior_smooth_start,
            args.alpha_interior_smooth_ramp,
            step,
        )
        comp["alpha_interior_smooth_w"] = alpha_smooth_w
        if alpha_smooth_w > 0 or _adapt_has("alpha_interior_smooth"):
            ais_raw = _alpha_interior_smooth_loss(
                alpha,
                fg,
                edge_band_px=args.alpha_interior_smooth_edge_band_px,
            )
            ais = _loss_weighted("alpha_interior_smooth", ais_raw, alpha_smooth_w)
            loss = loss + ais
            comp["alpha_interior_smooth"] = sf(ais)
        alpha_lattice_w = _ramped_weight(
            args.alpha_anti_lattice_weight,
            args.alpha_anti_lattice_start,
            args.alpha_anti_lattice_ramp,
            step,
        )
        comp["alpha_anti_lattice_w"] = alpha_lattice_w
        if alpha_lattice_w > 0 or _adapt_has("alpha_anti_lattice"):
            aal_raw = _alpha_anti_lattice_loss(
                alpha,
                tgt,
                fg,
                blur_px=args.alpha_anti_lattice_blur_px,
                edge_band_px=args.alpha_anti_lattice_edge_band_px,
                detail_edge_thresh=args.alpha_anti_lattice_detail_edge_thresh,
            )
            aal = _loss_weighted("alpha_anti_lattice", aal_raw, alpha_lattice_w)
            loss = loss + aal
            comp["alpha_anti_lattice"] = sf(aal)
        if args.fg_color_weight > 0 or _adapt_has("fg_color"):
            # Compare accumulated object color independent of alpha/background.
            # Detach alpha in the denominator so this term improves color instead
            # of learning to game opacity.
            alpha_det = alpha.detach()
            obj_rgb = (render - (1.0 - alpha) * cur_bg) / alpha_det.clamp_min(args.fg_color_alpha_min)
            valid = fg.bool() & (alpha_det > args.fg_color_alpha_min)
            if valid.any():
                fgc = Fnn.l1_loss(obj_rgb[valid.expand_as(obj_rgb)], frames[valid.expand_as(frames)])
                fgc_loss = _loss_weighted("fg_color", fgc, args.fg_color_weight)
                loss = loss + fgc_loss
                comp["fg_color"] = sf(fgc_loss)
        if args.bg_alpha_weight > 0 or _adapt_has("bg_alpha"):
            bg = (1.0 - fg).clamp(0.0, 1.0)
            bal = (alpha * bg).sum() / bg.sum().clamp_min(1.0)
            bal_loss = _loss_weighted("bg_alpha", bal, args.bg_alpha_weight)
            loss = loss + bal_loss
            comp["bg_alpha"] = sf(bal_loss)
        if (args.anchor_rgb_weight > 0 or args.anchor_opacity_weight > 0
                or args.anchor_scale_weight > 0 or args.anchor_visibility_weight > 0):
            ar_sum = p["rgb"].new_zeros(())
            ao_sum = p["rgb"].new_zeros(())
            asc_sum = p["rgb"].new_zeros(())
            av_sum = p["rgb"].new_zeros(())
            ar_n = ao_n = asc_n = av_n = 0
            for local_i, view_i in enumerate(anchor_ids):
                start = local_i * n_anchor_gauss
                stop = start + n_anchor_gauss
                anchor_rgb_source = (
                    cond.get("target_frames", cond["frames"])
                    if args.anchor_rgb_target == "gt" else cond["frames"]
                )
                ref_rgb = anchor_rgb_source[view_i].permute(2, 0, 1)[None]  # (1,3,H,W)
                ref_fg = cond["fg"][view_i].permute(2, 0, 1)[None]       # (1,1,H,W)
                den = Fnn.interpolate(ref_fg, size=(model.map_h, model.map_w), mode="area")
                fg_grid = den[0, 0].reshape(-1).clamp(0.0, 1.0)
                valid_rgb = fg_grid > 0.01
                if args.anchor_rgb_weight > 0 and valid_rgb.any():
                    num = Fnn.interpolate(ref_rgb * ref_fg, size=(model.map_h, model.map_w), mode="area")
                    rgb_tgt = (num / den.clamp_min(1e-6))[0].permute(1, 2, 0).reshape(-1, 3)
                    ar_sum = ar_sum + Fnn.l1_loss(p["rgb"][start:stop][valid_rgb], rgb_tgt[valid_rgb])
                    ar_n += 1
                if args.anchor_opacity_weight > 0:
                    # Rendered alpha can match the mask by stacking many faint splats.
                    # This direct target forces ref-silhouette splats themselves
                    # out of the near-transparent pseudo-equilibrium.
                    op = p["opacity"][start:stop].reshape(-1).clamp(1e-6, 1.0 - 1e-6)
                    op_w = 1.0 + (args.fg_weight - 1.0) * fg_grid
                    ao = Fnn.binary_cross_entropy(op, fg_grid, weight=op_w, reduction="sum")
                    ao_sum = ao_sum + ao / op_w.sum().clamp_min(1e-6)
                    ao_n += 1
                if args.anchor_scale_weight > 0 and valid_rgb.any():
                    st = args.anchor_scale_frac * radius
                    p_scale_i = p["scale"][start:stop]
                    asc_sum = asc_sum + Fnn.huber_loss(
                        p_scale_i[valid_rgb], torch.full_like(p_scale_i[valid_rgb], st),
                        delta=max(st * 0.25, 1e-4)
                    )
                    asc_n += 1
                if args.anchor_visibility_weight > 0:
                    vis = _visibility_condition(
                        view_i, cond["fg"], cond["depths"], cond["K"], cond["c2w"], cond["w2c"], radius
                    )[None]
                    vis_grid = Fnn.interpolate(vis, size=(model.map_h, model.map_w), mode="area")
                    target = (den * vis_grid).reshape(-1).clamp(0.0, 1.0)
                    op = p["opacity"][start:stop].reshape(-1).clamp(1e-6, 1.0 - 1e-6)
                    op_w = 1.0 + (args.fg_weight - 1.0) * target
                    av = Fnn.binary_cross_entropy(op, target, weight=op_w, reduction="sum")
                    av_sum = av_sum + av / op_w.sum().clamp_min(1e-6)
                    av_n += 1
            if ar_n:
                ar = args.anchor_rgb_weight * (ar_sum / ar_n)
                loss = loss + ar
                comp["anchor_rgb"] = sf(ar)
            if ao_n:
                ao = args.anchor_opacity_weight * (ao_sum / ao_n)
                loss = loss + ao
                comp["anchor_opacity"] = sf(ao)
            if asc_n:
                asc = args.anchor_scale_weight * (asc_sum / asc_n)
                loss = loss + asc
                comp["anchor_scale"] = sf(asc)
            if av_n:
                av = args.anchor_visibility_weight * (av_sum / av_n)
                loss = loss + av
                comp["anchor_visibility"] = sf(av)
        depth_w = _depth_weight(args.depth_weight, step)
        anchor_depth_w = _depth_weight(args.anchor_depth_weight, step)
        comp["depth_w"] = depth_w
        if depth_w > 0 or _adapt_has("depth"):
            dw, dh = w, h
            K_depth = K[0:1]
            gt_d = sample["ref_depth"].to(dev)
            valid = sample["ref_depth_valid"].to(dev)
            if args.depth_render_scale != 1.0:
                if not (0.0 < args.depth_render_scale <= 1.0):
                    raise ValueError("--depth_render_scale must be in (0, 1]")
                dw = max(1, int(round(w * args.depth_render_scale)))
                dh = max(1, int(round(h * args.depth_render_scale)))
                sx, sy = dw / w, dh / h
                K_depth = K_depth.clone()
                K_depth[:, 0, 0] *= sx
                K_depth[:, 0, 2] *= sx
                K_depth[:, 1, 1] *= sy
                K_depth[:, 1, 2] *= sy
                valid_f = valid.float()
                num = Fnn.interpolate((gt_d * valid_f)[None, None], size=(dh, dw),
                                      mode="area")[0, 0]
                den = Fnn.interpolate(valid_f[None, None], size=(dh, dw), mode="area")[0, 0]
                gt_d = num / den.clamp_min(1e-6)
                valid = den > 0.01
            ed = render_expected_depth(
                p, w2c[0:1], K_depth, dw, dh, mode=args.depth_render_mode
            )[0, ..., 0]   # (H,W)
            d_si = scale_invariant_depth_loss(ed, gt_d, valid)
            d_abs = absolute_depth_loss(ed, gt_d, valid)
            dl_raw = d_si + args.depth_abs_weight * d_abs
            dl = _loss_weighted("depth", dl_raw, depth_w)
            loss = loss + dl
            comp["depth"] = sf(dl)
            comp["depth_si"] = sf(d_si)
            comp["depth_abs"] = sf(d_abs)
        if anchor_depth_w > 0:
            depths = (
                cond["target_depths"]
                if args.anchor_depth_use_target and cond["target_depths"] is not None
                else cond["depths"]
            )
            ad = p["depth"].new_zeros(())
            n_ad = 0
            for local_i, view_i in enumerate(anchor_ids):
                if depths is not None:
                    z = depths[view_i]
                elif view_i == 0:
                    z = sample["ref_depth"].to(dev)
                else:
                    continue
                target_t, valid_t = depth_target_on_grid(
                    z, cond["fg"][view_i, ..., 0] > 0.5,
                    cond["K"][view_i], model.map_h, model.map_w
                )
                p_depth_i = p["depth"][local_i * n_anchor_gauss:(local_i + 1) * n_anchor_gauss]
                ad = ad + absolute_depth_loss(p_depth_i.squeeze(-1), target_t, valid_t)
                n_ad += 1
            ad = ad / max(n_ad, 1)
            adl = anchor_depth_w * ad
            loss = loss + adl
            comp["anchor_depth"] = sf(adl)
        if args.opacity_reg > 0 or args.scale_reg > 0:
            fg_g = None
            if args.opacity_reg_masked:
                # ref-view mask (view 0) resized to the Gaussian grid -> per-Gaussian FG weight
                m = fg[0].permute(2, 0, 1)[None]                                  # (1,1,H,W)
                fg_g = Fnn.interpolate(m, size=(model.map_h, model.map_w),
                                       mode="area").reshape(-1)                    # (N,)
                if fg_g.numel() != p["opacity"].reshape(-1).numel():
                    # Direct learned decoders (canonical/surface-token) emit a
                    # variable number of Gaussians, not the fixed decoder map.
                    # In that case a ref-view grid mask is not aligned with
                    # the Gaussian list, so use the unmasked regularizer.
                    fg_g = None
            op_term, sc_term = opacity_scale_reg(p["opacity"], p["scale"],
                                                 fg=fg_g, masked=bool(args.opacity_reg_masked))
            reg = args.opacity_reg * op_term + args.scale_reg * sc_term
            loss = loss + reg; comp["opreg"] = sf(reg)
        if args.opacity_entropy > 0:
            ent = args.opacity_entropy * opacity_entropy(p["opacity"])
            loss = loss + ent; comp["opent"] = sf(ent)
        return loss, comp

    @torch.no_grad()
    def eval_objs(samples, chunk=8):
        model.eval()
        if blend_head is not None:
            blend_head.eval()
        if depth_refine_head is not None:
            depth_refine_head.eval()
        if support_gate_head is not None:
            support_gate_head.eval()
        if surface_confidence_head is not None:
            surface_confidence_head.eval()
        if surface_refine_head is not None:
            surface_refine_head.eval()
        if fusion_candidate_head is not None:
            fusion_candidate_head.eval()
        if condition_mask_refine_head is not None:
            condition_mask_refine_head.eval()
        if condition_rgb_refine_head is not None:
            condition_rgb_refine_head.eval()
        if condition_rgbd_refine_head is not None:
            condition_rgbd_refine_head.eval()
        if condition_pose_head is not None:
            condition_pose_head.eval()
        if condition_depth_affine_head is not None:
            condition_depth_affine_head.eval()
        if condition_depth_confidence_head is not None:
            condition_depth_confidence_head.eval()
        if output_alpha_refine_head is not None:
            output_alpha_refine_head.eval()
        if surface_token_decoder is not None:
            surface_token_decoder.eval()
        if surface_token_view_selector is not None:
            surface_token_view_selector.eval()
        if canonical_voxel_decoder is not None:
            canonical_voxel_decoder.eval()
        ps, ss, ops, rows = [], [], [], []
        for s in samples:
            latent = s["latent"][None].to(dev)
            ref_K, ref_c2w = s["K"][0].to(dev), s["c2w_opengl"][0].to(dev)
            radius = float(s["radius"]); w, h = s["width"], s["height"]
            w2c, K = s["w2c"].to(dev), s["K"].to(dev)
            c2w = s["c2w_opengl"].to(dev)
            target = s["frames"].to(dev); fg = s["masks"].to(dev).bool()
            eval_depths = s.get("depth")
            eval_depths = eval_depths.to(dev) if eval_depths is not None else None
            cond = _condition_bundle(s, target, fg.float(), w2c, K, c2w, eval_depths)
            with _train_precision_context():
                parts, anchor_ids, render_depths, support_depths = _predict_anchor_parts(
                    latent, cond["K"], cond["c2w"], cond["frames"], cond["fg"], radius,
                    w2c_all=cond["w2c"],
                    depths=cond["depths"],
                    visibility_depths=cond["visibility_depths"],
                    confs=cond["confs"],
                    spread_anchors=True,
                )
            parts = [_renderable_params(part) for part in parts]
            p = parts[0] if len(parts) == 1 else _cat_params(parts)
            op = p["opacity"].reshape(-1)               # watch for collapse + de-fog
            op_stats = (
                float(op.mean()),
                float(op.quantile(0.99)),
                float((op > 0.1).float().mean()),
            )
            ops.append(op_stats)
            if args.anchor_render_mode == "concat" and args.fusion_voxel_size_frac <= 0:
                outs = []
                alpha_outs = []
                for i in range(0, w2c.shape[0], chunk):
                    r_i, a_i = _render_params(
                        p, w2c[i:i + chunk], K[i:i + chunk], w, h, bg=1.0,
                        sh_degree=args.fusion_sh_degree if args.fusion_sh_degree > 0 else None,
                    )
                    outs.append(r_i)
                    alpha_outs.append(a_i)
                r_all = torch.cat(outs, 0)
                a_all = torch.cat(alpha_outs, 0)
            else:
                outs = []
                alpha_outs = []
                for i in range(0, w2c.shape[0], chunk):
                    r_i, a_i = _render_with_anchor_mode(
                        parts, anchor_ids, w2c[i:i + chunk], K[i:i + chunk],
                        c2w[i:i + chunk], w, h, bg=1.0,
                        anchor_c2w_all=cond["c2w"],
                        radius=radius,
                        source_frames=cond["frames"],
                        source_fg=cond["fg"],
                        source_depths=render_depths,
                        source_visibility_depths=support_depths,
                        source_confs=cond["confs"],
                        source_w2c=cond["w2c"],
                        source_K=cond["K"],
                        source_c2w=cond["c2w"],
                    )
                    outs.append(r_i)
                    alpha_outs.append(a_i)
                r_all = torch.cat(outs, 0)
                a_all = torch.cat(alpha_outs, 0)
            ps_i = fg_masked_psnr(r_all, target, fg)
            ss_i = sharpness_ratio(r_all, target, fg.squeeze(-1))
            alpha_stats = _alpha_mask_stats(a_all, fg)
            ps.append(ps_i)
            ss.append(ss_i)
            rows.append({
                "uid": s["uid"],
                "views": int(w2c.shape[0]),
                "fg_psnr": float(ps_i),
                "sharpness": float(ss_i),
                "opacity_mean": op_stats[0],
                "opacity_p99": op_stats[1],
                "opacity_frac_gt_0_1": op_stats[2],
                **alpha_stats,
                **cond.get("pose_metrics", {}),
            })
        if args.freeze_decoder:
            model.eval()
        else:
            model.train()
        if blend_head is not None:
            blend_head.train()
        if depth_refine_head is not None and not args.freeze_depth_refine_head:
            depth_refine_head.train()
        if support_gate_head is not None:
            support_gate_head.train()
        if surface_confidence_head is not None:
            surface_confidence_head.train()
        if surface_refine_head is not None:
            surface_refine_head.train()
        if fusion_candidate_head is not None:
            fusion_candidate_head.train()
        if condition_mask_refine_head is not None:
            condition_mask_refine_head.train()
        if condition_rgb_refine_head is not None:
            condition_rgb_refine_head.train()
        if condition_rgbd_refine_head is not None:
            condition_rgbd_refine_head.train()
        if condition_pose_head is not None and not args.freeze_condition_pose_head:
            condition_pose_head.train()
        if condition_depth_affine_head is not None:
            condition_depth_affine_head.train()
        if condition_depth_confidence_head is not None:
            condition_depth_confidence_head.train()
        if output_alpha_refine_head is not None:
            output_alpha_refine_head.train()
        if surface_token_decoder is not None:
            surface_token_decoder.train()
        if surface_token_view_selector is not None:
            surface_token_view_selector.train()
        if canonical_voxel_decoder is not None:
            canonical_voxel_decoder.train()
        return ps, ss, ops, rows

    def _parse_float_list_csv(value: str) -> list[float]:
        if not value:
            return []
        out_vals = []
        for item in value.split(","):
            item = item.strip()
            if item:
                out_vals.append(float(item))
        return out_vals

    def _orbit_c2w_opengl(azimuth_deg: float, elevation_deg: float,
                          cam_radius: torch.Tensor) -> torch.Tensor:
        """OpenGL c2w for a camera on a sphere looking at the world origin."""
        dtype, device = cam_radius.dtype, cam_radius.device
        az = math.radians(float(azimuth_deg))
        el = math.radians(float(elevation_deg))
        eye = torch.stack([
            cam_radius * math.cos(el) * math.cos(az),
            cam_radius * math.sin(el),
            cam_radius * math.cos(el) * math.sin(az),
        ])
        fwd = -eye / eye.norm().clamp_min(1e-8)
        up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        if torch.abs((fwd * up).sum()) > 0.98:
            up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        right = torch.cross(fwd, up, dim=0)
        right = right / right.norm().clamp_min(1e-8)
        true_up = torch.cross(right, fwd, dim=0)
        c2w = torch.eye(4, dtype=dtype, device=device)
        c2w[:3, 0] = right
        c2w[:3, 1] = true_up
        c2w[:3, 2] = -fwd
        c2w[:3, 3] = eye
        return c2w

    def _novel_eval_camera_grid(K_all: torch.Tensor, c2w_all: torch.Tensor,
                                n_views: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        elevs = _parse_float_list_csv(args.save_eval_viz_novel_elevations)
        if not elevs:
            raise ValueError("novel camera grid requested without elevations")
        n_views = max(int(n_views), 1)
        az_offsets = _parse_float_list_csv(args.save_eval_viz_novel_azimuths)
        if not az_offsets:
            n_az = max(int(math.ceil(n_views / max(len(elevs), 1))), 1)
            az_offsets = [360.0 * az_i / n_az for az_i in range(n_az)]
        eye0 = c2w_all[0, :3, 3]
        cam_radius = eye0.norm().clamp_min(1e-6) * float(args.save_eval_viz_novel_radius_scale)
        base_az = math.degrees(math.atan2(float(eye0[2]), float(eye0[0])))
        base_az += float(args.save_eval_viz_novel_azimuth_offset)
        c2ws, labels = [], []
        for elev in elevs:
            for az_offset in az_offsets:
                if len(c2ws) >= n_views:
                    break
                az = base_az + az_offset
                c2ws.append(_orbit_c2w_opengl(az, elev, cam_radius))
                labels.append(
                    f"theta{az % 360.0:06.1f}_phi{elev:+05.1f}"
                    .replace("+", "p")
                    .replace("-", "m")
                )
        c2w = torch.stack(c2ws, 0)
        w2c = opengl_c2w_to_opencv_w2c(c2w)
        K = K_all[:1].expand(c2w.shape[0], -1, -1).clone()
        return K, c2w, w2c, labels

    @torch.no_grad()
    def log_viz(step):
        if not run:
            return
        import wandb
        imgs = []
        for s in (heldout[:2] + train_eval[:2]):
            latent = s["latent"][None].to(dev)
            frames = s["frames"].to(dev)
            fg = s["masks"].to(dev)
            eval_depths = s.get("depth")
            eval_depths = eval_depths.to(dev) if eval_depths is not None else None
            K_all = s["K"].to(dev)
            c2w_all = s["c2w_opengl"].to(dev)
            w2c_all = s["w2c"].to(dev)
            cond = _condition_bundle(s, frames, fg, w2c_all, K_all, c2w_all, eval_depths)
            with _train_precision_context():
                parts, anchor_ids, render_depths, support_depths = _predict_anchor_parts(
                    latent, cond["K"], cond["c2w"],
                    cond["frames"], cond["fg"], float(s["radius"]),
                    w2c_all=cond["w2c"],
                    depths=cond["depths"],
                    visibility_depths=cond["visibility_depths"],
                    confs=cond["confs"],
                    spread_anchors=True,
                )
            parts = [_renderable_params(part) for part in parts]
            r, a = _render_with_anchor_mode(
                parts, anchor_ids, w2c_all[:1], K_all[:1], c2w_all[:1],
                s["width"], s["height"], bg=1.0, anchor_c2w_all=cond["c2w"],
                radius=float(s["radius"]),
                source_frames=cond["frames"],
                source_fg=cond["fg"],
                source_depths=render_depths,
                source_visibility_depths=support_depths,
                source_confs=cond["confs"],
                source_w2c=cond["w2c"],
                source_K=cond["K"],
                source_c2w=cond["c2w"],
            )
            a3 = a[0].expand(-1, -1, 3).clamp(0, 1).cpu()
            grid = torch.cat([s["frames"][0], r[0].cpu(), a3], dim=1).clamp(0, 1)
            imgs.append(wandb.Image((grid.numpy() * 255).astype("uint8"), caption=s["uid"][:8]))
        run.log({"viz": imgs}, step=step)

    @torch.no_grad()
    def save_eval_viz(step):
        if not args.save_eval_viz:
            return
        from PIL import Image, ImageDraw
        viz_dir = out / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_splits = (
            ("heldout", heldout[:max(int(args.save_eval_viz_heldout_count), 0)]),
            ("train", train_eval[:max(int(args.save_eval_viz_train_count), 0)]),
        )
        for split, samples in viz_splits:
            for s in samples:
                latent = s["latent"][None].to(dev)
                frames = s["frames"].to(dev)
                fg = s["masks"].to(dev)
                eval_depths = s.get("depth")
                eval_depths = eval_depths.to(dev) if eval_depths is not None else None
                K_all = s["K"].to(dev)
                c2w_all = s["c2w_opengl"].to(dev)
                w2c_all = s["w2c"].to(dev)
                cond = _condition_bundle(s, frames, fg, w2c_all, K_all, c2w_all, eval_depths)
                with _train_precision_context():
                    parts, anchor_ids, render_depths, support_depths = _predict_anchor_parts(
                        latent, cond["K"], cond["c2w"],
                        cond["frames"], cond["fg"], float(s["radius"]),
                        w2c_all=cond["w2c"],
                        depths=cond["depths"],
                        visibility_depths=cond["visibility_depths"],
                        confs=cond["confs"],
                        spread_anchors=True,
                    )
                parts = [_renderable_params(part) for part in parts]
                novel_elevs = _parse_float_list_csv(args.save_eval_viz_novel_elevations)
                if novel_elevs:
                    n_viz = max(args.save_eval_viz_views, 1)
                    K_viz, c2w_viz, w2c_viz, view_labels = _novel_eval_camera_grid(
                        K_all, c2w_all, n_viz
                    )
                    n_viz = K_viz.shape[0]
                else:
                    n_viz = min(max(args.save_eval_viz_views, 1), frames.shape[0])
                    K_viz, c2w_viz, w2c_viz = K_all[:n_viz], c2w_all[:n_viz], w2c_all[:n_viz]
                    view_indices = s.get("view_indices", torch.arange(frames.shape[0]))
                    view_labels = [f"v{int(view_indices[i]):03d}" for i in range(n_viz)]
                for local_v in range(n_viz):
                    r, a = _render_with_anchor_mode(
                        parts, anchor_ids, w2c_viz[local_v:local_v + 1],
                        K_viz[local_v:local_v + 1], c2w_viz[local_v:local_v + 1],
                        s["width"], s["height"], bg=1.0, anchor_c2w_all=cond["c2w"],
                        radius=float(s["radius"]),
                        source_frames=cond["frames"],
                        source_fg=cond["fg"],
                        source_depths=render_depths,
                        source_visibility_depths=support_depths,
                        source_confs=cond["confs"],
                        source_w2c=cond["w2c"],
                        source_K=cond["K"],
                        source_c2w=cond["c2w"],
                    )
                    a3 = a[0].expand(-1, -1, 3).clamp(0, 1).cpu()
                    if novel_elevs:
                        left = torch.ones_like(r[0].cpu())
                    else:
                        left = s["frames"][local_v]
                    grid = torch.cat([left, r[0].cpu(), a3], dim=1).clamp(0, 1)
                    img = Image.fromarray((grid.numpy() * 255).astype("uint8"))
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([0, 0, img.width, 28], fill=(255, 255, 255))
                    draw.text((10, 8), "novel camera" if novel_elevs else "GT", fill=(0, 0, 0))
                    draw.text((s["width"] + 10, 8), "render", fill=(0, 0, 0))
                    draw.text((2 * s["width"] + 10, 8), "alpha", fill=(0, 0, 0))
                    draw.text((10, img.height - 22), view_labels[local_v], fill=(0, 0, 0))
                    img.save(viz_dir / f"step_{step:06d}_{split}_{s['uid'][:8]}_{view_labels[local_v]}.png")

    def write_eval_rows(step: int, split: str, rows: list[dict]) -> None:
        if not args.save_eval_jsonl:
            return
        path = out / "eval_metrics.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"step": step, "split": split, **row}, sort_keys=True) + "\n")

    def save_checkpoint(step: int, name: str = "phase2.pt") -> None:
        if not args.save_checkpoints:
            return
        ckpt = {
            "model": model.state_dict(),
            "blend_head": blend_head.state_dict() if blend_head is not None else None,
            "depth_refine_head": (
                depth_refine_head.state_dict() if depth_refine_head is not None else None
            ),
            "support_gate_head": (
                support_gate_head.state_dict() if support_gate_head is not None else None
            ),
            "surface_confidence_head": (
                surface_confidence_head.state_dict()
                if surface_confidence_head is not None else None
            ),
            "surface_refine_head": (
                surface_refine_head.state_dict()
                if surface_refine_head is not None else None
            ),
            "fusion_candidate_head": (
                fusion_candidate_head.state_dict()
                if fusion_candidate_head is not None else None
            ),
            "condition_mask_refine_head": (
                condition_mask_refine_head.state_dict()
                if condition_mask_refine_head is not None else None
            ),
            "condition_rgb_refine_head": (
                condition_rgb_refine_head.state_dict()
                if condition_rgb_refine_head is not None else None
            ),
            "condition_rgbd_refine_head": (
                condition_rgbd_refine_head.state_dict()
                if condition_rgbd_refine_head is not None else None
            ),
            "condition_pose_head": (
                condition_pose_head.state_dict()
                if condition_pose_head is not None else None
            ),
            "condition_depth_affine_head": (
                condition_depth_affine_head.state_dict()
                if condition_depth_affine_head is not None else None
            ),
            "condition_depth_confidence_head": (
                condition_depth_confidence_head.state_dict()
                if condition_depth_confidence_head is not None else None
            ),
            "output_alpha_refine_head": (
                output_alpha_refine_head.state_dict()
                if output_alpha_refine_head is not None else None
            ),
            "sparse_voxel_fusion_head": (
                sparse_voxel_fusion_head.state_dict()
                if sparse_voxel_fusion_head is not None else None
            ),
            "surface_token_decoder": (
                surface_token_decoder.state_dict()
                if surface_token_decoder is not None else None
            ),
            "surface_token_view_selector": (
                surface_token_view_selector.state_dict()
                if surface_token_view_selector is not None else None
            ),
            "canonical_voxel_decoder": (
                canonical_voxel_decoder.state_dict()
                if canonical_voxel_decoder is not None else None
            ),
            "adaptive_loss": (
                adaptive_loss.state_dict()
                if adaptive_loss is not None else None
            ),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "step": step,
            "args": vars(args),
        }
        torch.save(ckpt, out / name)

    def _mean_row(rows: list[dict], key: str) -> float:
        if not rows:
            return 0.0
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)

    print(f"[phase2] objects={len(train_ds)} n_params={n_params:,} "
          f"latent={args.latent_t}x{args.latent_h}x{args.latent_w} "
          f"map={model.map_h}x{model.map_w} ups={args.ups_stages} "
          f"trainable={n_trainable:,} blend={blend_params:,} "
          f"drefine={depth_refine_params:,} sgate={support_gate_params:,} "
          f"surfconf={surface_confidence_params:,} "
          f"surfref={surface_refine_params:,} "
          f"fcand={fusion_candidate_params:,} "
          f"dconf={depth_confidence_params:,} "
          f"maskrefine={mask_refine_params:,} "
          f"rgbrefine={rgb_refine_params:,} "
          f"rgbdrefine={rgbd_refine_params:,} "
          f"pose={pose_params:,} "
          f"depthaffine={depth_affine_params:,} "
          f"outalpharefine={output_alpha_refine_params:,} "
          f"surftok={surface_token_params:,} "
          f"stviewsel={surface_token_view_selector_params:,} "
          f"canvox={canonical_voxel_params:,} "
          f"adaptloss={adaptive_loss_params:,} "
          f"precision={args.train_precision} "
          f"freeze={args.freeze_decoder} "
          f"plrm={args.surface_token_proposal_lr_mult:g} "
          f"pplrm={args.surface_token_proposal_policy_lr_mult:g} "
          f"polrm={args.surface_token_policy_lr_mult:g} "
          f"poplrm={args.surface_token_proposal_opacity_lr_mult:g} "
          f"dnblrm={args.surface_token_depth_normal_blend_lr_mult:g} "
          f"cap={args.scale_cap_frac} up={args.upsample_mode} skip={args.latent_skip} "
          f"anchors={args.anchor_views}/{args.anchor_render_mode}/fillp{args.anchor_fill_alpha_power:g}"
          f"/ibap{args.anchor_iblend_alpha_power:g}/ibvw{args.anchor_iblend_view_weight}"
          f"/ibdw{args.anchor_iblend_depth_weight:g}"
          f"/ibdt{args.anchor_iblend_depth_tol_frac:g}"
          f"/ibcolor{args.anchor_iblend_color_mode}"
          f"/ibalpha{args.anchor_iblend_alpha_mode}"
          f"/ibsw{args.anchor_iblend_support_weight:g}"
          f"/ibsr{args.anchor_iblend_support_refs}"
          f"/ibsd{args.anchor_iblend_support_decay:g}"
          f"/ibsf{args.anchor_iblend_support_floor:g}"
          f"/ibst{args.anchor_iblend_support_tol_frac:g}"
          f"/srpx{args.support_sample_radius_px}"
          f"/ibaw{args.anchor_iblend_agree_weight:g}"
          f"/ibas{args.anchor_iblend_agree_sigma:g}"
          f"/m{args.anchor_fill_mask_alpha_min:g}/d{args.anchor_fill_mask_dilate_px}"
          f"/samin{args.anchor_fill_static_alpha_min:g}"
          f"/sasoft{args.anchor_fill_static_alpha_softness:g}"
          f"/hull{args.anchor_fill_hull_mask}/tsmask{args.anchor_fill_target_surface_mask}"
          f"/outhull{args.anchor_output_hull_mask} "
          f"outclean={args.output_alpha_cleanup_min:g}"
          f"/soft{args.output_alpha_cleanup_softness:g}"
          f"/e{args.output_alpha_cleanup_erode_px}"
          f"/d{args.output_alpha_cleanup_dilate_px} "
          f"outaref={args.output_alpha_refine_unet}"
          f"/h{args.output_alpha_refine_hidden}"
          f"/init{args.output_alpha_refine_init:g}"
          f"/floor{args.output_alpha_refine_floor:g}"
          f"/scale{args.output_alpha_refine_delta_scale:g}"
          f"/prior{args.output_alpha_refine_prior_weight:g}"
          f"/tv{args.output_alpha_refine_tv_weight:g} "
          f"svfusion={args.use_sparse_voxel_fusion}/{args.use_mlp_voxel_fusion}"
          f"/{args.use_message_voxel_fusion}"
          f"/h{args.sparse_voxel_hidden}"
          f"/layers{args.mlp_voxel_layers}"
          f"/nbr{args.mlp_voxel_neighbor_radius}"
          f"/msg{args.mlp_voxel_message_radius}"
          f"/dres{args.sparse_voxel_depth_res_frac:g}"
          f"/rgbres{args.sparse_voxel_rgb_res_scale:g}"
          f"/opres{args.sparse_voxel_opacity_res_scale:g}"
          f"/vis{args.sparse_voxel_vis_delta:g}"
          f"/id{args.sparse_voxel_identity_reg_weight:g}"
          f"/sup{args.sparse_voxel_support_reg_weight:g}"
          f"/tvis{args.sparse_voxel_target_vis_weight:g}"
          f"/fga{args.fg_alpha_weight:g} "
          f"targetsurf={args.target_surface_depth_tol_frac:g}"
          f"/s{args.target_surface_scale_frac:g}"
          f"/z{args.target_surface_normal_scale_frac:g}"
          f"/op{args.target_surface_opacity:g}"
          f"/vwtemp{args.target_surface_view_weight_temp_frac:g}"
          f"/sup{args.target_surface_min_support}"
          f"/suptol{args.target_surface_support_tol_frac:g}"
          f"/gatep{args.target_surface_gate_power:g}"
          f"/gated{args.target_surface_gate_dilate_px} "
          f"learnfill={args.anchor_learned_fill_arch}:"
          f"{args.anchor_learned_fill_hidden}x{args.anchor_learned_fill_layers}"
          f"/rgbres{args.anchor_learned_fill_rgb_residual_scale:g}"
          f"/detach{args.anchor_learned_fill_detach_inputs}"
          f"/dscale{args.anchor_learned_fill_delta_scale:g}"
          f"/cdscale{args.anchor_learned_fill_candidate_delta_scale:g}"
          f"/prior{args.anchor_learned_fill_prior_weight:g}"
          f"/tv{args.anchor_learned_fill_tv_weight:g}"
          f"/oracle{args.anchor_learned_fill_oracle_weight:g}"
          f"/ot{args.anchor_learned_fill_oracle_temp:g} "
          f"drefine={args.depth_refine_unet}/h{args.depth_refine_hidden}"
          f"/scale{args.depth_refine_delta_scale:g}"
          f"/detach{args.depth_refine_detach_inputs}"
          f"/gt{args.depth_refine_gt_weight:g}"
          f"/mgt{args.depth_refine_metric_gt_weight:g}"
          f"/mgt_delta{args.depth_refine_metric_delta_frac:g}"
          f"/freeze{args.freeze_depth_refine_head}"
          f"/confw{args.depth_refine_conflict_weight:g}"
          f"/gtout{args.depth_refine_gt_outlier_weight:g}"
          f"/gtpow{args.depth_refine_gt_outlier_power:g}"
          f"/prior{args.depth_refine_prior_weight:g}"
          f"/tv{args.depth_refine_tv_weight:g}"
          f"/mv{args.depth_refine_multiview_features}"
          f"/refs{args.depth_refine_multiview_refs}"
          f"/chunk{args.depth_refine_chunk_views}"
          f"/ckpt{args.depth_refine_checkpoint}"
          f"/mvapply{args.depth_refine_apply_mv_conflict_min:g}"
          f"/{args.depth_refine_apply_mv_support_max:g}"
          f"/{args.depth_refine_apply_mv_coverage_min:g}"
          f"/erode{args.depth_refine_apply_erode_px} "
          f"sgate={args.support_gate_unet}/h{args.support_gate_hidden}"
          f"/init{args.support_gate_init:g}"
          f"/floor{args.support_gate_floor:g}"
          f"/scale{args.support_gate_delta_scale:g}"
          f"/detach{args.support_gate_detach_inputs}"
          f"/tol{args.support_gate_depth_tol_frac:g}"
          f"/mv{args.support_gate_multiview_target}"
          f"/refs{args.support_gate_multiview_refs}"
          f"/gt{args.support_gate_gt_weight:g}"
          f"/prior{args.support_gate_prior_weight:g}"
          f"/tv{args.support_gate_tv_weight:g} "
          f"surfconf={args.surface_confidence_unet}/h{args.surface_confidence_hidden}"
          f"/init{args.surface_confidence_init:g}"
          f"/floor{args.surface_confidence_floor:g}"
          f"/opmax{args.surface_confidence_opacity_max:g}"
          f"/gstr{args.surface_confidence_gate_strength:g}"
          f"/sstr{args.surface_confidence_scale_strength:g}"
          f"/sfloor{args.surface_confidence_scale_floor:g}"
          f"/scale{args.surface_confidence_delta_scale:g}"
          f"/gt{args.surface_confidence_gt_weight:g}"
          f"/tpos{args.surface_confidence_target_pos_min:g}"
          f"/tneg{args.surface_confidence_target_neg_max:g}"
          f"/pw{args.surface_confidence_positive_weight:g}"
          f"/nw{args.surface_confidence_negative_weight:g}"
          f"/sw{args.surface_confidence_score_weight:g}"
          f"/protect{args.surface_confidence_protect_support_min:g}"
          f"/{args.surface_confidence_protect_conflict_max:g}"
          f"/{args.surface_confidence_protect_coverage_min:g} "
          f"surfref={args.surface_refine_unet}/h{args.surface_refine_hidden}"
          f"/init{args.surface_refine_init:g}"
          f"/floor{args.surface_refine_opacity_floor:g}"
          f"/opscale{args.surface_refine_opacity_delta_scale:g}"
          f"/sscale{args.surface_refine_scale_delta_scale:g}"
          f"/sfloor{args.surface_refine_scale_floor:g}"
          f"/rgbscale{args.surface_refine_rgb_delta_scale:g}"
          f"/rgbgt{args.surface_refine_rgb_gt_weight:g}"
          f"/rgbgrad{args.surface_refine_rgb_grad_gt_weight:g}"
          f"/prior{args.surface_refine_prior_weight:g}"
          f"/tv{args.surface_refine_tv_weight:g}"
          f"/ckpt{args.surface_refine_checkpoint} "
          f"fcand={args.fusion_candidate_gate}/h{args.fusion_candidate_hidden}"
          f"x{args.fusion_candidate_layers}"
          f"/cfeat{args.fusion_candidate_coord_features}"
          f"/rfeat{args.fusion_candidate_rich_features}"
          f"/vfeat{args.fusion_candidate_voxel_features}"
          f"/nfeat{args.fusion_candidate_neighbor_features}"
          f"/nr{args.fusion_candidate_neighbor_radius}"
          f"/ckpt{args.fusion_candidate_checkpoint}"
          f"/chunk{args.fusion_candidate_chunk_size}"
          f"/sdelta{args.fusion_candidate_score_delta_scale:g}"
          f"/odelta{args.fusion_candidate_opacity_delta_scale:g}"
          f"/init{args.fusion_candidate_opacity_init:g}"
          f"/floor{args.fusion_candidate_opacity_floor:g}"
          f"/gt{args.fusion_candidate_gt_weight:g}"
          f"/gtsrc{args.fusion_candidate_gt_source}"
          f"/prior{args.fusion_candidate_prior_weight:g} "
          f"coord={args.coord_inject} "
          f"cond={args.condition_source}:{args.cond_subdir or 'target'}:{args.cond_view_indices or 'default'} "
          f"cmask={args.condition_mask_source}"
          f"/thr{args.condition_mask_rgb_threshold:g}"
          f"/soft{args.condition_mask_rgb_softness:g}"
          f"/e{args.condition_mask_rgb_erode_px}"
          f"/d{args.condition_mask_rgb_dilate_px}"
          f"/ref{args.condition_mask_refine_unet}"
          f"/h{args.condition_mask_refine_hidden}"
          f"/s{args.condition_mask_refine_scale:g} "
          f"colorcal={args.condition_color_calibration} "
          f"depthcal={args.condition_depth_calibration}"
          f"/objs{args.condition_depth_calib_max_objects}"
          f"/views{args.condition_depth_calib_views} "
          f"dmed={args.condition_depth_median_radius_px}"
          f"/th{args.condition_depth_median_thresh_frac:g}"
          f"/mix{args.condition_depth_median_mix:g} "
          f"rgbref={args.condition_rgb_refine_unet}"
          f"/h{args.condition_rgb_refine_hidden}"
          f"/s{args.condition_rgb_refine_scale:g}"
          f"/gt{args.condition_rgb_refine_gt_weight:g} "
          f"rgbdref={args.condition_rgbd_refine_unet}"
          f"/{args.condition_rgbd_refine_arch}"
          f"/h{args.condition_rgbd_refine_hidden}"
          f"/ctx{args.condition_rgbd_refine_context_layers}"
          f"/mview{args.condition_rgbd_refine_multiview_features}"
          f"/rgbs{args.condition_rgbd_refine_rgb_scale:g}"
          f"/ds{args.condition_rgbd_refine_depth_scale:g}"
          f"/rgbgt{args.condition_rgbd_refine_rgb_gt_weight:g}"
          f"/dgt{args.condition_rgbd_refine_depth_gt_weight:g} "
          f"pose={args.condition_pose_head}"
          f"/h{args.condition_pose_hidden}"
          f"/ctx{args.condition_pose_context_layers}"
          f"/dn{args.condition_pose_depth_norm}"
          f"/pred{args.condition_pose_use_predicted}"
          f"/freeze{args.freeze_condition_pose_head}"
          f"/w{args.condition_pose_weight:g}"
          f"/cf{args.condition_pose_center_weight:g}"
          f"/ff{args.condition_pose_forward_weight:g}"
          f"/df{args.condition_pose_dist_weight:g} "
          f"daff={args.condition_depth_affine_head}"
          f"/h{args.condition_depth_affine_hidden}"
          f"x{args.condition_depth_affine_layers}"
          f"/sr{args.condition_depth_affine_scale_range:g}"
          f"/sh{args.condition_depth_affine_shift_range:g}"
          f"/gt{args.condition_depth_affine_gt_weight:g}"
          f"/prior{args.condition_depth_affine_prior_weight:g} "
          f"dconf={args.condition_depth_confidence_unet}"
          f"/h{args.condition_depth_confidence_hidden}"
          f"/chunk{args.condition_depth_confidence_chunk_views}"
          f"/mview{args.condition_depth_confidence_multiview_features}"
          f"/init{args.condition_depth_confidence_init:g}"
          f"/floor{args.condition_depth_confidence_floor:g}"
          f"/scale{args.condition_depth_confidence_delta_scale:g}"
          f"/gt{args.condition_depth_confidence_gt_weight:g}"
          f"/tol{args.condition_depth_confidence_tol_frac:g}"
          f"/ntol{args.condition_depth_confidence_neg_tol_frac:g}"
          f"/prior{args.condition_depth_confidence_prior_weight:g}"
          f"/tv{args.condition_depth_confidence_tv_weight:g} "
          f"cdepth={args.cond_depth_subdir or 'default'} "
          f"cvisdepth={args.cond_visibility_depth_subdir or 'same'} "
          f"cconf={args.cond_conf_subdir or 'none'}"
          f"/pow{args.condition_confidence_power}"
          f"/cnorm{args.condition_confidence_normalize} "
          f"fuse_support={args.fusion_min_support} "
          f"ffiltw={args.fusion_filter_silhouette_weight:g}/{args.fusion_filter_front_weight:g} "
          f"voxel={args.fusion_voxel_size_frac}/min{args.fusion_voxel_min_count}"
          f"/k{args.fusion_voxel_max_per_cell}/{args.fusion_voxel_mode}"
          f"/rgb{args.fusion_voxel_color_mode}"
          f"{args.fusion_voxel_color_select_mix:g}"
          f"/rep{args.fusion_voxel_representative}"
          f"/softT{args.fusion_voxel_score_softmax_temp:g}"
          f"/softOp{args.fusion_voxel_score_soft_opacity_mix:g}"
          f"/softGeom{args.fusion_voxel_score_soft_geometry_mix:g}"
          f"/scale{args.fusion_voxel_scale_mult}"
          f"/z{args.fusion_voxel_scale_floor_z_mult}"
          f"/lowscale{args.fusion_voxel_low_support_scale_mult} "
          f"avgdecay={args.fusion_voxel_average_dist_decay:g} "
          f"detailscale={args.fusion_voxel_detail_scale_min:g} "
          f"neigh={args.fusion_voxel_neighbor_min}"
          f"/r{args.fusion_voxel_neighbor_radius}"
          f"/d{args.fusion_voxel_neighbor_opacity_decay:g} "
          f"prop={args.fusion_voxel_support_propagation_steps}"
          f"/r{args.fusion_voxel_support_propagation_radius}"
          f"/d{args.fusion_voxel_support_propagation_opacity_decay:g} "
          f"cov{args.fusion_voxel_coverage_opacity_mult:g}"
          f"/s{args.fusion_voxel_coverage_scale_mult:g} "
          f"pcaquat={args.fusion_voxel_pca_quat} "
          f"score={args.fusion_voxel_score_depth}"
          f"/cw{args.fusion_voxel_score_conflict_weight:g} "
          f"/rgb{args.fusion_voxel_score_color}"
          f"/w{args.fusion_voxel_score_color_weight:g} "
          f"/conf{args.fusion_voxel_score_confidence}"
          f"/cnorm{args.fusion_voxel_score_confidence_normalize}"
          f"/csup{args.fusion_voxel_score_confidence_supports}"
          f"/cconf{args.fusion_voxel_score_confidence_conflicts} "
          f"/opnorm{args.fusion_voxel_score_opacity_norm:g} "
          f"vgate={args.view_opacity_gate}/floor{args.view_opacity_floor:g}"
          f"/pow{args.view_opacity_power:g}/flip{args.view_opacity_flip} "
          f"sh={args.fusion_sh_degree}/mix{args.fusion_sh_mix:g} "
          f"mask_erode={args.condition_mask_erode_px} "
          f"mask_blur={args.condition_mask_blur_px} "
          f"rgb_inpaint={args.condition_rgb_inpaint_px} "
          f"guidedps={args.image_guided_plane_sweep_depth} "
          f"hullclamp={args.image_hull_clamp_depth}"
          f"/{args.image_hull_clamp_mode}"
          f"/tol{args.image_hull_clamp_tol_frac:g}"
          f"/max{args.image_hull_clamp_max_shift_frac:g} "
          f"img={args.image_condition} depthimg={args.image_depth_condition} "
          f"visimg={args.image_visibility_condition}/support{args.image_visibility_min_support}"
          f"/nn{args.image_visibility_nearest_refs} "
          f"photovis={args.image_photo_visibility_condition}/refs{args.image_photo_visibility_refs} "
          f"confvis={args.image_confidence_condition}/mask{args.condition_confidence_as_mask} "
          f"normimg={args.image_normal_condition} "
          f"headskip={args.image_head_skip} dskip={args.image_depth_skip} "
          f"vskip={args.image_visibility_skip} "
          f"bscale={args.image_boundary_scale_mult:g}/bw{args.image_boundary_width} "
          f"surfel={args.image_camera_quat} normalq={args.image_normal_quat} "
          f"dhead={args.explicit_depth_head} vhead={args.explicit_visibility_head} "
          f"surftok={args.use_surface_token_decoder}"
          f"/g{args.surface_token_grid_h}x{args.surface_token_grid_w}"
          f"/h{args.surface_token_hidden}"
          f"/slots{args.surface_token_slots}"
          f"/layers{args.surface_token_layers}"
          f"/heads{args.surface_token_heads}"
          f"/zlayers{args.surface_token_latent_layers}"
          f"/zpool{args.surface_token_latent_pool}"
          f"/zgate{args.surface_token_latent_gate_init:g}"
          f"/slotref{args.surface_token_slot_refine_layers}"
          f"/slotmlp{args.surface_token_slot_refine_mlp_ratio}"
          f"/slotgate{args.surface_token_slot_refine_gate_init:g}"
          f"/mres{args.surface_token_mean_res_frac:g}"
          f"/scale{args.surface_token_scale_frac:g}"
          f"/nscale{args.surface_token_normal_scale_frac:g}"
          f"/dnquat{int(bool(args.surface_token_depth_normal_quat or args.image_normal_quat))}"
          f"/dnblend{args.surface_token_depth_normal_blend:g}"
          f"/ldnblend{args.surface_token_learned_depth_normal_blend}"
          f"/ldnhead{args.surface_token_learned_depth_normal_blend_head}"
          f"/dnheadscale{args.surface_token_depth_normal_blend_head_scale:g}"
          f"/lsbase{args.surface_token_learned_scale_base}"
          f"/lscale{args.surface_token_learned_scale_head}"
          f"/lsmin{args.surface_token_learned_scale_min_frac:g}"
          f"/lsmax{args.surface_token_learned_scale_max_frac:g}"
          f"/qres{args.surface_token_quat_res_scale:g}"
          f"/opinit{args.surface_token_opacity_init:g}"
          f"/lobias{args.surface_token_learned_opacity_bias}"
          f"/loprior{args.surface_token_learned_opacity_prior}"
          f"/loutscale{args.surface_token_learned_output_scales}"
          f"/lcolaff{args.surface_token_learned_color_affine}"
          f"/colaffs{args.surface_token_color_affine_scale:g}"
          f"/policy{args.surface_token_learned_policy_head}"
          f"/pdepth{args.surface_token_policy_depth_res_frac:g}"
          f"/pmove{args.surface_token_policy_move_res_frac:g}"
          f"/pscale{args.surface_token_policy_scale_res_scale:g}"
          f"/pop{args.surface_token_policy_opacity_res_scale:g}"
          f"/pview{args.surface_token_policy_view_res_scale:g}"
          f"/pconf{args.surface_token_policy_confidence_res_scale:g}"
          f"/pkeep{args.surface_token_policy_keep_res_scale:g}"
          f"/pcov{args.surface_token_policy_coverage_scale_res_scale:g}"
          f"/pbirth{args.surface_token_policy_birth_res_scale:g}"
          f"/lpout{args.surface_token_learned_policy_output_scales}"
          f"/srcdc{args.surface_token_learned_source_depth_confidence_head}"
          f"/srcd{args.surface_token_source_depth_res_frac:g}"
          f"/srcc{args.surface_token_source_confidence_res_scale:g}"
          f"/lsrcdc{args.surface_token_learned_source_depth_confidence_scales}"
          f"/vsel{args.surface_token_learned_view_selector}"
          f"/vselh{args.surface_token_view_selector_hidden}"
          f"/vsels{args.surface_token_view_selector_score_scale:g}"
          f"/vselg{args.surface_token_view_selector_gate_scale:g}"
          f"/vseln{args.surface_token_view_selector_train_noise:g}"
          f"/prop{args.surface_token_proposal_count}"
          f"/props{args.surface_token_proposal_scale_frac:g}"
          f"/propns{args.surface_token_proposal_normal_scale_frac:g}"
          f"/propext{args.surface_token_proposal_extent_frac:g}"
          f"/propcov{args.surface_token_proposal_coverage_scale_res_scale:g}"
          f"/propop{args.surface_token_proposal_opacity_init:g}"
          f"/propseed{args.surface_token_proposal_seed_surface}"
          f"/propspool{args.surface_token_proposal_seed_pool}"
          f"/propsres{args.surface_token_proposal_surface_res_frac:g}"
          f"/propam{args.surface_token_proposal_anchor_mode}"
          f"/propat{args.surface_token_proposal_anchor_temp:g}"
          f"/propalw{args.surface_token_proposal_anchor_local_window}"
          f"/propag{args.surface_token_proposal_anchor_gate_init:g}"
          f"/propamrs{args.surface_token_proposal_anchor_mix_res_scale:g}"
          f"/propeprior{args.surface_token_proposal_anchor_even_prior:g}"
          f"/propepf{args.surface_token_proposal_anchor_even_prior_final:g}"
          f"/propepdecay{args.surface_token_proposal_anchor_even_prior_decay_steps}"
          f"/proppol{args.surface_token_learned_proposal_policy_head}"
          f"/proppkeep{args.surface_token_proposal_policy_keep_res_scale:g}"
          f"/proppconf{args.surface_token_proposal_policy_confidence_res_scale:g}"
          f"/proppcov{args.surface_token_proposal_policy_coverage_res_scale:g}"
          f"/lpsbase{args.surface_token_learned_proposal_scale_base}"
          f"/lpscale{args.surface_token_learned_proposal_scale_head}"
          f"/lpsmin{args.surface_token_learned_proposal_scale_min_frac:g}"
          f"/lpsmax{args.surface_token_learned_proposal_scale_max_frac:g}"
          f"/pplrm{args.surface_token_proposal_policy_lr_mult:g}"
          f"/ppkw{args.surface_token_proposal_policy_keep_weight:g}"
          f"/ppcw{args.surface_token_proposal_policy_confidence_weight:g}"
          f"/ppcovw{args.surface_token_proposal_policy_coverage_weight:g}"
          f"/pptm{args.surface_token_proposal_policy_target_mode}"
          f"/rgbdrop{args.surface_token_source_rgb_dropout_prob:g}"
          f"/detail{args.surface_token_detail_layer}"
          f"/dscale{args.surface_token_detail_scale_frac:g}"
          f"/dnscale{args.surface_token_detail_normal_scale_frac:g}"
          f"/dopinit{args.surface_token_detail_opacity_init:g}"
          f"/sreg{args.surface_token_scale_reg_weight:g}"
          f"/mreg{args.surface_token_mean_reg_weight:g}"
          f"/prgb{args.surface_token_projective_rgb_weight:g}"
          f"/pdepth{args.surface_token_projective_depth_weight:g}"
          f"/pop{args.surface_token_projective_opacity_weight:g}"
          f"/pmax{args.surface_token_projective_max_points}"
          f"/srccw{args.surface_token_source_confidence_weight:g}"
          f"/srcdw{args.surface_token_source_depth_weight:g}"
          f"/srcpts{args.surface_token_source_policy_points}"
          f"/srctpts{args.surface_token_source_policy_target_points}"
          f"/srctol{args.surface_token_source_policy_depth_tol_frac:g}"
          f"/srccts{args.surface_token_source_policy_confidence_target_scale:g}"
          f"/srcsm{args.surface_token_source_policy_support_mode}"
          f"/srcptm{args.surface_token_source_policy_target_mode}"
          f"/pcovw{args.surface_token_proposal_cover_weight:g}"
          f"/psurfw{args.surface_token_proposal_surface_weight:g}"
          f"/popw{args.surface_token_proposal_opacity_weight:g}"
          f"/prgbw{args.surface_token_proposal_rgb_weight:g}"
          f"/pdetw{args.surface_token_proposal_detail_weight:g}"
          f"/pdetth{args.surface_token_proposal_detail_edge_thresh:g}"
          f"/paew{args.surface_token_proposal_anchor_entropy_weight:g}"
          f"/pauw{args.surface_token_proposal_anchor_usage_weight:g}"
          f"/pacw{args.surface_token_proposal_anchor_collision_weight:g}"
          f"/pcpts{args.surface_token_proposal_cover_points} "
          f"/srgb{args.surface_token_scaffold_rgb_weight:g}"
          f"/sa{args.surface_token_scaffold_alpha_weight:g}"
          f"/sd{args.surface_token_scaffold_detail_weight:g}"
          f"/sda{args.surface_token_scaffold_detail_alpha_min:g}"
          f"/sm{args.surface_token_scaffold_margin:g} "
          f"canvox={args.use_canonical_voxel_decoder}"
          f"/g{args.canonical_voxel_grid_h}x{args.canonical_voxel_grid_w}"
          f"/h{args.canonical_voxel_hidden}"
          f"/layers{args.canonical_voxel_layers}"
          f"/heads{args.canonical_voxel_heads}"
          f"/zlayers{args.canonical_voxel_latent_layers}"
          f"/slots{args.canonical_voxel_scene_slots}"
          f"/vox{args.canonical_voxel_size_frac:g}"
          f"/max{args.canonical_voxel_max_voxels}"
          f"/k{args.canonical_voxel_gaussians_per_voxel}"
          f"/coff{args.canonical_voxel_child_offset_mult:g}"
          f"/pool{args.canonical_voxel_latent_pool}"
          f"/opinit{args.canonical_voxel_opacity_init:g}"
          f"/opsup{args.canonical_voxel_opacity_support_target:g}"
          f"/opw{args.canonical_voxel_opacity_prior_weight:g}"
          f"/vfeat{args.canonical_voxel_view_feature_channels} "
          f"oracle_depth={args.oracle_anchor_depth} maxtrain={args.max_train_objects} "
          f"accum={args.accum} opt={args.optimizer} "
          f"adaptloss={args.adaptive_loss_weights}:{args.adaptive_loss_names} "
          f"wandb={args.wandb_mode}", flush=True)
    it = iter(loader); t0 = time.time()

    def run_eval_step(step: int) -> None:
        tp, tsh, top, train_rows = eval_objs(train_eval, chunk=args.eval_render_chunk)
        hp, hsh, _, heldout_rows = eval_objs(heldout, chunk=args.eval_render_chunk)
        mtp = sum(tp) / len(tp) if tp else float("nan")
        mts = sum(tsh) / len(tsh) if tsh else float("nan")
        mhp = sum(hp) / len(hp) if hp else float("nan")
        mhs = sum(hsh) / len(hsh) if hsh else float("nan")
        mto = sum(o[0] for o in top) / len(top) if top else float("nan")      # mean opacity
        tp99 = sum(o[1] for o in top) / len(top) if top else float("nan")     # p99 opacity
        tfrac = sum(o[2] for o in top) / len(top) if top else float("nan")    # frac(opacity > 0.1)
        pose_eval_msg = ""
        if condition_pose_head is not None:
            pose_eval_msg = (
                f" pose={_mean_row(heldout_rows, 'pose_center_deg'):.2f}"
                f"/{_mean_row(heldout_rows, 'pose_forward_deg'):.2f}"
                f"/{_mean_row(heldout_rows, 'pose_logdist_abs'):.4f}"
            )
        print(f"[phase2] EVAL step {step} train={mtp:.2f}/{mts:.3f} "
              f"heldout={mhp:.2f}/{mhs:.3f} gap={mtp - mhp:.2f} "
              f"op={mto:.4f} op99={tp99:.3f} opF={tfrac:.3f} "
              f"hm={_mean_row(heldout_rows, 'alpha_l1'):.3f}"
              f"/bg{_mean_row(heldout_rows, 'alpha_bg_mean'):.3f}"
              f"/fp{_mean_row(heldout_rows, 'alpha_fp_gt_0_1'):.3f}"
              f"/fn{_mean_row(heldout_rows, 'alpha_fn_le_0_5'):.3f}"
              f"/iou{_mean_row(heldout_rows, 'alpha_iou_0_5'):.3f}"
              f"{pose_eval_msg}",
              flush=True)
        if run:
            run.log({"eval/train_psnr": mtp, "eval/train_sharp": mts,
                     "eval/train_op": mto, "eval/train_op_p99": tp99,
                     "eval/train_op_frac01": tfrac, "eval/heldout_psnr": mhp,
                     "eval/heldout_sharp": mhs, "eval/gap_psnr": mtp - mhp,
                     "eval/heldout_alpha_l1": _mean_row(heldout_rows, "alpha_l1"),
                     "eval/heldout_alpha_bg": _mean_row(heldout_rows, "alpha_bg_mean"),
                     "eval/heldout_alpha_fp01": _mean_row(heldout_rows, "alpha_fp_gt_0_1"),
                     "eval/heldout_alpha_fn05": _mean_row(heldout_rows, "alpha_fn_le_0_5"),
                     "eval/heldout_alpha_iou05": _mean_row(heldout_rows, "alpha_iou_0_5"),
                     "eval/heldout_pose_center_deg": _mean_row(heldout_rows, "pose_center_deg"),
                     "eval/heldout_pose_forward_deg": _mean_row(heldout_rows, "pose_forward_deg"),
                     "eval/heldout_pose_logdist_abs": _mean_row(heldout_rows, "pose_logdist_abs")},
                    step=step)
            log_viz(step)
        save_eval_viz(step)
        write_eval_rows(step, "train", train_rows)
        write_eval_rows(step, "heldout", heldout_rows)

    if args.eval_only:
        run_eval_step(start_step)
        if run:
            run.finish()
        return
    if args.eval_before_train and start_step == 0:
        run_eval_step(-1)

    initial_proposal_anchor_even_prior = float(args.surface_token_proposal_anchor_even_prior)

    def _apply_proposal_anchor_even_prior(step_i: int) -> float:
        if surface_token_decoder is None:
            return 0.0
        decay_steps = max(int(args.surface_token_proposal_anchor_even_prior_decay_steps), 0)
        final = float(args.surface_token_proposal_anchor_even_prior_final)
        if decay_steps <= 0 or final < 0.0:
            value = initial_proposal_anchor_even_prior
        else:
            rel = min(1.0, max(0.0, float(step_i - start_step + 1) / float(decay_steps)))
            value = initial_proposal_anchor_even_prior + rel * (
                max(final, 0.0) - initial_proposal_anchor_even_prior
            )
        value = max(float(value), 0.0)
        if hasattr(surface_token_decoder, "proposal_anchor_even_prior"):
            surface_token_decoder.proposal_anchor_even_prior = value
        return value

    for step in range(start_step, args.steps):      # OPTIMIZER step; each accumulates args.accum objects
        current_anchor_even_prior = _apply_proposal_anchor_even_prior(step)
        opt.zero_grad()
        tot = 0.0; comp = {"l1": 0.0, "ssim": 0.0, "mask": 0.0, "hinge": 0.0,
                           "percep": 0.0, "percep_w": 0.0, "fg_color": 0.0,
                           "anchor_rgb": 0.0, "anchor_opacity": 0.0, "anchor_scale": 0.0,
                           "anchor_visibility": 0.0,
                           "sv_identity": 0.0, "sv_support": 0.0,
                           "sv_target_vis": 0.0,
                           "canonical_target_vis": 0.0,
                           "canonical_source_vis_distill": 0.0,
                           "fg_alpha": 0.0,
                           "grad": 0.0, "grad_w": 0.0,
                           "scaffold_detail": 0.0, "scaffold_detail_w": 0.0,
                           "detail_teacher": 0.0, "detail_teacher_w": 0.0,
                           "alpha_grad": 0.0, "alpha_grad_w": 0.0,
                           "alpha_interior_smooth": 0.0,
                           "alpha_interior_smooth_w": 0.0,
                           "alpha_anti_lattice": 0.0,
                           "alpha_anti_lattice_w": 0.0,
                           "depth": 0.0, "depth_si": 0.0, "depth_abs": 0.0,
                           "anchor_depth": 0.0, "depth_w": 0.0,
                           "opreg": 0.0, "opent": 0.0, "bg_alpha": 0.0,
                           "fill_prior": 0.0, "fill_tv": 0.0, "fill_delta": 0.0,
                           "fill_oracle": 0.0,
                           "depth_refine_delta": 0.0, "depth_refine_prior": 0.0,
                           "depth_refine_tv": 0.0, "depth_refine_gt": 0.0,
                           "depth_refine_metric_gt": 0.0,
                           "support_gate_delta": 0.0, "support_gate_prior": 0.0,
                           "support_gate_tv": 0.0, "support_gate_gt": 0.0,
                           "surface_confidence_delta": 0.0,
                           "surface_confidence_prior": 0.0,
                           "surface_confidence_tv": 0.0,
                           "surface_confidence_gt": 0.0,
                           "surface_refine_delta": 0.0,
                           "surface_refine_prior": 0.0,
                           "surface_refine_tv": 0.0,
                           "surface_refine_rgb_gt": 0.0,
                           "surface_refine_rgb_grad_gt": 0.0,
                           "surface_token_scale_reg": 0.0,
                           "surface_token_mean_reg": 0.0,
                           "surface_token_depth_normal_blend": 0.0,
                           "surface_token_selected_view_mean": 0.0,
                           "surface_token_selected_view_span": 0.0,
                           "surface_token_selected_view_max": 0.0,
                           "surface_token_candidate_view_count": 0.0,
                           "surface_token_source_depth_abs_frac": 0.0,
                           "surface_token_source_confidence_gate": 0.0,
                           "surface_token_policy_depth_abs_frac": 0.0,
                           "surface_token_policy_move_abs_frac": 0.0,
                           "surface_token_policy_scale_mult": 0.0,
                           "surface_token_policy_opacity_mult": 0.0,
                           "surface_token_policy_view_gate": 0.0,
                           "surface_token_policy_confidence_gate": 0.0,
                           "surface_token_policy_keep_gate": 0.0,
                           "surface_token_policy_coverage_mult": 0.0,
                           "surface_token_policy_birth_gate": 0.0,
                           "surface_token_projective_rgb": 0.0,
                           "surface_token_projective_depth": 0.0,
                           "surface_token_projective_opacity": 0.0,
                           "surface_token_source_confidence": 0.0,
                           "surface_token_source_depth": 0.0,
                           "surface_token_source_support": 0.0,
                           "surface_token_source_confidence_target": 0.0,
                           "surface_token_source_depth_target_abs_frac": 0.0,
                           "surface_token_proposal_cover": 0.0,
                           "surface_token_proposal_surface": 0.0,
                           "surface_token_proposal_opacity": 0.0,
                           "surface_token_proposal_rgb": 0.0,
                           "surface_token_proposal_detail": 0.0,
                           "surface_token_proposal_support": 0.0,
                           "surface_token_proposal_opacity_mean": 0.0,
                           "surface_token_proposal_detail_mean": 0.0,
                           "surface_token_proposal_coverage_mult": 0.0,
                           "surface_token_proposal_anchor_mix": 0.0,
                           "surface_token_proposal_anchor_entropy": 0.0,
                           "surface_token_proposal_anchor_entropy_loss": 0.0,
                           "surface_token_proposal_anchor_usage_loss": 0.0,
                           "surface_token_proposal_anchor_usage_perplexity": 0.0,
                           "surface_token_proposal_anchor_unique_frac": 0.0,
                           "surface_token_proposal_anchor_collision_loss": 0.0,
                           "surface_token_proposal_anchor_collision_frac": 0.0,
                           "surface_token_proposal_anchor_even_prior": current_anchor_even_prior,
                           "surface_token_proposal_policy_keep_gate": 0.0,
                           "surface_token_proposal_policy_confidence_gate": 0.0,
                           "surface_token_proposal_policy_coverage_mult": 0.0,
                           "surface_token_proposal_policy_keep": 0.0,
                           "surface_token_proposal_policy_confidence": 0.0,
                           "surface_token_proposal_policy_coverage": 0.0,
                           "surface_token_proposal_policy_keep_target": 0.0,
                           "surface_token_proposal_policy_confidence_target": 0.0,
                           "surface_token_proposal_policy_coverage_target": 0.0,
                           "surface_token_scaffold_rgb": 0.0,
                           "surface_token_scaffold_alpha": 0.0,
                           "surface_token_scaffold_detail": 0.0,
                           "canonical_scale_reg": 0.0,
                           "canonical_scale_reg_w": 0.0,
                           "fusion_candidate_delta": 0.0,
                           "fusion_candidate_prior": 0.0,
                           "fusion_candidate_gt": 0.0,
                           "output_alpha_refine_delta": 0.0,
                           "output_alpha_refine_prior": 0.0,
                           "output_alpha_refine_tv": 0.0,
                           "rgb_refine_gt": 0.0,
                           "rgbd_refine_delta": 0.0,
                           "rgbd_refine_prior": 0.0,
                           "rgbd_refine_tv": 0.0,
                           "rgbd_refine_rgb_gt": 0.0,
                           "rgbd_refine_depth_gt": 0.0,
                           "pose": 0.0,
                           "pose_center": 0.0,
                           "pose_forward": 0.0,
                           "pose_dist": 0.0,
                           "depth_affine_delta": 0.0,
                           "depth_affine_prior": 0.0,
                           "depth_affine_gt": 0.0,
                           "depth_conf_delta": 0.0,
                           "depth_conf_prior": 0.0,
                           "depth_conf_tv": 0.0,
                           "depth_conf_gt": 0.0,
                           "resid_rgb": 0.0, "resid_geom": 0.0,
                           "resid_depth": 0.0, "resid_opacity": 0.0,
                           "resid_offset": 0.0,
                           "resid_rgb_l2": 0.0, "resid_geom_l2": 0.0,
                           "resid_depth_l2": 0.0, "resid_opacity_l2": 0.0,
                           "resid_offset_l2": 0.0,
                           "op_mean": 0.0, "op_p99": 0.0, "op_frac01": 0.0,
                           "scale_mean": 0.0, "scale_frac98cap": 0.0,
                           "scale_raw_over_cap": 0.0}
        for _ in range(args.accum):
            try:
                sample = next(it)
            except StopIteration:
                it = iter(loader); sample = next(it)
            loss_i, comp_i = train_loss(sample, step)
            if loss_i.requires_grad:
                (loss_i / args.accum).backward()     # accumulate; grads sum across the micro-batch
            tot += float(loss_i.detach()) / args.accum
            for k in comp:
                comp[k] += comp_i[k] / args.accum
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0:
            objs = (step - start_step + 1) * args.accum
            sps = objs / (time.time() - t0)
            mem = _mem_stats()
            depth_msg = ""
            if args.depth_weight > 0 or args.anchor_depth_weight > 0:
                depth_msg = (f" depth={comp['depth']:.4f}"
                             f" anchor={comp['anchor_depth']:.4f}"
                             f" dw={comp['depth_w']:.3g}")
            color_msg = ""
            if (args.anchor_rgb_weight > 0 or args.fg_color_weight > 0
                    or args.anchor_opacity_weight > 0 or args.anchor_scale_weight > 0
                    or args.anchor_visibility_weight > 0 or args.bg_alpha_weight > 0
                    or args.fg_alpha_weight > 0
                    or args.condition_rgb_refine_gt_weight > 0):
                color_msg = (f" ar={comp['anchor_rgb']:.4f} ao={comp['anchor_opacity']:.4f}"
                             f" as={comp['anchor_scale']:.4f} av={comp['anchor_visibility']:.4f}"
                             f" fgc={comp['fg_color']:.4f}"
                             f" bga={comp['bg_alpha']:.4f}"
                             f" fga={comp['fg_alpha']:.4f}"
                             f" rgbgt={comp['rgb_refine_gt']:.4f}")
            rgbd_msg = ""
            if condition_rgbd_refine_head is not None:
                rgbd_msg = (f" rgbd_delta={comp.get('rgbd_refine_delta', 0.0):.4f}"
                            f" rgbd_p={comp.get('rgbd_refine_prior', 0.0):.4f}"
                            f" rgbd_tv={comp.get('rgbd_refine_tv', 0.0):.4f}"
                            f" rgbd_rgb={comp.get('rgbd_refine_rgb_gt', 0.0):.4f}"
                            f" rgbd_d={comp.get('rgbd_refine_depth_gt', 0.0):.4f}")
            pose_msg = ""
            if condition_pose_head is not None:
                pose_msg = (f" pose={comp.get('pose', 0.0):.4f}"
                            f"/c{comp.get('pose_center', 0.0):.3f}"
                            f"/f{comp.get('pose_forward', 0.0):.3f}"
                            f"/d{comp.get('pose_dist', 0.0):.3f}")
            daff_msg = ""
            if condition_depth_affine_head is not None:
                daff_msg = (f" daff_gt={comp.get('depth_affine_gt', 0.0):.4f}"
                            f" daff_delta={comp.get('depth_affine_delta', 0.0):.4f}"
                            f" daff_p={comp.get('depth_affine_prior', 0.0):.4f}")
            dconf_msg = ""
            if condition_depth_confidence_head is not None:
                dconf_msg = (f" dconf_gt={comp.get('depth_conf_gt', 0.0):.4f}"
                             f" dconf_delta={comp.get('depth_conf_delta', 0.0):.4f}"
                             f" dconf_p={comp.get('depth_conf_prior', 0.0):.4f}"
                             f" dconf_tv={comp.get('depth_conf_tv', 0.0):.4f}")
            sharp_msg = ""
            if (args.grad_weight > 0 or args.perceptual_weight > 0
                    or args.scaffold_detail_weight > 0
                    or args.detail_teacher_weight > 0
                    or args.alpha_grad_weight > 0
                    or args.alpha_interior_smooth_weight > 0
                    or args.alpha_anti_lattice_weight > 0):
                sharp_msg = (f" grad={comp['grad']:.4f} gw={comp['grad_w']:.3g}"
                             f" sdet={comp['scaffold_detail']:.4f}"
                             f" sdw={comp['scaffold_detail_w']:.3g}"
                             f" dteach={comp['detail_teacher']:.4f}"
                             f" dtw={comp['detail_teacher_w']:.3g}"
                             f" agrad={comp['alpha_grad']:.4f}"
                             f" agw={comp['alpha_grad_w']:.3g}"
                             f" aismooth={comp['alpha_interior_smooth']:.4f}"
                             f" aisw={comp['alpha_interior_smooth_w']:.3g}"
                             f" aal={comp['alpha_anti_lattice']:.4f}"
                             f" aalw={comp['alpha_anti_lattice_w']:.3g}"
                             f" perc={comp['percep']:.4f} pw={comp['percep_w']:.3g}")
            fill_msg = ""
            if args.anchor_render_mode in {"learned_fill", "learned_iblend_fill"}:
                fill_msg = (f" filld={comp['fill_delta']:.4f}"
                            f" fillp={comp['fill_prior']:.4f}"
                            f" filltv={comp['fill_tv']:.4f}"
                            f" fillo={comp['fill_oracle']:.4f}")
            dref_msg = ""
            if depth_refine_head is not None:
                dref_msg = (f" dref_gt={comp['depth_refine_gt']:.4f}"
                            f" dref_mgt={comp['depth_refine_metric_gt']:.4f}"
                            f" dref_delta={comp['depth_refine_delta']:.4f}"
                            f" dref_p={comp['depth_refine_prior']:.4f}"
                            f" dref_tv={comp['depth_refine_tv']:.4f}")
            sgate_msg = ""
            if support_gate_head is not None:
                sgate_msg = (f" sgate_gt={comp['support_gate_gt']:.4f}"
                             f" sgate_delta={comp['support_gate_delta']:.4f}"
                             f" sgate_p={comp['support_gate_prior']:.4f}"
                             f" sgate_tv={comp['support_gate_tv']:.4f}")
            surf_msg = ""
            if surface_confidence_head is not None:
                surf_msg = (f" surf_gt={comp['surface_confidence_gt']:.4f}"
                            f" surf_delta={comp['surface_confidence_delta']:.4f}"
                            f" surf_p={comp['surface_confidence_prior']:.4f}"
                            f" surf_tv={comp['surface_confidence_tv']:.4f}")
            surfref_msg = ""
            if surface_refine_head is not None:
                surfref_msg = (f" surfref_rgb={comp['surface_refine_rgb_gt']:.4f}"
                               f" surfref_grad={comp['surface_refine_rgb_grad_gt']:.4f}"
                               f" surfref_delta={comp['surface_refine_delta']:.4f}"
                               f" surfref_p={comp['surface_refine_prior']:.4f}"
                               f" surfref_tv={comp['surface_refine_tv']:.4f}")
            surftok_msg = ""
            if (args.surface_token_scale_reg_weight > 0
                    or args.surface_token_mean_reg_weight > 0
                    or args.surface_token_projective_rgb_weight > 0
                    or args.surface_token_projective_depth_weight > 0
                    or args.surface_token_projective_opacity_weight > 0
                    or args.surface_token_source_confidence_weight > 0
                    or args.surface_token_source_depth_weight > 0
                    or _adapt_has("surface_token_source_confidence")
                    or _adapt_has("surface_token_source_depth")
                    or args.surface_token_proposal_cover_weight > 0
                    or args.surface_token_proposal_surface_weight > 0
                    or args.surface_token_proposal_opacity_weight > 0
                    or args.surface_token_proposal_rgb_weight > 0
                    or args.surface_token_proposal_detail_weight > 0
                    or args.surface_token_proposal_anchor_entropy_weight > 0
                    or args.surface_token_proposal_anchor_usage_weight > 0
                    or args.surface_token_proposal_anchor_collision_weight > 0
                    or args.surface_token_proposal_policy_keep_weight > 0
                    or args.surface_token_proposal_policy_confidence_weight > 0
                    or args.surface_token_proposal_policy_coverage_weight > 0
                    or args.surface_token_learned_depth_normal_blend
                    or args.surface_token_learned_depth_normal_blend_head
                    or args.surface_token_learned_source_depth_confidence_head
                    or args.surface_token_learned_policy_head
                    or args.surface_token_learned_proposal_policy_head
                    or args.surface_token_scaffold_rgb_weight > 0
                    or args.surface_token_scaffold_alpha_weight > 0
                    or args.surface_token_scaffold_detail_weight > 0):
                surftok_msg = (f" surftok_sreg={comp['surface_token_scale_reg']:.4f}"
                               f" surftok_mreg={comp['surface_token_mean_reg']:.4f}"
                               f" stok_nb={comp['surface_token_depth_normal_blend']:.3f}"
                               f" vsel_mean={comp['surface_token_selected_view_mean']:.1f}"
                               f" vsel_span={comp['surface_token_selected_view_span']:.1f}"
                               f" vsel_max={comp['surface_token_selected_view_max']:.1f}"
                               f" vcand={comp['surface_token_candidate_view_count']:.0f}"
                               f" src_d={comp['surface_token_source_depth_abs_frac']:.6f}"
                               f" src_c={comp['surface_token_source_confidence_gate']:.6f}"
                               f" pol_d={comp['surface_token_policy_depth_abs_frac']:.4f}"
                               f" pol_m={comp['surface_token_policy_move_abs_frac']:.4f}"
                               f" pol_s={comp['surface_token_policy_scale_mult']:.4f}"
                               f" pol_o={comp['surface_token_policy_opacity_mult']:.4f}"
                               f" pol_v={comp['surface_token_policy_view_gate']:.4f}"
                               f" pol_c={comp['surface_token_policy_confidence_gate']:.4f}"
                               f" pol_k={comp['surface_token_policy_keep_gate']:.4f}"
                               f" pol_cov={comp['surface_token_policy_coverage_mult']:.4f}"
                               f" pol_b={comp['surface_token_policy_birth_gate']:.4f}"
                               f" stok_prgb={comp['surface_token_projective_rgb']:.4f}"
                               f" stok_pdepth={comp['surface_token_projective_depth']:.4f}"
                               f" stok_pop={comp['surface_token_projective_opacity']:.4f}"
                               f" src_sup={comp['surface_token_source_support']:.3f}"
                               f" src_ct={comp['surface_token_source_confidence_target']:.3f}"
                               f" src_conf_l={comp['surface_token_source_confidence']:.4f}"
                               f" src_depth_l={comp['surface_token_source_depth']:.4f}"
                               f" prop_cov={comp['surface_token_proposal_cover']:.4f}"
                               f" prop_surf={comp['surface_token_proposal_surface']:.4f}"
                               f" prop_op={comp['surface_token_proposal_opacity']:.4f}"
                               f" prop_rgb={comp['surface_token_proposal_rgb']:.4f}"
                               f" prop_det={comp['surface_token_proposal_detail']:.4f}"
                               f" prop_sup={comp['surface_token_proposal_support']:.3f}"
                               f" prop_om={comp['surface_token_proposal_opacity_mean']:.4g}"
                               f" prop_dm={comp['surface_token_proposal_detail_mean']:.4g}"
                               f" prop_cm={comp['surface_token_proposal_coverage_mult']:.4f}"
                               f" prop_am={comp['surface_token_proposal_anchor_mix']:.3f}"
                               f" prop_ep={comp['surface_token_proposal_anchor_even_prior']:.3f}"
                               f" prop_ae={comp['surface_token_proposal_anchor_entropy']:.3f}"
                               f" prop_pk={comp['surface_token_proposal_policy_keep_gate']:.4f}"
                               f" prop_pc={comp['surface_token_proposal_policy_confidence_gate']:.4f}"
                               f" prop_pcov={comp['surface_token_proposal_policy_coverage_mult']:.4f}"
                               f" prop_pkt={comp['surface_token_proposal_policy_keep_target']:.3f}"
                               f" prop_pct={comp['surface_token_proposal_policy_confidence_target']:.3f}"
                               f" prop_pcvt={comp['surface_token_proposal_policy_coverage_target']:.3f}"
                               f" prop_pkl={comp['surface_token_proposal_policy_keep']:.4f}"
                               f" prop_pcl={comp['surface_token_proposal_policy_confidence']:.4f}"
                               f" prop_pcvl={comp['surface_token_proposal_policy_coverage']:.4f}"
                               f" prop_ael={comp['surface_token_proposal_anchor_entropy_loss']:.4f}"
                               f" prop_aul={comp['surface_token_proposal_anchor_usage_loss']:.4f}"
                               f" prop_ap={comp['surface_token_proposal_anchor_usage_perplexity']:.3f}"
                               f" prop_au={comp['surface_token_proposal_anchor_unique_frac']:.3f}"
                               f" prop_acl={comp['surface_token_proposal_anchor_collision_loss']:.4f}"
                               f" prop_acf={comp['surface_token_proposal_anchor_collision_frac']:.3f}"
                               f" stok_srgb={comp['surface_token_scaffold_rgb']:.4f}"
                               f" stok_sa={comp['surface_token_scaffold_alpha']:.4f}"
                               f" stok_sd={comp['surface_token_scaffold_detail']:.4f}")
            if args.canonical_scale_reg_weight > 0:
                surftok_msg = (
                    f"{surftok_msg} can_sreg={comp['canonical_scale_reg']:.4f}"
                    f" can_sw={comp['canonical_scale_reg_w']:.3g}"
                )
            fcand_msg = ""
            if fusion_candidate_head is not None:
                fcand_msg = (f" fcand_gt={comp['fusion_candidate_gt']:.4f}"
                             f" fcand_delta={comp['fusion_candidate_delta']:.4f}"
                             f" fcand_p={comp['fusion_candidate_prior']:.4f}")
            outa_msg = ""
            if output_alpha_refine_head is not None:
                outa_msg = (f" outa_delta={comp['output_alpha_refine_delta']:.4f}"
                            f" outa_p={comp['output_alpha_refine_prior']:.4f}"
                            f" outa_tv={comp['output_alpha_refine_tv']:.4f}")
            sv_msg = ""
            if (args.sparse_voxel_identity_reg_weight > 0
                    or args.sparse_voxel_support_reg_weight > 0
                    or args.sparse_voxel_target_vis_weight > 0
                    or args.canonical_target_vis_weight > 0
                    or args.canonical_source_vis_distill_weight > 0):
                sv_msg = (f" sv_id={comp['sv_identity']:.4f}"
                          f" sv_sup={comp['sv_support']:.4f}"
                          f" sv_tvis={comp['sv_target_vis']:.4f}"
                          f" can_tvis={comp['canonical_target_vis']:.4f}"
                          f" can_svis={comp['canonical_source_vis_distill']:.4f}")
            resid_msg = ""
            if (args.residual_rgb_weight > 0 or args.residual_geom_weight > 0
                    or args.residual_depth_weight > 0
                    or args.residual_opacity_weight > 0
                    or args.residual_offset_weight > 0):
                resid_msg = (f" resid={comp['resid_rgb']:.4f}/{comp['resid_geom']:.4f}"
                             f"/{comp['resid_depth']:.4f}/{comp['resid_opacity']:.4f}"
                             f"/{comp['resid_offset']:.4f}")
            print(f"[phase2] step {step} ({objs} objs) loss={tot:.4f}{depth_msg} "
                  f"{color_msg}{rgbd_msg}{pose_msg}{daff_msg}{dconf_msg}{sharp_msg}{fill_msg}{dref_msg}{sgate_msg}{surf_msg}{surfref_msg}{surftok_msg}{fcand_msg}{outa_msg}"
                  f"{sv_msg}{resid_msg} "
                  f"obj/s={sps:.2f} "
                  f"{_mem_msg(mem)}", flush=True)
            if run:
                run.log({"train/loss": tot, **{f"train/{k}": v for k, v in comp.items()},
                         "lr": sched.get_last_lr()[0], "obj_per_s": sps,
                         **{f"mem/{k}": v for k, v in mem.items()}}, step=step)
        do_eval = step == args.steps - 1 or (
            step % args.eval_every == 0 and (step > 0 or args.eval_at_step0)
        )
        if do_eval and args.save_before_eval:
            save_checkpoint(step)
            if args.save_named_checkpoints:
                save_checkpoint(step, name=f"phase2_step_{step:06d}_pre_eval.pt")
        if do_eval:
            run_eval_step(step)
        if do_eval or (args.save_every > 0 and step > start_step and step % args.save_every == 0):
            save_checkpoint(step)
            if args.save_named_checkpoints:
                save_checkpoint(step, name=f"phase2_step_{step:06d}.pt")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
