"""Feed-forward Gaussian fusion utilities.

These helpers operate on already predicted Gaussian dictionaries. They are
intended for diagnostics and lightweight deterministic fusion, not per-object
optimization.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from decoder.data import zdepth_to_raydist
from decoder.clean.geometry import ray_dirs_world


def voxel_fuse_params(params: dict, voxel_size: float,
                      min_count: int = 1,
                      max_per_voxel: int = 1,
                      mode: str = "select",
                      color_mode: str = "average",
                      color_select_mix: float = 1.0,
                      representative_mode: str = "opacity",
                      score_key: str = "_fusion_score",
                      score_softmax_temp: float = 1.0,
                      score_soft_opacity_mix: float = 0.0,
                      score_soft_geometry_mix: float = 0.0,
                      scale_floor: float | None = None,
                      scale_floor_z_mult: float = 1.0,
                      low_support_scale_floor_mult: float = 1.0,
                      scale_floor_detail_key: str | None = None,
                      scale_floor_detail_min: float = 1.0,
                      average_dist_decay: float = 0.0,
                      neighbor_support_radius: int = 1,
                      neighbor_support_min: int = 0,
                      neighbor_opacity_decay: float = 0.0,
                      support_propagation_steps: int = 0,
                      support_propagation_radius: int = 1,
                      support_propagation_opacity_decay: float = 0.0,
                      support_key: str | None = None,
                      low_support_opacity_decay: float = 0.0,
                      coverage_opacity_mult: float = 0.0,
                      coverage_scale_mult: float = 1.0,
                      pca_quat: bool = False) -> tuple[dict, dict]:
    """Fuse nearby splats by world-space voxel.

    In ``select`` mode this keeps up to ``max_per_voxel`` representatives.
    With ``max_per_voxel=1`` the representative is the highest-opacity splat.
    For select mode, ``color_select_mix < 1`` blends representative RGB toward
    the opacity-weighted voxel average without averaging geometry.
    In ``average`` mode this emits one opacity-weighted representative per
    voxel and optionally lifts its scale to ``scale_floor`` for coverage. For
    average fusion, ``color_mode=select`` can keep or blend in representative
    RGB instead of pure averaged colors; this avoids texture washout while still
    stabilizing geometry. ``color_mode=score_select`` decouples texture from
    geometry by taking RGB from the highest-scored splat in the voxel while the
    geometric representative can still be opacity/medoid-selected.
    ``color_mode=score_soft`` uses a differentiable per-voxel softmax over
    ``score_key`` to blend source RGB, giving learned score heads a render-loss
    path without moving geometry. ``score_soft_opacity_mix`` similarly blends
    output opacity toward a score-soft source opacity.
    ``score_soft_geometry_mix`` blends output geometry toward score-soft source
    geometry, giving learned pre-fusion score heads a render-loss path through
    surface position/scale rather than only color or opacity. This is off by
    default because it changes the deterministic scaffold.
    ``representative_mode=medoid`` picks the splat closest to the voxel's
    opacity-weighted center, which is less view-order-biased than the default
    highest-opacity representative. ``representative_mode=score`` picks the
    highest value from ``score_key`` when available.

    ``scale_floor_detail_key`` optionally points at a per-splat [0,1] detail
    signal. Averaged voxels with high detail get a smaller scale floor, so flat
    regions keep coverage without smearing texture edges as much.
    ``scale_floor_z_mult`` can keep the local normal axis thinner than the
    tangent axes; for surfel-like Gaussians this avoids making fused points
    volumetric just to cover tangent-plane holes.
    ``low_support_scale_floor_mult`` shrinks the scale floor for soft-kept
    low-support voxels, so one-view fill can stay thin instead of becoming
    broad translucent sheets.
    ``average_dist_decay`` robustifies average fusion by downweighting splats
    far from the opacity-weighted voxel center before averaging attributes.
    ``neighbor_support_min`` downweights or removes output voxels that are not
    near enough supported voxels. This suppresses isolated low-support shell
    flecks while keeping low-support fill attached to a real surface.
    ``support_propagation_steps`` is a softer connectedness check: starting
    from supported voxels, it iteratively expands through nearby output voxels
    and downweights or removes voxels that remain disconnected. This preserves
    low-support fill connected to real multi-view support while suppressing
    detached shell spray.
    ``coverage_opacity_mult`` optionally appends a dim opacity-weighted average
    coverage splat for each selected voxel. This lets a sharp representative
    carry detail while a low-opacity average splat fills pinholes.
    ``pca_quat`` replaces fused quaternions with a local covariance normal from
    the source points in each voxel, so thin scale floors align to the fused
    surface rather than the selected source camera.

    ``min_count`` optionally drops voxels seen fewer than that many times, which
    is a deterministic support filter when several anchor shells are
    concatenated. If ``support_key`` is present in ``params``, support counts
    distinct source ids rather than raw splats, so adjacent pixels from the same
    anchor do not masquerade as multi-view support.

    Returns ``(fused_params, stats)``. The selected tensors still carry gradient
    to the kept splats; averaged tensors carry gradient through the weighted
    averages. The voxel assignment itself is discrete.
    """
    if mode not in {"select", "average"}:
        raise ValueError("voxel fusion mode must be 'select' or 'average'")
    if color_mode not in {"average", "select", "score_select", "score_soft"}:
        raise ValueError(
            "voxel fusion color mode must be 'average', 'select', "
            "'score_select', or 'score_soft'"
        )
    color_select_mix = min(max(float(color_select_mix), 0.0), 1.0)
    score_softmax_temp = max(float(score_softmax_temp), 1e-6)
    score_soft_opacity_mix = min(max(float(score_soft_opacity_mix), 0.0), 1.0)
    score_soft_geometry_mix = min(max(float(score_soft_geometry_mix), 0.0), 1.0)
    if representative_mode not in {"opacity", "medoid", "score"}:
        raise ValueError("voxel representative mode must be 'opacity', 'medoid', or 'score'")
    if voxel_size <= 0:
        return params, {
            "input": int(params["mean"].shape[0]),
            "output": int(params["mean"].shape[0]),
            "voxels": int(params["mean"].shape[0]),
            "dropped_low_support": 0,
        }

    mean = params["mean"]
    n = mean.shape[0]
    if n == 0:
        return params, {"input": 0, "output": 0, "voxels": 0, "dropped_low_support": 0}

    q = torch.floor(mean.detach() / float(voxel_size)).to(torch.int64)
    q = q - q.amin(dim=0, keepdim=True)
    dims = q.amax(dim=0) + 1
    key = q[:, 0] + dims[0] * (q[:, 1] + dims[1] * q[:, 2])

    def is_per_splat(v: object) -> bool:
        return hasattr(v, "shape") and v.shape[:1] == (n,)

    _, inverse = torch.unique(key, sorted=False, return_inverse=True)
    n_vox = int(inverse.max().item()) + 1
    op = params["opacity"].reshape(-1).detach()

    counts = torch.bincount(inverse, minlength=n_vox)
    if support_key and support_key in params:
        src = params[support_key].reshape(-1).detach().to(torch.int64)
        src = src - src.amin()
        src_base = int(src.max().item()) + 1
        pair = inverse * max(src_base, 1) + src
        unique_pair = torch.unique(pair, sorted=False)
        support_counts = torch.bincount(
            unique_pair // max(src_base, 1), minlength=n_vox
        )
    else:
        support_counts = counts
    min_support = max(int(min_count), 1)
    support = support_counts >= min_support
    soft_support = low_support_opacity_decay > 0 and min_support > 1
    missing_support = (min_support - support_counts).clamp_min(0).to(mean.dtype)
    support_decay = torch.exp(-float(low_support_opacity_decay) * missing_support)
    coverage_opacity_mult = max(float(coverage_opacity_mult), 0.0)
    coverage_scale_mult = max(float(coverage_scale_mult), 0.0)

    idx = torch.arange(n, device=mean.device, dtype=torch.long)
    max_per_voxel = max(int(max_per_voxel), 1)
    max_op = torch.full((n_vox,), -1.0, dtype=op.dtype, device=op.device)
    max_op.scatter_reduce_(0, inverse, op, reduce="amax", include_self=True)
    coverage_rows = None

    def choose_representative() -> torch.Tensor:
        sentinel = torch.full_like(idx, n)
        if representative_mode == "score":
            if score_key in params:
                score = params[score_key].reshape(-1).detach().to(dtype=op.dtype, device=op.device)
                score = torch.nan_to_num(score, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
            else:
                score = op
            max_score = torch.full((n_vox,), -torch.inf, dtype=score.dtype, device=score.device)
            max_score.scatter_reduce_(0, inverse, score, reduce="amax", include_self=True)
            candidates = torch.where(score >= max_score[inverse] - 1e-8, idx, sentinel)
        elif representative_mode == "medoid":
            w_rep = params["opacity"].reshape(-1, 1).detach().clamp_min(1e-6).to(mean.dtype)
            denom_rep = torch.zeros(n_vox, 1, dtype=mean.dtype, device=mean.device)
            denom_rep.scatter_add_(0, inverse[:, None].expand(-1, 1), w_rep)
            center = torch.zeros(n_vox, mean.shape[1], dtype=mean.dtype, device=mean.device)
            center.scatter_add_(0, inverse[:, None].expand(-1, mean.shape[1]), mean.detach() * w_rep)
            center = center / denom_rep.clamp_min(1e-6)
            dist = ((mean.detach() - center[inverse]) ** 2).sum(dim=-1)
            min_dist = torch.full((n_vox,), torch.inf, dtype=dist.dtype, device=dist.device)
            min_dist.scatter_reduce_(0, inverse, dist, reduce="amin", include_self=True)
            candidates = torch.where(dist <= min_dist[inverse] + 1e-12, idx, sentinel)
        else:
            candidates = torch.where(op >= max_op[inverse] - 1e-8, idx, sentinel)
        out = torch.full((n_vox,), n, dtype=torch.long, device=mean.device)
        out.scatter_reduce_(0, inverse, candidates, reduce="amin", include_self=True)
        return out

    chosen = choose_representative()

    def choose_score_representative() -> torch.Tensor:
        if score_key not in params:
            return chosen
        score = params[score_key].reshape(-1).detach().to(dtype=op.dtype, device=op.device)
        score = torch.nan_to_num(score, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
        max_score = torch.full((n_vox,), -torch.inf, dtype=score.dtype, device=score.device)
        max_score.scatter_reduce_(0, inverse, score, reduce="amax", include_self=True)
        sentinel = torch.full_like(idx, n)
        candidates = torch.where(score >= max_score[inverse] - 1e-8, idx, sentinel)
        out = torch.full((n_vox,), n, dtype=torch.long, device=mean.device)
        out.scatter_reduce_(0, inverse, candidates, reduce="amin", include_self=True)
        return torch.where(out < n, out, chosen)

    score_soft_weight_cache: torch.Tensor | None = None

    def score_soft_weights() -> torch.Tensor | None:
        nonlocal score_soft_weight_cache
        if score_key not in params:
            return None
        if score_soft_weight_cache is not None:
            return score_soft_weight_cache
        score = params[score_key].reshape(n, -1)[:, :1].to(dtype=mean.dtype, device=mean.device)
        score = torch.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
        max_score = torch.full((n_vox, 1), -1e6, dtype=score.dtype, device=score.device)
        max_score.scatter_reduce_(
            0, inverse[:, None].expand(-1, 1), score.detach(),
            reduce="amax", include_self=True,
        )
        exp_score = torch.exp(((score - max_score[inverse]) / score_softmax_temp).clamp(-60.0, 60.0))
        denom = torch.zeros(n_vox, 1, dtype=exp_score.dtype, device=exp_score.device)
        denom.scatter_add_(0, inverse[:, None].expand(-1, 1), exp_score)
        score_soft_weight_cache = exp_score / denom[inverse].clamp_min(1e-12)
        return score_soft_weight_cache

    def score_soft_tensor(v: torch.Tensor, out_vox: torch.Tensor) -> torch.Tensor:
        weights = score_soft_weights()
        if weights is None or not is_per_splat(v) or not torch.is_floating_point(v):
            fallback_idx = chosen[out_vox].clamp_max(n - 1)
            return v[fallback_idx]
        flat = v.reshape(n, -1)
        ww = weights.to(dtype=flat.dtype)
        out = torch.zeros(n_vox, flat.shape[1], dtype=flat.dtype, device=flat.device)
        out.scatter_add_(0, inverse[:, None].expand(-1, flat.shape[1]), flat * ww)
        return out[out_vox].reshape((out_vox.numel(),) + v.shape[1:])

    def quat_local_z(q_in: torch.Tensor) -> torch.Tensor:
        qn = F.normalize(q_in, dim=-1)
        wq, xq, yq, zq = qn.unbind(-1)
        return torch.stack([
            2.0 * (xq * zq + wq * yq),
            2.0 * (yq * zq - wq * xq),
            1.0 - 2.0 * (xq * xq + yq * yq),
        ], dim=-1)

    def pca_quats_for(out_vox: torch.Tensor, fallback_quat: torch.Tensor) -> torch.Tensor:
        if out_vox.numel() == 0:
            return fallback_quat
        w_pca = params["opacity"].reshape(-1, 1).detach().clamp_min(1e-6).to(mean.dtype)
        denom_pca = torch.zeros(n_vox, 1, dtype=mean.dtype, device=mean.device)
        denom_pca.scatter_add_(0, inverse[:, None].expand(-1, 1), w_pca)
        center_pca = torch.zeros(n_vox, 3, dtype=mean.dtype, device=mean.device)
        center_pca.scatter_add_(0, inverse[:, None].expand(-1, 3), mean.detach() * w_pca)
        center_pca = center_pca / denom_pca.clamp_min(1e-6)
        d_pca = mean.detach() - center_pca[inverse]
        elem = torch.stack([
            d_pca[:, 0] * d_pca[:, 0],
            d_pca[:, 0] * d_pca[:, 1],
            d_pca[:, 0] * d_pca[:, 2],
            d_pca[:, 1] * d_pca[:, 1],
            d_pca[:, 1] * d_pca[:, 2],
            d_pca[:, 2] * d_pca[:, 2],
        ], dim=-1) * w_pca
        cov_elem = torch.zeros(n_vox, 6, dtype=mean.dtype, device=mean.device)
        cov_elem.scatter_add_(0, inverse[:, None].expand(-1, 6), elem)
        cov_elem = cov_elem / denom_pca.clamp_min(1e-6)
        c = cov_elem[out_vox].to(torch.float32)
        cov = torch.zeros(c.shape[0], 3, 3, dtype=torch.float32, device=mean.device)
        cov[:, 0, 0] = c[:, 0]
        cov[:, 0, 1] = cov[:, 1, 0] = c[:, 1]
        cov[:, 0, 2] = cov[:, 2, 0] = c[:, 2]
        cov[:, 1, 1] = c[:, 3]
        cov[:, 1, 2] = cov[:, 2, 1] = c[:, 4]
        cov[:, 2, 2] = c[:, 5]
        _, eigvec = torch.linalg.eigh(cov)
        normal = eigvec[:, :, 0].to(mean.dtype)
        valid = (c[:, 0] + c[:, 3] + c[:, 5]).to(mean.dtype) > 1e-12
        fallback = quat_local_z(fallback_quat).to(mean.dtype)
        normal = torch.where(valid[:, None], normal, fallback)
        return _quat_from_normals(normal).to(dtype=fallback_quat.dtype)

    color_chosen = choose_score_representative() if color_mode == "score_select" else chosen

    if mode == "average":
        keep_mask = (chosen < n) if soft_support else (support & (chosen < n))
        keep_vox = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        output_support = support[keep_vox]
        w_base = params["opacity"].reshape(-1, 1).detach().clamp_min(1e-6)
        w = w_base
        denom = torch.zeros(n_vox, 1, dtype=mean.dtype, device=mean.device)
        if average_dist_decay > 0:
            center_denom = torch.zeros(n_vox, 1, dtype=mean.dtype, device=mean.device)
            center_denom.scatter_add_(0, inverse[:, None].expand(-1, 1), w_base.to(mean.dtype))
            center = torch.zeros(n_vox, mean.shape[1], dtype=mean.dtype, device=mean.device)
            center.scatter_add_(
                0, inverse[:, None].expand(-1, mean.shape[1]), mean.detach() * w_base.to(mean.dtype)
            )
            center = center / center_denom.clamp_min(1e-6)
            dist2 = ((mean.detach() - center[inverse]) ** 2).sum(dim=-1, keepdim=True)
            norm = max(float(voxel_size) ** 2, 1e-12)
            robust = torch.exp(-float(average_dist_decay) * dist2 / norm)
            w = w_base * robust.to(dtype=w_base.dtype)
        denom.scatter_add_(0, inverse[:, None].expand(-1, 1), w.to(mean.dtype))

        def avg_tensor(v: torch.Tensor) -> torch.Tensor:
            if not is_per_splat(v):
                return v
            if not torch.is_floating_point(v):
                return v[chosen[keep_vox]]
            flat = v.reshape(n, -1)
            ww = w.to(dtype=flat.dtype)
            out = torch.zeros(n_vox, flat.shape[1], dtype=flat.dtype, device=flat.device)
            out.scatter_add_(0, inverse[:, None].expand(-1, flat.shape[1]), flat * ww)
            out = out / denom.to(dtype=flat.dtype).clamp_min(1e-6)
            return out[keep_vox].reshape((keep_vox.numel(),) + v.shape[1:])

        fused = {}
        for k, v in params.items():
            if k == "opacity":
                fused[k] = max_op[keep_vox, None].to(dtype=v.dtype)
                if soft_support:
                    fused[k] = fused[k] * support_decay[keep_vox, None].to(dtype=v.dtype)
            elif k == "rgb" and color_mode in {"select", "score_select", "score_soft"}:
                selected = v[color_chosen[keep_vox]]
                if color_mode == "score_soft":
                    selected = score_soft_tensor(v, keep_vox)
                if color_select_mix >= 1.0:
                    fused[k] = selected
                elif color_select_mix <= 0.0:
                    fused[k] = avg_tensor(v)
                else:
                    fused[k] = avg_tensor(v) * (1.0 - color_select_mix) + selected * color_select_mix
            elif k == "quat":
                q_avg = avg_tensor(v)
                fused[k] = F.normalize(q_avg, dim=-1)
            else:
                fused[k] = avg_tensor(v)
        keep_idx = chosen[keep_vox]
    elif max_per_voxel == 1:
        keep_vox = torch.nonzero((chosen < n) if soft_support else (support & (chosen < n)),
                                 as_tuple=False).squeeze(1)
        keep_idx = chosen[keep_vox]
        output_support = support[keep_vox]
    else:
        if representative_mode == "score" and score_key in params:
            rank_score = params[score_key].reshape(-1).detach().to(dtype=op.dtype, device=op.device)
            rank_score = torch.nan_to_num(
                rank_score, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf
            )
            remaining = rank_score.clone()
            selected = []
            for _ in range(max_per_voxel):
                max_score = torch.full((n_vox,), -torch.inf, dtype=remaining.dtype,
                                       device=remaining.device)
                max_score.scatter_reduce_(0, inverse, remaining, reduce="amax", include_self=True)
                candidates = torch.where(remaining >= max_score[inverse] - 1e-8, idx,
                                         torch.full_like(idx, n))
                chosen_k = torch.full((n_vox,), n, dtype=torch.long, device=mean.device)
                chosen_k.scatter_reduce_(0, inverse, candidates, reduce="amin", include_self=True)
                keep_vox_k = torch.nonzero(
                    (chosen_k < n) if soft_support else (support & (chosen_k < n)),
                    as_tuple=False,
                ).squeeze(1)
                if keep_vox_k.numel() == 0:
                    break
                keep_k = chosen_k[keep_vox_k]
                selected.append(keep_k)
                remaining[keep_k] = -torch.inf
            keep_idx = torch.cat(selected, 0) if selected else idx[:0]
        else:
            order = torch.argsort(key)
            inv_s = inverse[order]
            first = torch.ones(n, dtype=torch.bool, device=mean.device)
            first[1:] = inv_s[1:] != inv_s[:-1]
            group_start = torch.nonzero(first, as_tuple=False).squeeze(1)
            group_id = first.to(torch.long).cumsum(0) - 1
            rank = torch.arange(n, device=mean.device) - group_start[group_id]
            keep_sorted = (rank < max_per_voxel) & support[inv_s]
            keep_idx = order[keep_sorted]
        output_support = support[inverse[keep_idx]]

    if mode != "average":
        fused = {k: (v[keep_idx] if is_per_splat(v) else v) for k, v in params.items()}
        selected_rgb = None
        can_recolor = (
            "rgb" in fused
            and torch.is_floating_point(params["rgb"])
            and params["rgb"].shape[0] == n
        )
        if can_recolor and color_mode == "score_select":
            color_idx = color_chosen[inverse[keep_idx]]
            selected_rgb = params["rgb"][color_idx].reshape_as(fused["rgb"])
            fused["rgb"] = selected_rgb
        if can_recolor and color_mode == "score_soft":
            selected_rgb = score_soft_tensor(params["rgb"], inverse[keep_idx]).reshape_as(fused["rgb"])
            fused["rgb"] = selected_rgb
        if color_select_mix < 1.0 and max_per_voxel == 1 and can_recolor:
            rgb_flat = params["rgb"].reshape(n, -1)
            w_rgb = params["opacity"].reshape(-1, 1).detach().clamp_min(1e-6).to(rgb_flat.dtype)
            denom_rgb = torch.zeros(n_vox, 1, dtype=rgb_flat.dtype, device=rgb_flat.device)
            denom_rgb.scatter_add_(0, inverse[:, None].expand(-1, 1), w_rgb)
            avg_rgb = torch.zeros(n_vox, rgb_flat.shape[1], dtype=rgb_flat.dtype, device=rgb_flat.device)
            avg_rgb.scatter_add_(0, inverse[:, None].expand(-1, rgb_flat.shape[1]), rgb_flat * w_rgb)
            avg_rgb = avg_rgb / denom_rgb.clamp_min(1e-6)
            avg_rgb = avg_rgb[inverse[keep_idx]].reshape_as(fused["rgb"])
            if selected_rgb is None:
                selected_rgb = fused["rgb"]
            fused["rgb"] = avg_rgb * (1.0 - color_select_mix) + selected_rgb * color_select_mix
        if soft_support and "opacity" in fused:
            fused["opacity"] = fused["opacity"] * support_decay[inverse[keep_idx], None].to(
                dtype=fused["opacity"].dtype
            )
    if score_soft_opacity_mix > 0.0 and score_key in params and "opacity" in fused:
        out_vox_for_rows = keep_vox if mode == "average" else inverse[keep_idx]
        if out_vox_for_rows.numel() == fused["opacity"].shape[0]:
            soft_opacity = score_soft_tensor(params["opacity"], out_vox_for_rows).to(
                dtype=fused["opacity"].dtype, device=fused["opacity"].device
            )
            if soft_support:
                soft_opacity = soft_opacity * support_decay[out_vox_for_rows, None].to(
                    dtype=soft_opacity.dtype, device=soft_opacity.device
                )
            fused["opacity"] = (
                fused["opacity"] * (1.0 - score_soft_opacity_mix)
                + soft_opacity * score_soft_opacity_mix
            )
    if score_soft_geometry_mix > 0.0 and score_key in params and "mean" in fused:
        out_vox_for_rows = keep_vox if mode == "average" else inverse[keep_idx]
        if out_vox_for_rows.numel() == fused["mean"].shape[0]:
            for k in ("mean", "mean_anchor", "mean_offset", "depth", "scale", "scale_raw", "quat"):
                src = params.get(k)
                dst = fused.get(k)
                if (src is None or dst is None or not is_per_splat(src)
                        or not torch.is_floating_point(src)
                        or dst.shape[:1] != out_vox_for_rows.shape):
                    continue
                soft = score_soft_tensor(src, out_vox_for_rows).to(
                    dtype=dst.dtype, device=dst.device
                )
                mixed = dst * (1.0 - score_soft_geometry_mix) + soft * score_soft_geometry_mix
                if k == "quat":
                    mixed = F.normalize(mixed, dim=-1)
                fused[k] = mixed
    neighbor_dropped = 0
    neighbor_mean = 0.0
    if neighbor_support_min > 0 and keep_idx.numel() > 0 and "opacity" in fused:
        radius_n = max(int(neighbor_support_radius), 0)
        supported_vox = torch.nonzero(support & (chosen < n), as_tuple=False).squeeze(1)
        if supported_vox.numel() > 0:
            support_keys = key[chosen[supported_vox]].sort().values
            q_out = q[keep_idx]
            neighbor_counts = torch.zeros(q_out.shape[0], dtype=mean.dtype, device=mean.device)
            offsets = torch.stack(torch.meshgrid(
                torch.arange(-radius_n, radius_n + 1, device=mean.device, dtype=torch.int64),
                torch.arange(-radius_n, radius_n + 1, device=mean.device, dtype=torch.int64),
                torch.arange(-radius_n, radius_n + 1, device=mean.device, dtype=torch.int64),
                indexing="ij",
            ), dim=-1).reshape(-1, 3)
            for off in offsets:
                nq = q_out + off
                valid = ((nq >= 0) & (nq < dims[None])).all(dim=1)
                if not valid.any():
                    continue
                nk = nq[valid, 0] + dims[0] * (nq[valid, 1] + dims[1] * nq[valid, 2])
                pos = torch.searchsorted(support_keys, nk)
                pos_safe = pos.clamp_max(max(int(support_keys.numel()) - 1, 0))
                hit = (pos < support_keys.numel()) & (support_keys[pos_safe] == nk)
                neighbor_counts[valid] += hit.to(neighbor_counts.dtype)
            neighbor_mean = float(neighbor_counts.mean().item())
            missing = (float(neighbor_support_min) - neighbor_counts).clamp_min(0.0)
            if neighbor_opacity_decay > 0:
                decay = torch.exp(-float(neighbor_opacity_decay) * missing).unsqueeze(-1)
                fused["opacity"] = fused["opacity"] * decay.to(dtype=fused["opacity"].dtype)
                neighbor_dropped = int((missing > 0).sum().item())
            else:
                keep_neighbor = neighbor_counts >= float(neighbor_support_min)
                neighbor_dropped = int((~keep_neighbor).sum().item())
                fused = {
                    k: (v[keep_neighbor] if hasattr(v, "shape") and v.shape[:1] == keep_neighbor.shape else v)
                    for k, v in fused.items()
                }
                keep_idx = keep_idx[keep_neighbor]
                output_support = output_support[keep_neighbor]
        elif neighbor_opacity_decay > 0:
            fused["opacity"] = fused["opacity"] * math.exp(
                -float(neighbor_opacity_decay) * float(neighbor_support_min)
            )
            neighbor_dropped = int(keep_idx.numel())
        else:
            fused = {
                k: (v[:0] if hasattr(v, "shape") and v.shape[:1] == keep_idx.shape else v)
                for k, v in fused.items()
            }
            neighbor_dropped = int(keep_idx.numel())
            keep_idx = keep_idx[:0]
            output_support = output_support[:0]
    propagation_dropped = 0
    propagation_reachable_frac = 1.0
    propagation_steps = max(int(support_propagation_steps), 0)
    if (propagation_steps > 0 and keep_idx.numel() > 0 and "opacity" in fused
            and output_support.shape[0] == keep_idx.shape[0]):
        reachable = output_support.to(device=mean.device, dtype=torch.bool).clone()
        if reachable.any() and not reachable.all():
            q_out = q[keep_idx]
            key_out = key[keep_idx]
            radius_p = max(int(support_propagation_radius), 0)
            offsets = torch.stack(torch.meshgrid(
                torch.arange(-radius_p, radius_p + 1, device=mean.device, dtype=torch.int64),
                torch.arange(-radius_p, radius_p + 1, device=mean.device, dtype=torch.int64),
                torch.arange(-radius_p, radius_p + 1, device=mean.device, dtype=torch.int64),
                indexing="ij",
            ), dim=-1).reshape(-1, 3)
            for _ in range(propagation_steps):
                reach_keys = key_out[reachable].sort().values
                if reach_keys.numel() == 0:
                    break
                next_reachable = reachable.clone()
                for off in offsets:
                    nq = q_out + off
                    valid = ((nq >= 0) & (nq < dims[None])).all(dim=1)
                    if not valid.any():
                        continue
                    nk = nq[valid, 0] + dims[0] * (nq[valid, 1] + dims[1] * nq[valid, 2])
                    pos = torch.searchsorted(reach_keys, nk)
                    pos_safe = pos.clamp_max(max(int(reach_keys.numel()) - 1, 0))
                    hit = (pos < reach_keys.numel()) & (reach_keys[pos_safe] == nk)
                    if hit.any():
                        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                        next_reachable[valid_idx[hit]] = True
                if bool(torch.equal(next_reachable, reachable)):
                    break
                reachable = next_reachable
            unreachable = ~reachable
            propagation_dropped = int(unreachable.sum().item())
            propagation_reachable_frac = float(reachable.to(torch.float32).mean().item())
            if propagation_dropped > 0:
                decay_p = max(float(support_propagation_opacity_decay), 0.0)
                if decay_p > 0:
                    fused["opacity"] = fused["opacity"] * torch.where(
                        reachable[:, None].to(device=fused["opacity"].device),
                        fused["opacity"].new_ones((reachable.shape[0], 1)),
                        fused["opacity"].new_full((reachable.shape[0], 1), math.exp(-decay_p)),
                    )
                else:
                    keep_prop = reachable.to(device=fused["opacity"].device)
                    fused = {
                        k: (v[keep_prop] if hasattr(v, "shape") and v.shape[:1] == keep_prop.shape else v)
                        for k, v in fused.items()
                    }
                    keep_idx = keep_idx[keep_prop]
                    output_support = output_support[keep_prop]
    if pca_quat and "quat" in fused and keep_idx.numel() > 0:
        fused["quat"] = pca_quats_for(inverse[keep_idx], fused["quat"])
    if (mode != "average" and coverage_opacity_mult > 0.0 and max_per_voxel == 1
            and keep_idx.numel() > 0 and "mean" in fused and "opacity" in fused):
        keep_vox_for_coverage = inverse[keep_idx]
        w_cov = params["opacity"].reshape(-1, 1).detach().clamp_min(1e-6)
        denom_cov = torch.zeros(n_vox, 1, dtype=mean.dtype, device=mean.device)
        denom_cov.scatter_add_(0, inverse[:, None].expand(-1, 1), w_cov.to(mean.dtype))

        def avg_for_coverage(v: torch.Tensor) -> torch.Tensor:
            if not torch.is_floating_point(v) or v.shape[0] != n:
                return v[keep_idx]
            flat = v.reshape(n, -1)
            ww = w_cov.to(dtype=flat.dtype)
            out = torch.zeros(n_vox, flat.shape[1], dtype=flat.dtype, device=flat.device)
            out.scatter_add_(0, inverse[:, None].expand(-1, flat.shape[1]), flat * ww)
            out = out / denom_cov.to(dtype=flat.dtype).clamp_min(1e-6)
            return out[keep_vox_for_coverage].reshape((keep_idx.numel(),) + v.shape[1:])

        cov = {}
        for k, v in params.items():
            if not hasattr(v, "shape") or v.shape[:1] != (n,):
                continue
            if k == "opacity":
                cov[k] = fused[k] * coverage_opacity_mult
            elif k == "quat":
                cov[k] = F.normalize(avg_for_coverage(v), dim=-1)
            else:
                cov[k] = avg_for_coverage(v)
        base_n = fused["mean"].shape[0]
        for k, v in list(fused.items()):
            if k in cov and hasattr(v, "shape") and v.shape[:1] == (base_n,):
                fused[k] = torch.cat([v, cov[k].to(device=v.device, dtype=v.dtype)], dim=0)
        if fused["mean"].shape[0] > base_n:
            coverage_rows = torch.zeros(
                fused["mean"].shape[0], dtype=torch.bool, device=fused["mean"].device
            )
            coverage_rows[base_n:] = True
            if output_support.shape[0] == base_n:
                output_support = torch.cat([output_support, output_support.clone()], dim=0)
    if scale_floor is not None and scale_floor > 0 and "scale" in fused:
        floor = torch.as_tensor(scale_floor, dtype=fused["scale"].dtype,
                                device=fused["scale"].device)
        if fused["scale"].shape[-1] >= 3 and float(scale_floor_z_mult) != 1.0:
            floor = floor * fused["scale"].new_tensor([
                1.0, 1.0, max(float(scale_floor_z_mult), 0.0)
            ])
        detail_min = min(max(float(scale_floor_detail_min), 0.0), 1.0)
        if (scale_floor_detail_key and scale_floor_detail_key in fused
                and detail_min < 1.0):
            detail = fused[scale_floor_detail_key].reshape(fused["scale"].shape[0], -1)
            detail = detail[:, :1].clamp(0.0, 1.0).to(dtype=fused["scale"].dtype)
            mult = 1.0 - (1.0 - detail_min) * detail
            floor = floor * mult
        low_mult = min(max(float(low_support_scale_floor_mult), 0.0), 1.0)
        if low_mult < 1.0 and output_support.shape[0] == fused["scale"].shape[0]:
            support_mult = torch.where(
                output_support[:, None].to(device=fused["scale"].device),
                fused["scale"].new_ones((fused["scale"].shape[0], 1)),
                fused["scale"].new_full((fused["scale"].shape[0], 1), low_mult),
            )
            floor = floor * support_mult
        fused["scale"] = torch.maximum(fused["scale"], floor)
    if (coverage_rows is not None and "scale" in fused and coverage_scale_mult != 1.0
            and coverage_rows.shape[0] == fused["scale"].shape[0]):
        fused["scale"] = fused["scale"].clone()
        fused["scale"][coverage_rows] = fused["scale"][coverage_rows] * coverage_scale_mult
    stats = {
        "input": int(n),
        "output": int(fused["mean"].shape[0] if "mean" in fused else keep_idx.numel()),
        "voxels": int(n_vox),
        "supported_voxels": int(support.sum().item()),
        "dropped_low_support": int((~support).sum().item()),
        "representative_mode": representative_mode,
        "neighbor_dropped": neighbor_dropped,
        "neighbor_mean": neighbor_mean,
        "propagation_dropped": propagation_dropped,
        "propagation_reachable_frac": propagation_reachable_frac,
        "coverage_added": int(coverage_rows.sum().item()) if coverage_rows is not None else 0,
        "score_soft_geometry_mix": score_soft_geometry_mix,
    }
    return fused, stats


def _quat_from_normals(normals: torch.Tensor) -> torch.Tensor:
    """World normals ``(N,3)`` -> quaternions whose local z follows the normal."""
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


def rgbd_target_view_surface(
    frames: torch.Tensor,
    masks: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    c2w: torch.Tensor,
    target_K: torch.Tensor,
    target_w2c: torch.Tensor,
    target_c2w: torch.Tensor,
    width: int,
    height: int,
    radius: float,
    scale_frac: float,
    normal_scale_frac: float,
    opacity: float = 0.95,
    depth_tol: float = 0.02,
    mask_thresh: float = 0.5,
) -> tuple[dict, dict]:
    """Z-buffer source RGBD views into one target-camera Gaussian shell.

    This is a feed-forward visibility diagnostic: source RGBD pixels are
    unprojected, projected into the target camera, front-z-buffered per target
    pixel, then lifted back to target-camera ray-aligned Gaussians. It performs
    no per-object optimization and emits a view-conditioned 3DGS surface.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (V,H,W,3)")
    if masks.ndim != 4 or masks.shape[-1] != 1:
        raise ValueError("masks must have shape (V,H,W,1)")
    if depths.ndim != 3:
        raise ValueError("depths must have shape (V,H,W)")
    n_src, src_h, src_w = depths.shape
    if frames.shape[:3] != (n_src, src_h, src_w) or masks.shape[:3] != (n_src, src_h, src_w):
        raise ValueError("frames, masks, and depths must agree on view/height/width")
    device, dtype = frames.device, frames.dtype
    n_pix = int(width) * int(height)
    inf = torch.tensor(1e10, device=device, dtype=dtype)
    best_z = torch.full((n_pix,), inf, device=device, dtype=dtype)

    def candidates(src_i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_src = depths[src_i].to(device=device, dtype=dtype)
        valid_src = (z_src < 1e5) & (masks[src_i, ..., 0].to(device=device) > float(mask_thresh))
        if not valid_src.any():
            empty_i = torch.empty(0, device=device, dtype=torch.long)
            empty_f = torch.empty(0, device=device, dtype=dtype)
            return empty_i, empty_f, empty_f, torch.empty(0, 3, device=device, dtype=dtype)
        t_src = zdepth_to_raydist(z_src, K[src_i].to(device=device, dtype=dtype))
        dirs = ray_dirs_world(K[src_i].to(device=device, dtype=dtype),
                              c2w[src_i].to(device=device, dtype=dtype),
                              src_h, src_w).to(device=device, dtype=dtype)
        origin = c2w[src_i, :3, 3].to(device=device, dtype=dtype)
        pts = origin[None] + t_src.reshape(-1, 1) * dirs
        keep_src = valid_src.reshape(-1)
        pts = pts[keep_src]
        rgb = frames[src_i].reshape(-1, 3).to(device=device, dtype=dtype)[keep_src]
        cam = pts @ target_w2c[:3, :3].to(device=device, dtype=dtype).T + target_w2c[:3, 3].to(
            device=device, dtype=dtype
        )
        z = cam[:, 2]
        fx, fy = target_K[0, 0].to(device=device, dtype=dtype), target_K[1, 1].to(device=device, dtype=dtype)
        cx, cy = target_K[0, 2].to(device=device, dtype=dtype), target_K[1, 2].to(device=device, dtype=dtype)
        u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
        v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
        inb = (z > 1e-6) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
        if not inb.any():
            empty_i = torch.empty(0, device=device, dtype=torch.long)
            empty_f = torch.empty(0, device=device, dtype=dtype)
            return empty_i, empty_f, empty_f, torch.empty(0, 3, device=device, dtype=dtype)
        ui = u[inb].round().long().clamp(0, width - 1)
        vi = v[inb].round().long().clamp(0, height - 1)
        pix = vi * int(width) + ui
        return pix, z[inb], torch.ones_like(z[inb]), rgb[inb]

    for src_i in range(n_src):
        pix, z, _, _ = candidates(src_i)
        if pix.numel() > 0:
            best_z.scatter_reduce_(0, pix, z, reduce="amin", include_self=True)

    sum_w = torch.zeros((n_pix,), device=device, dtype=dtype)
    sum_z = torch.zeros((n_pix,), device=device, dtype=dtype)
    sum_rgb = torch.zeros((n_pix, 3), device=device, dtype=dtype)
    tol = max(float(depth_tol), 1e-6)
    for src_i in range(n_src):
        pix, z, _, rgb = candidates(src_i)
        if pix.numel() == 0:
            continue
        dz = (z - best_z[pix]).clamp_min(0.0)
        front = dz <= tol
        if not front.any():
            continue
        pix_f = pix[front]
        z_f = z[front]
        w_f = torch.exp(-dz[front] / tol)
        rgb_f = rgb[front]
        sum_w.index_add_(0, pix_f, w_f)
        sum_z.index_add_(0, pix_f, w_f * z_f)
        sum_rgb.index_add_(0, pix_f, w_f[:, None] * rgb_f)

    valid = sum_w > 1e-8
    if not valid.any():
        empty = frames.new_zeros(0)
        return {
            "mean": frames.new_zeros(0, 3),
            "quat": frames.new_zeros(0, 4),
            "scale": frames.new_zeros(0, 3),
            "opacity": frames.new_zeros(0, 1),
            "rgb": frames.new_zeros(0, 3),
            "depth": frames.new_zeros(0, 1),
        }, {"input": 0, "output": 0, "valid_pixels": 0}

    z_map = (sum_z / sum_w.clamp_min(1e-8)).reshape(height, width)
    rgb_map = (sum_rgb / sum_w[:, None].clamp_min(1e-8)).clamp(0.0, 1.0)
    t_map = zdepth_to_raydist(z_map, target_K.to(device=device, dtype=dtype)).reshape(-1)
    dirs_t = ray_dirs_world(target_K.to(device=device, dtype=dtype),
                            target_c2w.to(device=device, dtype=dtype),
                            height, width).to(device=device, dtype=dtype)
    origin_t = target_c2w[:3, 3].to(device=device, dtype=dtype)
    mean_all = origin_t[None] + t_map[:, None] * dirs_t
    mean = mean_all[valid]
    rgb = rgb_map[valid]
    view_normal = F.normalize(origin_t[None] - mean, dim=-1)
    quat = _quat_from_normals(view_normal)
    tang = max(float(scale_frac), 1e-8) * float(radius)
    norm = max(float(normal_scale_frac), 1e-8) * float(radius)
    scale = frames.new_tensor([tang, tang, norm]).view(1, 3).expand(mean.shape[0], 3).clone()
    opacity_t = frames.new_full((mean.shape[0], 1), float(opacity))
    depth_t = t_map[valid, None]
    return {
        "mean": mean,
        "quat": quat,
        "scale": scale,
        "opacity": opacity_t,
        "rgb": rgb,
        "depth": depth_t,
    }, {
        "input": int(n_src * src_h * src_w),
        "output": int(mean.shape[0]),
        "valid_pixels": int(valid.sum().item()),
    }


def rgbd_target_view_surface_splat(
    frames: torch.Tensor,
    masks: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    c2w: torch.Tensor,
    target_K: torch.Tensor,
    target_w2c: torch.Tensor,
    target_c2w: torch.Tensor,
    width: int,
    height: int,
    radius: float,
    scale_frac: float,
    normal_scale_frac: float,
    opacity: float = 0.95,
    depth_tol: float = 0.02,
    mask_thresh: float = 0.5,
    view_weight_temp: float = 0.0,
    min_support: int = 1,
    support_depth_tol: float | None = None,
) -> tuple[dict, dict]:
    """Bilinear forward-splat source RGBD into one target-camera Gaussian shell.

    Compared with ``rgbd_target_view_surface``, this distributes each projected
    source pixel to its four neighboring target pixels before front-depth
    compositing. The output is still feed-forward and view-conditioned, but it
    avoids the harsh aliasing from integer target-pixel z-buffering.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (V,H,W,3)")
    if masks.ndim != 4 or masks.shape[-1] != 1:
        raise ValueError("masks must have shape (V,H,W,1)")
    if depths.ndim != 3:
        raise ValueError("depths must have shape (V,H,W)")
    n_src, src_h, src_w = depths.shape
    if frames.shape[:3] != (n_src, src_h, src_w) or masks.shape[:3] != (n_src, src_h, src_w):
        raise ValueError("frames, masks, and depths must agree on view/height/width")
    device, dtype = frames.device, frames.dtype
    n_pix = int(width) * int(height)
    inf = torch.tensor(1e10, device=device, dtype=dtype)
    best_z = torch.full((n_pix,), inf, device=device, dtype=dtype)

    view_weights = torch.ones(n_src, device=device, dtype=dtype)
    if view_weight_temp > 0:
        src_centers = c2w[:, :3, 3].to(device=device, dtype=dtype)
        tgt_center = target_c2w[:3, 3].to(device=device, dtype=dtype)
        dist = torch.linalg.norm(src_centers - tgt_center[None], dim=-1)
        view_weights = torch.softmax(-dist / max(float(view_weight_temp), 1e-6), dim=0)
        view_weights = view_weights * float(n_src)
    min_support = max(int(min_support), 1)
    support_tol = max(float(support_depth_tol if support_depth_tol is not None else depth_tol), 1e-6)

    def support_for_points(src_i: int, pts: torch.Tensor) -> torch.Tensor:
        if min_support <= 1 or pts.numel() == 0:
            return torch.ones(pts.shape[0], device=device, dtype=dtype)
        support = torch.ones(pts.shape[0], device=device, dtype=dtype)
        for ref_i in range(n_src):
            if ref_i == src_i:
                continue
            cam = pts @ w2c[ref_i, :3, :3].to(device=device, dtype=dtype).T + w2c[ref_i, :3, 3].to(
                device=device, dtype=dtype
            )
            z = cam[:, 2]
            fx = K[ref_i, 0, 0].to(device=device, dtype=dtype)
            fy = K[ref_i, 1, 1].to(device=device, dtype=dtype)
            cx = K[ref_i, 0, 2].to(device=device, dtype=dtype)
            cy = K[ref_i, 1, 2].to(device=device, dtype=dtype)
            u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
            v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
            inb = (z > 1e-6) & (u >= 0) & (u <= src_w - 1) & (v >= 0) & (v <= src_h - 1)
            grid_x = (u / max(src_w - 1, 1)) * 2.0 - 1.0
            grid_y = (v / max(src_h - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], -1).view(1, -1, 1, 2)
            depth_ref = depths[ref_i:ref_i + 1, None].to(device=device, dtype=dtype)
            mask_ref = masks[ref_i:ref_i + 1].permute(0, 3, 1, 2).to(device=device, dtype=dtype)
            samp_d = F.grid_sample(
                depth_ref, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).view(-1)
            samp_m = F.grid_sample(
                mask_ref, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).view(-1)
            match = inb & (samp_m > float(mask_thresh)) & (samp_d < 1e5) & (
                (z - samp_d).abs() <= support_tol
            )
            support = support + match.to(dtype)
        return support

    def candidates(src_i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_src = depths[src_i].to(device=device, dtype=dtype)
        mask_src = masks[src_i, ..., 0].to(device=device, dtype=dtype)
        valid_src = (z_src < 1e5) & (mask_src > float(mask_thresh))
        if not valid_src.any():
            empty_i = torch.empty(0, device=device, dtype=torch.long)
            empty_f = torch.empty(0, device=device, dtype=dtype)
            return empty_i, empty_f, empty_f, torch.empty(0, 3, device=device, dtype=dtype)
        t_src = zdepth_to_raydist(z_src, K[src_i].to(device=device, dtype=dtype))
        dirs = ray_dirs_world(
            K[src_i].to(device=device, dtype=dtype),
            c2w[src_i].to(device=device, dtype=dtype),
            src_h, src_w,
        ).to(device=device, dtype=dtype)
        origin = c2w[src_i, :3, 3].to(device=device, dtype=dtype)
        pts = origin[None] + t_src.reshape(-1, 1) * dirs
        keep_src = valid_src.reshape(-1)
        pts = pts[keep_src]
        rgb = frames[src_i].reshape(-1, 3).to(device=device, dtype=dtype)[keep_src]
        src_alpha = mask_src.reshape(-1)[keep_src].clamp(0.0, 1.0)
        support = support_for_points(src_i, pts)
        keep_support = support >= float(min_support)
        if not keep_support.any():
            empty_i = torch.empty(0, device=device, dtype=torch.long)
            empty_f = torch.empty(0, device=device, dtype=dtype)
            return empty_i, empty_f, empty_f, torch.empty(0, 3, device=device, dtype=dtype)
        pts = pts[keep_support]
        rgb = rgb[keep_support]
        src_alpha = src_alpha[keep_support]
        support = support[keep_support]
        cam = pts @ target_w2c[:3, :3].to(device=device, dtype=dtype).T + target_w2c[:3, 3].to(
            device=device, dtype=dtype
        )
        z = cam[:, 2]
        fx = target_K[0, 0].to(device=device, dtype=dtype)
        fy = target_K[1, 1].to(device=device, dtype=dtype)
        cx = target_K[0, 2].to(device=device, dtype=dtype)
        cy = target_K[1, 2].to(device=device, dtype=dtype)
        u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
        v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
        inb = (z > 1e-6) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
        if not inb.any():
            empty_i = torch.empty(0, device=device, dtype=torch.long)
            empty_f = torch.empty(0, device=device, dtype=dtype)
            return empty_i, empty_f, empty_f, torch.empty(0, 3, device=device, dtype=dtype)
        u = u[inb]
        v = v[inb]
        z = z[inb]
        rgb = rgb[inb]
        src_alpha = src_alpha[inb]
        support = support[inb]

        u0 = torch.floor(u).long().clamp(0, width - 1)
        v0 = torch.floor(v).long().clamp(0, height - 1)
        u1 = (u0 + 1).clamp(0, width - 1)
        v1 = (v0 + 1).clamp(0, height - 1)
        du = (u - u0.to(dtype)).clamp(0.0, 1.0)
        dv = (v - v0.to(dtype)).clamp(0.0, 1.0)
        pix = torch.stack([
            v0 * int(width) + u0,
            v0 * int(width) + u1,
            v1 * int(width) + u0,
            v1 * int(width) + u1,
        ], dim=1).reshape(-1)
        bw = torch.stack([
            (1.0 - du) * (1.0 - dv),
            du * (1.0 - dv),
            (1.0 - du) * dv,
            du * dv,
        ], dim=1).reshape(-1)
        z4 = z[:, None].expand(-1, 4).reshape(-1)
        rgb4 = rgb[:, None, :].expand(-1, 4, -1).reshape(-1, 3)
        support_weight = (support / float(n_src)).clamp(0.0, 1.0)
        weight = bw * src_alpha[:, None].expand(-1, 4).reshape(-1)
        weight = weight * support_weight[:, None].expand(-1, 4).reshape(-1)
        weight = weight * view_weights[src_i]
        keep = weight > 1e-6
        return pix[keep], z4[keep], weight[keep], rgb4[keep]

    for src_i in range(n_src):
        pix, z, _, _ = candidates(src_i)
        if pix.numel() > 0:
            best_z.scatter_reduce_(0, pix, z, reduce="amin", include_self=True)

    sum_w = torch.zeros((n_pix,), device=device, dtype=dtype)
    sum_z = torch.zeros((n_pix,), device=device, dtype=dtype)
    sum_rgb = torch.zeros((n_pix, 3), device=device, dtype=dtype)
    tol = max(float(depth_tol), 1e-6)
    for src_i in range(n_src):
        pix, z, weight, rgb = candidates(src_i)
        if pix.numel() == 0:
            continue
        dz = (z - best_z[pix]).clamp_min(0.0)
        front = dz <= tol
        if not front.any():
            continue
        pix_f = pix[front]
        z_f = z[front]
        w_f = weight[front] * torch.exp(-dz[front] / tol)
        rgb_f = rgb[front]
        sum_w.index_add_(0, pix_f, w_f)
        sum_z.index_add_(0, pix_f, w_f * z_f)
        sum_rgb.index_add_(0, pix_f, w_f[:, None] * rgb_f)

    valid = sum_w > 1e-8
    if not valid.any():
        return {
            "mean": frames.new_zeros(0, 3),
            "quat": frames.new_zeros(0, 4),
            "scale": frames.new_zeros(0, 3),
            "opacity": frames.new_zeros(0, 1),
            "rgb": frames.new_zeros(0, 3),
            "depth": frames.new_zeros(0, 1),
        }, {"input": 0, "output": 0, "valid_pixels": 0}

    z_map = (sum_z / sum_w.clamp_min(1e-8)).reshape(height, width)
    rgb_map = (sum_rgb / sum_w[:, None].clamp_min(1e-8)).clamp(0.0, 1.0)
    t_map = zdepth_to_raydist(z_map, target_K.to(device=device, dtype=dtype)).reshape(-1)
    dirs_t = ray_dirs_world(
        target_K.to(device=device, dtype=dtype),
        target_c2w.to(device=device, dtype=dtype),
        height, width,
    ).to(device=device, dtype=dtype)
    origin_t = target_c2w[:3, 3].to(device=device, dtype=dtype)
    mean_all = origin_t[None] + t_map[:, None] * dirs_t
    mean = mean_all[valid]
    rgb = rgb_map[valid]
    view_normal = F.normalize(origin_t[None] - mean, dim=-1)
    quat = _quat_from_normals(view_normal)
    tang = max(float(scale_frac), 1e-8) * float(radius)
    norm = max(float(normal_scale_frac), 1e-8) * float(radius)
    scale = frames.new_tensor([tang, tang, norm]).view(1, 3).expand(mean.shape[0], 3).clone()
    opacity_t = frames.new_full((mean.shape[0], 1), float(opacity))
    depth_t = t_map[valid, None]
    return {
        "mean": mean,
        "quat": quat,
        "scale": scale,
        "opacity": opacity_t,
        "rgb": rgb,
        "depth": depth_t,
    }, {
        "input": int(n_src * src_h * src_w),
        "output": int(mean.shape[0]),
        "valid_pixels": int(valid.sum().item()),
    }


def _sample_map(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    n = grid.shape[0]
    return F.grid_sample(
        x, grid.view(1, n, 1, 2), mode="bilinear", padding_mode="zeros",
        align_corners=True,
    ).view(x.shape[1], n).T


def _sh_bases(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """Spherical-harmonic bases matching gsplat's ordering through degree 2."""
    if degree < 0 or degree > 2:
        raise ValueError("feed-forward SH color fitting supports degree 0, 1, or 2")
    dirs = F.normalize(dirs, dim=-1)
    x, y, z = dirs.unbind(-1)
    out = [torch.full_like(x, 0.2820947917738781)]
    if degree >= 1:
        c1 = -0.48860251190292
        out.extend([c1 * y, -c1 * z, c1 * x])
    if degree >= 2:
        z2 = z * z
        f_c1 = x * x - y * y
        f_s1 = 2 * x * y
        f_tmp_b = -1.092548430592079 * z
        out.extend([
            0.5462742152960395 * f_s1,
            f_tmp_b * y,
            0.9461746957575601 * z2 - 0.3153915652525201,
            f_tmp_b * x,
            0.5462742152960395 * f_c1,
        ])
    return torch.stack(out, dim=-1)


def rgbd_fit_sh_colors(
    params: dict,
    frames: torch.Tensor,
    masks: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    *,
    degree: int = 1,
    depth_tol: float,
    ridge: float = 1e-3,
    min_obs: int = 2,
    mix: float = 1.0,
    chunk: int = 32768,
) -> tuple[dict, dict]:
    """Fit static SH color coefficients from source RGBD observations.

    This is a closed-form feed-forward color fusion step, not per-object
    gradient optimization. The fitted coefficients let one static Gaussian carry
    different source-view colors without averaging them into a muddy constant.
    """
    if degree <= 0:
        return params, {"sh_degree": 0, "fitted": 0}
    if depth_tol <= 0:
        raise ValueError("depth_tol must be positive")
    mix = min(max(float(mix), 0.0), 1.0)
    n_bases = (degree + 1) ** 2
    device, dtype = params["mean"].device, params["mean"].dtype
    frames = frames.to(device=device, dtype=dtype)
    masks = masks.to(device=device, dtype=dtype)
    depths = depths.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    w2c = w2c.to(device=device, dtype=dtype)
    means_all = params["mean"]
    c0 = 0.2820947917738781
    coeffs = means_all.new_zeros(means_all.shape[0], n_bases, 3)
    coeffs[:, 0, :] = (params["rgb"].to(device=device, dtype=dtype) - 0.5) / c0
    frame_imgs = frames.permute(0, 3, 1, 2)
    depth_imgs = depths[:, None]
    mask_imgs = masks.permute(0, 3, 1, 2)
    h, w = frames.shape[1], frames.shape[2]
    eye = torch.eye(n_bases, device=device, dtype=dtype)
    fitted_total = 0
    obs_total = 0.0
    for start in range(0, means_all.shape[0], chunk):
        p = means_all[start:start + chunk]
        n = p.shape[0]
        yty = eye[None].expand(n, -1, -1).clone() * float(ridge)
        ytb = p.new_zeros(n, n_bases, 3)
        obs = torch.zeros(n, device=device, dtype=dtype)
        for i in range(frames.shape[0]):
            z, grid, inb = _project_points(p, w2c[i], K[i], h, w)
            samp_d = _sample_map(depth_imgs[i:i + 1], grid)[:, 0]
            samp_m = _sample_map(mask_imgs[i:i + 1], grid)[:, 0]
            samp_rgb = _sample_map(frame_imgs[i:i + 1], grid)
            sdf = (samp_d - z).abs()
            valid = inb & (samp_m > 0.5) & (samp_d < 1e5) & (sdf <= float(depth_tol))
            if not valid.any():
                continue
            c2w_i = torch.linalg.inv(w2c[i])
            dirs = p - c2w_i[:3, 3]
            bases = _sh_bases(dirs, degree)
            ww = torch.exp(-sdf / float(depth_tol)) * valid.to(dtype)
            centered = samp_rgb - 0.5
            yty = yty + ww[:, None, None] * bases[:, :, None] * bases[:, None, :]
            ytb = ytb + ww[:, None, None] * bases[:, :, None] * centered[:, None, :]
            obs = obs + valid.to(dtype)
        enough = obs >= float(min_obs)
        if enough.any():
            sol = torch.linalg.solve(yty[enough].float(), ytb[enough].float()).to(dtype)
            coeff_slice = coeffs[start:start + n]
            coeff_slice[enough] = coeff_slice[enough] * (1.0 - mix) + sol * mix
            fitted_total += int(enough.sum().item())
        obs_total += float(obs.sum().item())
    out = dict(params)
    out["rgb_sh"] = coeffs
    stats = {
        "sh_degree": int(degree),
        "fitted": int(fitted_total),
        "mix": float(mix),
        "mean_obs": obs_total / max(int(means_all.shape[0]), 1),
    }
    return out, stats


def _rgbd_bounds(depths: torch.Tensor, masks: torch.Tensor, K: torch.Tensor,
                 w2c: torch.Tensor, max_samples: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    v, h, w = depths.shape
    pts_out = []
    per_view = max(max_samples // max(v, 1), 64)
    vv, uu = torch.meshgrid(
        torch.arange(h, device=depths.device, dtype=depths.dtype),
        torch.arange(w, device=depths.device, dtype=depths.dtype),
        indexing="ij",
    )
    for i in range(v):
        valid = (depths[i] < 1e5) & (masks[i, ..., 0] > 0.5)
        idx = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        if idx.numel() > per_view:
            take = torch.linspace(0, idx.numel() - 1, per_view, device=idx.device).long()
            idx = idx[take]
        z = depths[i].reshape(-1)
        x = (uu.reshape(-1) - K[i, 0, 2]) / K[i, 0, 0].clamp_min(1e-6) * z
        y = (vv.reshape(-1) - K[i, 1, 2]) / K[i, 1, 1].clamp_min(1e-6) * z
        cam = torch.stack([x, y, z], dim=-1)
        c2w_cv = torch.linalg.inv(w2c[i])
        pts = cam @ c2w_cv[:3, :3].T + c2w_cv[:3, 3]
        pts_out.append(pts[idx])
    if not pts_out:
        return None
    pts_all = torch.cat(pts_out, dim=0)
    return pts_all.amin(dim=0), pts_all.amax(dim=0)


def rgbd_tsdf_fuse(
    frames: torch.Tensor,
    masks: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    c2w: torch.Tensor,
    *,
    voxel_size: float,
    trunc_mult: float = 4.0,
    min_weight: float = 2.0,
    surface_thresh: float = 0.75,
    max_voxels: int = 1_500_000,
    max_points: int = 250_000,
    bounds_pad_voxels: float = 3.0,
    scale_mult: float = 0.7,
    normal_scale_mult: float = 0.25,
    opacity: float = 0.95,
    color_mode: str = "select",
    surface_mode: str = "centers",
    chunk: int = 262_144,
) -> tuple[dict, dict]:
    """Fuse RGBD views into a single static 3DGS surface by projective TSDF.

    This is a deterministic feed-forward diagnostic. It does not optimize the
    object; it converts the conditioning RGBD clip into one surface-like Gaussian
    set so we can test whether shell concatenation is the artifact source.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if color_mode not in {"average", "select"}:
        raise ValueError("color_mode must be 'average' or 'select'")
    if surface_mode not in {"centers", "edges"}:
        raise ValueError("surface_mode must be 'centers' or 'edges'")
    if depths is None:
        raise ValueError("rgbd_tsdf_fuse requires depth maps")
    v, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    device, dtype = frames.device, frames.dtype
    depths = depths.to(device=device, dtype=dtype)
    masks = masks.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    w2c = w2c.to(device=device, dtype=dtype)
    c2w = c2w.to(device=device, dtype=dtype)
    bounds = _rgbd_bounds(depths, masks, K, w2c, max_samples=200_000)
    if bounds is None:
        empty = frames.new_zeros(0)
        return {
            "mean": frames.new_zeros(0, 3),
            "quat": frames.new_zeros(0, 4),
            "scale": frames.new_zeros(0, 3),
            "opacity": frames.new_zeros(0, 1),
            "rgb": frames.new_zeros(0, 3),
            "depth": empty[:, None],
            "scale_raw": frames.new_zeros(0, 3),
            "mean_anchor": frames.new_zeros(0, 3),
            "mean_offset": frames.new_zeros(0, 3),
        }, {"output": 0, "voxels": 0, "voxel_size": float(voxel_size)}

    mn, mx = bounds
    pad = max(float(voxel_size) * bounds_pad_voxels, 1e-4)
    mn = mn - pad
    mx = mx + pad
    dims = torch.ceil((mx - mn) / float(voxel_size)).to(torch.long) + 1
    dims = torch.clamp(dims, min=2)
    n_vox = int(dims.prod().item())
    if n_vox > max_voxels:
        factor = (n_vox / float(max_voxels)) ** (1.0 / 3.0)
        voxel_size = float(voxel_size) * factor
        dims = torch.ceil((mx - mn) / float(voxel_size)).to(torch.long) + 1
        dims = torch.clamp(dims, min=2)
        n_vox = int(dims.prod().item())
    nx, ny, nz = [int(x) for x in dims.tolist()]
    xs = mn[0] + torch.arange(nx, device=device, dtype=dtype) * float(voxel_size)
    ys = mn[1] + torch.arange(ny, device=device, dtype=dtype) * float(voxel_size)
    zs = mn[2] + torch.arange(nz, device=device, dtype=dtype) * float(voxel_size)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    depth_imgs = depths[:, None]
    mask_imgs = masks.permute(0, 3, 1, 2)
    trunc = max(float(voxel_size) * trunc_mult, 1e-6)
    tsdf_sum = torch.zeros(pts.shape[0], device=device, dtype=dtype)
    weight_sum = torch.zeros_like(tsdf_sum)
    for start in range(0, pts.shape[0], chunk):
        p = pts[start:start + chunk]
        ts = torch.zeros(p.shape[0], device=device, dtype=dtype)
        ws = torch.zeros_like(ts)
        for i in range(v):
            z, grid, inb = _project_points(p, w2c[i], K[i], h, w)
            samp_d = _sample_map(depth_imgs[i:i + 1], grid)[:, 0]
            samp_m = _sample_map(mask_imgs[i:i + 1], grid)[:, 0]
            valid = inb & (samp_m > 0.5) & (samp_d < 1e5)
            sdf = (samp_d - z).clamp(-trunc, trunc) / trunc
            ww = valid.to(dtype)
            ts = ts + sdf * ww
            ws = ws + ww
        tsdf_sum[start:start + p.shape[0]] = ts
        weight_sum[start:start + p.shape[0]] = ws
    valid = weight_sum >= float(min_weight)
    avg = tsdf_sum / weight_sum.clamp_min(1e-6)
    tsdf_vol = avg.reshape(nz, ny, nx)
    valid_vol = valid.reshape(nz, ny, nx)

    # Approximate normals from the TSDF grid gradient. Missing/flat gradients
    # fall back to camera-facing-ish z normals; isotropic fallback would blur.
    vol = tsdf_vol
    gx = F.pad(vol[:, :, 2:] - vol[:, :, :-2], (1, 1, 0, 0, 0, 0))
    gy = F.pad(vol[:, 2:, :] - vol[:, :-2, :], (0, 0, 1, 1, 0, 0))
    gz = F.pad(vol[2:, :, :] - vol[:-2, :, :], (0, 0, 0, 0, 1, 1))
    grad_flat = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    def center_surface() -> tuple[torch.Tensor, torch.Tensor, str]:
        cross = torch.zeros_like(valid_vol)
        for dim_i in range(3):
            a = [slice(None), slice(None), slice(None)]
            b = [slice(None), slice(None), slice(None)]
            a[dim_i] = slice(0, -1)
            b[dim_i] = slice(1, None)
            va = valid_vol[tuple(a)]
            vb = valid_vol[tuple(b)]
            ta = tsdf_vol[tuple(a)]
            tb = tsdf_vol[tuple(b)]
            edge_i = va & vb & ((ta * tb) <= 0)
            ca = cross[tuple(a)].clone()
            cb = cross[tuple(b)].clone()
            ca |= edge_i
            cb |= edge_i
            cross[tuple(a)] = ca
            cross[tuple(b)] = cb
        surf = (cross & valid_vol & (tsdf_vol.abs() <= float(surface_thresh))).reshape(-1)
        keep_i = torch.nonzero(surf, as_tuple=False).squeeze(1)
        if keep_i.numel() == 0:
            surf = valid & (avg.abs() <= min(float(surface_thresh), 0.25))
            keep_i = torch.nonzero(surf, as_tuple=False).squeeze(1)
        if keep_i.numel() == 0:
            keep_i = weight_sum.argmax().view(1)
        if keep_i.numel() > max_points:
            score_i = weight_sum[keep_i] - avg[keep_i].abs()
            keep_i = keep_i[score_i.topk(max_points, largest=True).indices]
        return pts[keep_i], grad_flat[keep_i], "centers"

    def edge_surface() -> tuple[torch.Tensor, torch.Tensor, str]:
        means_out = []
        normals_out = []
        scores_out = []
        for dim_i in range(3):
            a = [slice(None), slice(None), slice(None)]
            b = [slice(None), slice(None), slice(None)]
            a[dim_i] = slice(0, -1)
            b[dim_i] = slice(1, None)
            ta = tsdf_vol[tuple(a)]
            tb = tsdf_vol[tuple(b)]
            va = valid_vol[tuple(a)]
            vb = valid_vol[tuple(b)]
            edge_i = va & vb & ((ta * tb) <= 0) & (
                (ta.abs() <= float(surface_thresh)) | (tb.abs() <= float(surface_thresh))
            )
            idx = torch.nonzero(edge_i, as_tuple=False)
            if idx.numel() == 0:
                continue
            ta_i = ta[edge_i]
            tb_i = tb[edge_i]
            denom = ta_i - tb_i
            frac = torch.where(
                denom.abs() > 1e-6,
                (ta_i / denom).clamp(0.0, 1.0),
                ta_i.new_full(ta_i.shape, 0.5),
            )
            iz, iy, ix = idx[:, 0], idx[:, 1], idx[:, 2]
            x = xs[ix]
            y = ys[iy]
            z = zs[iz]
            if dim_i == 0:
                z = z + frac * float(voxel_size)
                ib = ((iz + 1) * ny + iy) * nx + ix
            elif dim_i == 1:
                y = y + frac * float(voxel_size)
                ib = (iz * ny + (iy + 1)) * nx + ix
            else:
                x = x + frac * float(voxel_size)
                ib = (iz * ny + iy) * nx + (ix + 1)
            ia = (iz * ny + iy) * nx + ix
            means_i = torch.stack([x, y, z], dim=-1)
            normals_i = grad_flat[ia] * (1.0 - frac[:, None]) + grad_flat[ib] * frac[:, None]
            scores_i = 0.5 * (weight_sum[ia] + weight_sum[ib]) - 0.5 * (ta_i.abs() + tb_i.abs())
            means_out.append(means_i)
            normals_out.append(normals_i)
            scores_out.append(scores_i)
        if not means_out:
            return center_surface()
        means_i = torch.cat(means_out, dim=0)
        normals_i = torch.cat(normals_out, dim=0)
        scores_i = torch.cat(scores_out, dim=0)
        if means_i.shape[0] > max_points:
            keep_i = scores_i.topk(max_points, largest=True).indices
            means_i = means_i[keep_i]
            normals_i = normals_i[keep_i]
        return means_i, normals_i, "edges"

    if surface_mode == "edges":
        means, normals, actual_surface_mode = edge_surface()
    else:
        means, normals, actual_surface_mode = center_surface()

    rgb_sum = torch.zeros(means.shape[0], 3, device=device, dtype=dtype)
    color_w = torch.zeros(means.shape[0], 1, device=device, dtype=dtype)
    frame_imgs = frames.permute(0, 3, 1, 2)
    for start in range(0, means.shape[0], chunk):
        p = means[start:start + chunk]
        rs = torch.zeros(p.shape[0], 3, device=device, dtype=dtype)
        rw = torch.zeros(p.shape[0], 1, device=device, dtype=dtype)
        best_w = torch.full((p.shape[0], 1), -1.0, device=device, dtype=dtype)
        best_rgb = torch.zeros(p.shape[0], 3, device=device, dtype=dtype)
        for i in range(v):
            z, grid, inb = _project_points(p, w2c[i], K[i], h, w)
            samp_d = _sample_map(depth_imgs[i:i + 1], grid)[:, 0]
            samp_m = _sample_map(mask_imgs[i:i + 1], grid)[:, 0]
            samp_rgb = _sample_map(frame_imgs[i:i + 1], grid)
            valid_i = inb & (samp_m > 0.5) & (samp_d < 1e5)
            sdf = (samp_d - z).abs()
            ww = torch.exp(-sdf / trunc)[:, None] * valid_i.to(dtype)[:, None]
            if color_mode == "select":
                take = ww > best_w
                best_rgb = torch.where(take.expand_as(best_rgb), samp_rgb, best_rgb)
                best_w = torch.maximum(best_w, ww)
            else:
                rs = rs + samp_rgb * ww
                rw = rw + ww
        if color_mode == "select":
            rs = best_rgb
            rw = (best_w > 0).to(dtype)
        rgb_sum[start:start + p.shape[0]] = rs
        color_w[start:start + p.shape[0]] = rw
    rgb = rgb_sum / color_w.clamp_min(1e-6)
    rgb = torch.where(color_w > 0, rgb, rgb.new_full(rgb.shape, 0.5)).clamp(0.0, 1.0)

    fallback = normals.new_tensor([0.0, 0.0, 1.0]).expand_as(normals)
    normals = torch.where(normals.norm(dim=-1, keepdim=True) > 1e-6, normals, fallback)
    quat = _quat_from_normals(normals)
    tang = float(voxel_size) * scale_mult
    norm = float(voxel_size) * normal_scale_mult
    scale = means.new_tensor([tang, tang, norm]).view(1, 3).expand(means.shape[0], 3).clone()
    opacity_t = means.new_full((means.shape[0], 1), float(opacity))
    params = {
        "mean": means,
        "quat": quat.to(dtype=dtype),
        "scale": scale,
        "opacity": opacity_t,
        "rgb": rgb,
        "depth": means.new_zeros(means.shape[0], 1),
        "scale_raw": scale.clone(),
        "mean_anchor": means.clone(),
        "mean_offset": means.new_zeros(means.shape),
    }
    stats = {
        "output": int(means.shape[0]),
        "voxels": int(n_vox),
        "voxel_size": float(voxel_size),
        "grid": [nx, ny, nz],
        "valid_voxels": int(valid.sum().item()),
        "color_mode": color_mode,
        "surface_mode": actual_surface_mode,
    }
    return params, stats


def rgbd_tsdf_filter_params(
    params: dict,
    frames: torch.Tensor,
    masks: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    c2w: torch.Tensor,
    *,
    voxel_size: float,
    trunc_mult: float = 4.0,
    min_weight: float = 2.0,
    band: float = 0.5,
    opacity_decay: float = 4.0,
    invalid_opacity_mult: float = 0.15,
    max_voxels: int = 1_500_000,
    bounds_pad_voxels: float = 3.0,
    chunk: int = 262_144,
) -> tuple[dict, dict]:
    """Downweight splats that are away from a projective TSDF surface.

    Unlike ``rgbd_tsdf_fuse``, this keeps the high-resolution decoder splats and
    uses the fused RGBD field only as a visibility/surface prior. It is still
    feed-forward and deterministic.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    v, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    device, dtype = params["mean"].device, params["mean"].dtype
    depths = depths.to(device=device, dtype=dtype)
    masks = masks.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    w2c = w2c.to(device=device, dtype=dtype)
    c2w = c2w.to(device=device, dtype=dtype)
    bounds = _rgbd_bounds(depths, masks, K, w2c, max_samples=200_000)
    if bounds is None:
        return params, {"filtered": 0, "voxels": 0, "valid_voxels": 0}
    mn, mx = bounds
    pad = max(float(voxel_size) * bounds_pad_voxels, 1e-4)
    mn = mn - pad
    mx = mx + pad
    dims = torch.ceil((mx - mn) / float(voxel_size)).to(torch.long) + 1
    dims = torch.clamp(dims, min=2)
    n_vox = int(dims.prod().item())
    if n_vox > max_voxels:
        factor = (n_vox / float(max_voxels)) ** (1.0 / 3.0)
        voxel_size = float(voxel_size) * factor
        dims = torch.ceil((mx - mn) / float(voxel_size)).to(torch.long) + 1
        dims = torch.clamp(dims, min=2)
        n_vox = int(dims.prod().item())
    nx, ny, nz = [int(x) for x in dims.tolist()]
    xs = mn[0] + torch.arange(nx, device=device, dtype=dtype) * float(voxel_size)
    ys = mn[1] + torch.arange(ny, device=device, dtype=dtype) * float(voxel_size)
    zs = mn[2] + torch.arange(nz, device=device, dtype=dtype) * float(voxel_size)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    depth_imgs = depths[:, None]
    mask_imgs = masks.permute(0, 3, 1, 2)
    trunc = max(float(voxel_size) * trunc_mult, 1e-6)
    tsdf_sum = torch.zeros(pts.shape[0], device=device, dtype=dtype)
    weight_sum = torch.zeros_like(tsdf_sum)
    for start in range(0, pts.shape[0], chunk):
        p = pts[start:start + chunk]
        ts = torch.zeros(p.shape[0], device=device, dtype=dtype)
        ws = torch.zeros_like(ts)
        for i in range(v):
            z, grid, inb = _project_points(p, w2c[i], K[i], h, w)
            samp_d = _sample_map(depth_imgs[i:i + 1], grid)[:, 0]
            samp_m = _sample_map(mask_imgs[i:i + 1], grid)[:, 0]
            valid = inb & (samp_m > 0.5) & (samp_d < 1e5)
            sdf = (samp_d - z).clamp(-trunc, trunc) / trunc
            ww = valid.to(dtype)
            ts = ts + sdf * ww
            ws = ws + ww
        tsdf_sum[start:start + p.shape[0]] = ts
        weight_sum[start:start + p.shape[0]] = ws
    avg = tsdf_sum / weight_sum.clamp_min(1e-6)
    tsdf_vol = avg.reshape(1, 1, nz, ny, nx)
    weight_vol = weight_sum.reshape(1, 1, nz, ny, nx)
    means = params["mean"]
    coords = (means - mn[None]) / (float(voxel_size) * (dims.to(dtype=dtype, device=device) - 1).clamp_min(1))
    grid = (coords * 2.0 - 1.0).view(1, -1, 1, 1, 3)
    tsdf_s = F.grid_sample(tsdf_vol, grid, mode="bilinear", padding_mode="border",
                           align_corners=True).view(-1)
    weight_s = F.grid_sample(weight_vol, grid, mode="bilinear", padding_mode="zeros",
                             align_corners=True).view(-1)
    valid_s = weight_s >= float(min_weight)
    excess = (tsdf_s.abs() - float(band)).clamp_min(0.0)
    decay = torch.exp(-float(opacity_decay) * excess).unsqueeze(-1)
    decay = torch.where(valid_s[:, None], decay, decay.new_full(decay.shape, float(invalid_opacity_mult)))
    out = dict(params)
    out["opacity"] = params["opacity"] * decay.to(dtype=params["opacity"].dtype)
    stats = {
        "filtered": int((decay.reshape(-1) < 0.99).sum().item()),
        "voxels": int(n_vox),
        "valid_voxels": int((weight_sum >= float(min_weight)).sum().item()),
        "voxel_size": float(voxel_size),
    }
    return out, stats
