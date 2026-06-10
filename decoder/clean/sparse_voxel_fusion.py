"""Learned multi-view fusion via a sparse 3D U-Net over voxelized splats.

Drop-in alternative to the deterministic ``fusion.voxel_fuse_params`` call in
``_predict_anchor_parts``. Replaces the hand-tuned per-voxel score/keep/decay
heuristics with a learned head that sees each voxel's actual 3D neighborhood.

Design (v1.1, first-principles fix after v1 collapsed to sharpness-via-deletion):
1. Reuse deterministic ``voxel_fuse_params`` to pick one representative splat
   per occupied voxel.  This stays differentiable through the kept splats.
2. Build a per-voxel feature vector that includes the multi-view consistency
   signal (``_fusion_score`` if available — the strongest cue for distinguishing
   shell artifacts from real surface).
3. Run a 4-layer same-resolution sparse 3D conv stack (SubMConv3d) on those
   voxels so each voxel sees its 3D neighborhood.
4. A zero-init head emits per-voxel (visibility, opacity_res, rgb_res,
   depth_res).
5. **Symmetric visibility** ``vis = 1 + tanh(.)*vis_delta`` ∈ [1-δ, 1+δ].
   This is the critical change vs v1's suppression-only ``vis = 1 - sigmoid(.)``:
   the network can BOOST under-supported correct surfaces, not just delete
   over-saturated shell.  Without symmetry the loss surface preferentially
   over-suppresses (drops FN catastrophically while gaming sharpness).
6. **Identity regularization** ``L_id = mean((vis - 1)^2)`` is exposed via
   ``vis_reg_loss()`` so the train loop can penalize movement from the prior
   unless there's strong photometric evidence.

The non-negotiable step-0 invariant (output ≡ prior at init) is preserved:
zero-init head with bias=0 ⇒ raw=0 ⇒ tanh(0)=0 ⇒ vis=1, residuals=0.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from decoder.clean.fusion import voxel_fuse_params

SPARSE_VOXEL_FUSION_FEATURES = 26
DENSE_VOXEL_CONTEXT_EXTRA_FEATURES = SPARSE_VOXEL_FUSION_FEATURES * 2 + 1


def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def _positive_zero_centered(raw: torch.Tensor) -> torch.Tensor:
    """Positive residual with identity value and useful gradient at init.

    ``torch.relu`` is zero-centered but PyTorch uses a zero gradient at exactly
    zero.  Enhance-only heads are zero-initialized, so ReLU would make the first
    step dead.  This equivalent max(raw, 0) form chooses the positive branch at
    raw == 0, giving gradient 1 for boost directions while still blocking
    negative/deletion directions.
    """
    return torch.where(raw >= 0, raw, raw * 0.0)


def sparse_voxel_fusion_features(fused: dict,
                                 radius: float,
                                 voxel_size: float) -> torch.Tensor:
    """Inference-available per-voxel features for the sparse fusion head."""
    rgb = fused["rgb"].reshape(-1, 3)
    opacity = fused["opacity"].reshape(-1, 1)
    scale = fused["scale"].reshape(-1, 3)
    mean = fused["mean"].reshape(-1, 3)
    radius_norm = max(float(radius), 1e-6)
    voxel_norm = max(float(voxel_size), 1e-6)

    # Geometry/context.  The radial term alone is ambiguous for symmetric
    # objects; normalized xyz and within-voxel offset give the sparse conv
    # enough signal to learn location-specific cleanup without target views.
    mean_norm = (mean / radius_norm).clamp(-2.0, 2.0) / 2.0
    q = torch.floor(mean.detach() / voxel_norm).to(dtype=mean.dtype)
    voxel_center = (q + 0.5) * voxel_norm
    voxel_offset = ((mean - voxel_center) / voxel_norm).clamp(-1.0, 1.0)

    log_scale_norm = _safe_log(scale.mean(dim=-1, keepdim=True) / radius_norm)
    depth_norm = (mean.norm(dim=-1, keepdim=True) / radius_norm) - 1.0

    def fusion_col(key: str, default: float = 0.0) -> torch.Tensor:
        value = fused.get(key)
        if value is None:
            return opacity.new_full(opacity.shape, float(default))
        return value.reshape(-1, 1).to(device=opacity.device, dtype=opacity.dtype)

    support = fusion_col("_fusion_support")
    conflict = fusion_col("_fusion_conflict")
    coverage = fusion_col("_fusion_coverage")
    color_support = fusion_col("_fusion_color_support")
    score = fusion_col("_fusion_score")
    depth_error_sum = fusion_col("_fusion_depth_error")
    color_error_sum = fusion_col("_fusion_color_error")
    front_conflict = fusion_col("_fusion_front_conflict")
    silhouette_conflict = fusion_col("_fusion_silhouette_conflict")
    detail = fusion_col("_fusion_detail")

    coverage_safe = coverage.clamp_min(1.0)
    support_safe = support.clamp_min(1.0)
    support_n = (support / 4.0).clamp(0.0, 2.0)
    conflict_n = (conflict / 4.0).clamp(0.0, 2.0)
    coverage_n = (coverage / 4.0).clamp(0.0, 2.0)
    color_support_n = (color_support / 4.0).clamp(0.0, 2.0)
    score_n = (score / 2.0).clamp(-2.0, 2.0) / 2.0

    support_ratio = (support / coverage_safe).clamp(0.0, 1.0)
    conflict_ratio = (conflict / coverage_safe).clamp(0.0, 1.0)
    color_support_ratio = (color_support / support_safe).clamp(0.0, 1.0)
    net_support = ((support - conflict) / coverage_safe).clamp(-1.0, 1.0)
    depth_error = (depth_error_sum / coverage_safe).clamp(0.0, 4.0) / 4.0
    color_error = (color_error_sum / coverage_safe).clamp(0.0, 1.0)
    front_ratio = (front_conflict / coverage_safe).clamp(0.0, 1.0)
    silhouette_ratio = (silhouette_conflict / coverage_safe).clamp(0.0, 1.0)

    return torch.cat([
        rgb,
        opacity,
        log_scale_norm,
        depth_norm,
        support_n,
        conflict_n,
        coverage_n,
        color_support_n,
        score_n,
        mean_norm,
        voxel_offset,
        support_ratio,
        conflict_ratio,
        color_support_ratio,
        net_support,
        depth_error,
        color_error,
        front_ratio,
        silhouette_ratio,
        detail.clamp(0.0, 1.0),
    ], dim=-1)


def dense_voxel_context_features(fused: dict,
                                 base_features: torch.Tensor,
                                 voxel_size: float,
                                 neighbor_radius: int = 1) -> torch.Tensor:
    """Append occupied-neighborhood aggregate features for dense MLP fallback.

    ``SparseVoxelFusion`` gets 3D context from sparse convolutions.  This helper
    gives the dependency-free MLP a cheaper local context signal: for every
    fused voxel, aggregate the already-computed per-voxel features over occupied
    voxels in a cubic neighborhood.
    """
    r = max(int(neighbor_radius), 0)
    if r <= 0 or base_features.shape[0] == 0:
        return base_features
    mean = fused["mean"].detach()
    voxel = max(float(voxel_size), 1e-9)
    q = torch.floor(mean / voxel).to(torch.long)
    q = q - q.amin(dim=0, keepdim=True)
    dims = (q.amax(dim=0) + 1).to(torch.long)
    stride_yz = dims[1] * dims[2]
    keys = q[:, 0] * stride_yz + q[:, 1] * dims[2] + q[:, 2]
    order = torch.argsort(keys)
    sorted_keys = keys[order]

    n = base_features.shape[0]
    sum_features = torch.zeros_like(base_features)
    count = torch.zeros(n, 1, device=base_features.device, dtype=base_features.dtype)
    one = torch.ones(n, 1, device=base_features.device, dtype=base_features.dtype)

    offsets = range(-r, r + 1)
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                qn = q + q.new_tensor([dx, dy, dz])
                inb = (
                    (qn[:, 0] >= 0) & (qn[:, 0] < dims[0])
                    & (qn[:, 1] >= 0) & (qn[:, 1] < dims[1])
                    & (qn[:, 2] >= 0) & (qn[:, 2] < dims[2])
                )
                if not bool(inb.any()):
                    continue
                dst = torch.nonzero(inb, as_tuple=False).squeeze(1)
                qn_v = qn[dst]
                neighbor_keys = (
                    qn_v[:, 0] * stride_yz + qn_v[:, 1] * dims[2] + qn_v[:, 2]
                )
                pos = torch.searchsorted(sorted_keys, neighbor_keys)
                hit = (pos < sorted_keys.shape[0]) & (sorted_keys[pos.clamp_max(sorted_keys.shape[0] - 1)] == neighbor_keys)
                if not bool(hit.any()):
                    continue
                dst_h = dst[hit]
                src_h = order[pos[hit]]
                sum_features.index_add_(0, dst_h, base_features[src_h])
                count.index_add_(0, dst_h, one[dst_h])

    count_safe = count.clamp_min(1.0)
    neighbor_mean = sum_features / count_safe
    max_neighbors = max((2 * r + 1) ** 3, 1)
    count_norm = (count / float(max_neighbors)).clamp(0.0, 1.0)
    return torch.cat([
        base_features,
        neighbor_mean,
        base_features - neighbor_mean,
        count_norm,
    ], dim=-1)


def _message_offsets(radius: int) -> tuple[tuple[int, int, int], ...]:
    r = max(int(radius), 0)
    return tuple(
        (dx, dy, dz)
        for dx in range(-r, r + 1)
        for dy in range(-r, r + 1)
        for dz in range(-r, r + 1)
    )


def dense_voxel_message_pairs(fused: dict,
                              voxel_size: float,
                              neighbor_radius: int = 1
                              ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return source/destination voxel pairs for dependency-free message passing.

    Each returned entry corresponds to one integer voxel offset. For a pair
    `(src, dst)`, messages flow from `src` voxels into `dst` voxels. The helper
    operates only on occupied voxels, so it is a sparse-conv-like neighborhood
    primitive without requiring `spconv`.
    """
    mean = fused["mean"].detach()
    if mean.shape[0] == 0:
        return []
    voxel = max(float(voxel_size), 1e-9)
    q = torch.floor(mean / voxel).to(torch.long)
    q = q - q.amin(dim=0, keepdim=True)
    dims = (q.amax(dim=0) + 1).to(torch.long)
    stride_yz = dims[1] * dims[2]
    keys = q[:, 0] * stride_yz + q[:, 1] * dims[2] + q[:, 2]
    order = torch.argsort(keys)
    sorted_keys = keys[order]

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for dx, dy, dz in _message_offsets(neighbor_radius):
        qn = q + q.new_tensor([dx, dy, dz])
        inb = (
            (qn[:, 0] >= 0) & (qn[:, 0] < dims[0])
            & (qn[:, 1] >= 0) & (qn[:, 1] < dims[1])
            & (qn[:, 2] >= 0) & (qn[:, 2] < dims[2])
        )
        if not bool(inb.any()):
            empty = torch.empty(0, dtype=torch.long, device=mean.device)
            pairs.append((empty, empty))
            continue
        dst = torch.nonzero(inb, as_tuple=False).squeeze(1)
        qv = qn[dst]
        neighbor_keys = qv[:, 0] * stride_yz + qv[:, 1] * dims[2] + qv[:, 2]
        pos = torch.searchsorted(sorted_keys, neighbor_keys)
        pos_safe = pos.clamp_max(max(int(sorted_keys.numel()) - 1, 0))
        hit = (pos < sorted_keys.numel()) & (sorted_keys[pos_safe] == neighbor_keys)
        if not bool(hit.any()):
            empty = torch.empty(0, dtype=torch.long, device=mean.device)
            pairs.append((empty, empty))
            continue
        pairs.append((order[pos[hit]], dst[hit]))
    return pairs


class DenseVoxelMessageBlock(nn.Module):
    """Small learned occupied-neighborhood message block.

    This is a dependency-free sparse convolution substitute. It applies one
    learned linear transform per integer voxel offset, aggregates messages onto
    destination voxels, then uses a residual feed-forward update.
    """

    def __init__(self, hidden: int, n_offsets: int):
        super().__init__()
        hidden = max(int(hidden), 8)
        n_offsets = max(int(n_offsets), 1)
        self.offset_linears = nn.ModuleList([
            nn.Linear(hidden, hidden, bias=False) for _ in range(n_offsets)
        ])
        self.norm_msg = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm_ff = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor,
                pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if h.shape[0] == 0:
            return h
        aggr = h.new_zeros(h.shape)
        count = h.new_zeros((h.shape[0], 1))
        for linear, (src, dst) in zip(self.offset_linears, pairs):
            if src.numel() == 0:
                continue
            aggr.index_add_(0, dst, linear(h[src]))
            count.index_add_(
                0,
                dst,
                torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype),
            )
        aggr = aggr / count.clamp_min(1.0)
        h = self.norm_msg(h + aggr)
        h = self.norm_ff(h + self.ff(h))
        return h


class SparseVoxelFusion(nn.Module):
    """Sparse 3D U-Net residual head on top of deterministic voxel fusion.

    Args:
        hidden: feature dim of the sparse 3D conv trunk.
        depth_res_frac: bound on per-voxel depth residual along the outward
            ray, as a fraction of orbit radius.
        rgb_res_scale: bound on rgb residual (tanh scale).
        opacity_res_scale: bound on additive opacity residual (tanh scale).
        vis_delta: bound on visibility multiplier deviation from 1.0.
            vis = 1 + tanh(raw)*vis_delta ∈ [1-δ, 1+δ].  Symmetric so the
            net can both suppress and boost.  Default 0.5 → vis ∈ [0.5, 1.5].
    """

    IN_FEATURES = SPARSE_VOXEL_FUSION_FEATURES
    OUT_FEATURES = 6  # [vis_raw(1), opacity_res(1), rgb_res(3), depth_res(1)]

    def __init__(self, hidden: int = 32,
                 depth_res_frac: float = 0.05,
                 rgb_res_scale: float = 0.1,
                 opacity_res_scale: float = 0.1,
                 vis_delta: float = 0.5,
                 enhance_only: bool = False,
                 target_vis_pos_min: float = 0.75,
                 target_vis_neg_max: float = 0.25,
                 target_vis_positive_weight: float = 1.0,
                 target_vis_negative_weight: float = 1.0):
        super().__init__()
        import spconv.pytorch as spconv
        self.enhance_only = bool(enhance_only)

        self._spconv = spconv
        self.depth_res_frac = float(depth_res_frac)
        self.rgb_res_scale = float(rgb_res_scale)
        self.opacity_res_scale = float(opacity_res_scale)
        self.vis_delta = float(vis_delta)
        self.target_vis_pos_min = float(target_vis_pos_min)
        self.target_vis_neg_max = float(target_vis_neg_max)
        self.target_vis_positive_weight = float(target_vis_positive_weight)
        self.target_vis_negative_weight = float(target_vis_negative_weight)

        # Same-resolution sparse 3D conv stack.  SubMConv3d preserves the
        # active site set (same voxels in / out) so we can stack cheaply.
        self.conv1 = spconv.SubMConv3d(self.IN_FEATURES, hidden, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = spconv.SubMConv3d(hidden, hidden * 2, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden * 2)
        self.conv3 = spconv.SubMConv3d(hidden * 2, hidden * 2, 3, bias=False)
        self.bn3 = nn.BatchNorm1d(hidden * 2)
        self.conv4 = spconv.SubMConv3d(hidden * 2, hidden, 3, bias=False)
        self.bn4 = nn.BatchNorm1d(hidden)

        # 1x1 head, zero-init → step-0 output is the deterministic prior.
        # With symmetric vis = 1 + tanh(raw)·δ, raw=0 ⇒ vis=1 exactly.  No
        # bias trick needed (and no sigmoid offset error).
        self.head = spconv.SubMConv3d(hidden, self.OUT_FEATURES, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Enhance-only mode uses ``_positive_zero_centered`` in decoding:
        # raw=0 is exact identity with a usable positive-branch gradient.
        # This avoids the old sigmoid(-10) near-identity trick, which also
        # made the first learning steps nearly gradient-free.

        # Cached for the train loop to query as a regularization term.
        # Set during refine() and consumed via vis_reg_loss().
        self._last_vis_for_reg: Optional[torch.Tensor] = None
        self._last_op_res_for_reg: Optional[torch.Tensor] = None
        self._last_rgb_res_for_reg: Optional[torch.Tensor] = None
        self._last_support_reg: Optional[torch.Tensor] = None
        self._last_target_vis_loss: Optional[torch.Tensor] = None

    def _per_voxel_features(self, fused: dict, radius: float, voxel_size: float) -> torch.Tensor:
        return sparse_voxel_fusion_features(fused, radius, voxel_size)

    def _voxel_coords(self, fused: dict, voxel_size: float) -> tuple[torch.Tensor, list[int]]:
        mean = fused["mean"].detach()
        q = torch.floor(mean / float(voxel_size)).to(torch.int32)
        q = q - q.amin(dim=0, keepdim=True)
        dims = (q.amax(dim=0) + 1).tolist()
        m = q.shape[0]
        coords = torch.cat([
            torch.zeros(m, 1, dtype=torch.int32, device=q.device),
            q,
        ], dim=1)
        return coords, dims

    def _apply_bn(self, x, bn: nn.BatchNorm1d):
        return x.replace_feature(bn(x.features))

    def _act(self, x):
        return x.replace_feature(torch.relu(x.features))

    def _cache_support_reg(self, fused: dict, out_opacity: torch.Tensor) -> None:
        support = fused.get("_fusion_support")
        conflict = fused.get("_fusion_conflict")
        if support is None or conflict is None:
            self._last_support_reg = None
            return
        support = support.detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        conflict = conflict.detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        base_opacity = fused["opacity"].detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        support_frac = (support / 2.0).clamp(0.0, 1.0)
        conflict_frac = (conflict / (support + conflict + 1e-6)).clamp(0.0, 1.0)
        weak_or_conflict = (1.0 - support_frac + conflict_frac).clamp(0.0, 1.0)
        strong_support = (support_frac * (1.0 - conflict_frac)).clamp(0.0, 1.0)
        delta = out_opacity - base_opacity
        pos = torch.relu(delta)
        neg = torch.relu(-delta)
        self._last_support_reg = (
            (weak_or_conflict * pos.square()).mean()
            + 0.5 * (strong_support * neg.square()).mean()
        )

    def _cache_target_vis_loss(self, fused: dict, out_opacity: torch.Tensor) -> None:
        support = fused.get("_fusion_target_support")
        conflict = fused.get("_fusion_target_conflict")
        if support is None or conflict is None:
            self._last_target_vis_loss = None
            return
        support = support.detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        conflict = conflict.detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        denom = support + conflict
        valid = denom > 0
        if not bool(valid.any()):
            self._last_target_vis_loss = out_opacity.new_zeros(())
            return
        target_soft = torch.where(valid, support / denom.clamp_min(1.0), support.new_zeros(()))
        pos_min = float(self.target_vis_pos_min)
        neg_max = float(self.target_vis_neg_max)
        if pos_min > 0.0 or neg_max < 1.0:
            pos = valid & (target_soft >= pos_min)
            neg = valid & (target_soft <= neg_max)
            valid = pos | neg
            target = torch.where(pos, target_soft.new_ones(()), target_soft.new_zeros(()))
        else:
            target = target_soft
        if not bool(valid.any()):
            self._last_target_vis_loss = out_opacity.new_zeros(())
            return
        base_opacity = fused["opacity"].detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        gate = (out_opacity / base_opacity.clamp_min(1e-4)).clamp(1e-4, 1.0 - 1e-4)
        loss_px = nn.functional.binary_cross_entropy(gate, target, reduction="none")
        pos_w = max(float(self.target_vis_positive_weight), 0.0)
        neg_w = max(float(self.target_vis_negative_weight), 0.0)
        weights = target * pos_w + (1.0 - target) * neg_w
        self._last_target_vis_loss = (
            (loss_px[valid] * weights[valid]).sum()
            / weights[valid].sum().clamp_min(1e-6)
        )

    def _decode_outputs(self, raw: torch.Tensor, fused: dict, radius: float) -> dict:
        # raw: (M, 6) = [vis_raw, op_res_raw, rgb_res_raw(3), depth_res_raw]
        if self.enhance_only:
            # Enhance-only: vis ∈ [1, 1+2δ], op_res ∈ [0, 2·op_max].  Deletion
            # is architecturally impossible.  Zero-init gives exact identity,
            # and ``_positive_zero_centered`` provides gradient 1 at raw=0 for
            # boost directions instead of the dead sigmoid(-10) behavior.
            vis_boost = torch.tanh(_positive_zero_centered(raw[:, 0:1]))
            op_boost = torch.tanh(_positive_zero_centered(raw[:, 1:2]))
            vis = 1.0 + vis_boost * (2.0 * self.vis_delta)
            op_res = op_boost * (2.0 * self.opacity_res_scale)
        else:
            # Symmetric vis: tanh(0)=0 → vis=1.0 exactly at init.
            vis = 1.0 + torch.tanh(raw[:, 0:1]) * self.vis_delta
            op_res = torch.tanh(raw[:, 1:2]) * self.opacity_res_scale
        rgb_res = torch.tanh(raw[:, 2:5]) * self.rgb_res_scale
        depth_res = torch.tanh(raw[:, 5:6]) * (self.depth_res_frac * radius)

        # Cache for regularization: penalize the WHOLE delta from prior,
        # not just vis.  v1.1 failed because deletion can also flow through
        # op_res (subtract opacity below silhouette).
        self._last_vis_for_reg = vis
        self._last_op_res_for_reg = op_res
        self._last_rgb_res_for_reg = rgb_res

        out = dict(fused)
        out["opacity"] = (fused["opacity"] * vis + op_res).clamp(0.0, 1.0)
        self._cache_support_reg(fused, out["opacity"])
        self._cache_target_vis_loss(fused, out["opacity"])
        out["rgb"] = (fused["rgb"] + rgb_res).clamp(0.0, 1.0)
        m = fused["mean"]
        norm = m.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ray_dir = m / norm
        out["mean"] = m + depth_res * ray_dir
        return out

    def vis_reg_loss(self) -> torch.Tensor:
        """Identity regularization across ALL residual channels.

        mean((vis-1)^2) + mean(op_res^2) + 0.25 * mean(rgb_res^2)

        Pushes the entire residual head toward zero-delta (preserve the
        prior) unless there is strong photometric gradient signal otherwise.
        Without this, the network gamed the metric via deletion through
        op_res even when vis was constrained (v1.1 failure).
        """
        if self._last_vis_for_reg is None:
            return torch.zeros((), dtype=torch.float32)
        reg = ((self._last_vis_for_reg - 1.0) ** 2).mean()
        if self._last_op_res_for_reg is not None:
            reg = reg + (self._last_op_res_for_reg ** 2).mean()
        if self._last_rgb_res_for_reg is not None:
            # 0.25 weight on rgb because color shifts are 3-channel and
            # generally less destructive than geometry/opacity changes.
            reg = reg + 0.25 * (self._last_rgb_res_for_reg ** 2).mean()
        return reg

    def support_reg_loss(self) -> torch.Tensor:
        if self._last_support_reg is None:
            if self._last_vis_for_reg is not None:
                return self._last_vis_for_reg.new_zeros(())
            return torch.zeros((), dtype=torch.float32)
        return self._last_support_reg

    def target_vis_loss(self) -> torch.Tensor:
        if self._last_target_vis_loss is None:
            if self._last_vis_for_reg is not None:
                return self._last_vis_for_reg.new_zeros(())
            return torch.zeros((), dtype=torch.float32)
        return self._last_target_vis_loss

    def refine(self, fused: dict, voxel_size: float, radius: float) -> dict:
        """Sparse-voxel learned residual on an already-fused splat dict.

        Call site: ``_maybe_voxel_fuse_params`` in train_phase2.py, right
        after the deterministic ``voxel_fuse_params(...)`` produces the
        representative-per-voxel dict.
        """
        m = fused["mean"].shape[0]
        if m == 0:
            self._last_vis_for_reg = None
            self._last_support_reg = None
            self._last_target_vis_loss = None
            return fused

        feats = self._per_voxel_features(fused, radius, voxel_size)
        coords, dims = self._voxel_coords(fused, voxel_size)

        spconv = self._spconv
        sp = spconv.SparseConvTensor(feats, coords, spatial_shape=dims, batch_size=1)
        sp = self._act(self._apply_bn(self.conv1(sp), self.bn1))
        sp = self._act(self._apply_bn(self.conv2(sp), self.bn2))
        sp = self._act(self._apply_bn(self.conv3(sp), self.bn3))
        sp = self._act(self._apply_bn(self.conv4(sp), self.bn4))
        raw = self.head(sp).features  # (M, 6)

        return self._decode_outputs(raw, fused, radius)

    def forward(self, fused: dict, voxel_size: float, radius: float) -> dict:
        return self.refine(fused, voxel_size, radius)


class DenseVoxelFusionMLP(nn.Module):
    """Dense MLP fallback for learned fused-splat residuals.

    This uses the same per-voxel/fused-splat features and residual decoding as
    ``SparseVoxelFusion`` but avoids the optional ``spconv`` dependency. It has
    no 3D neighborhood context, so it is less expressive than the sparse 3D conv
    head, but it keeps the important properties for local iteration:

    - zero-init output exactly preserves the deterministic scaffold at step 0;
    - render loss flows through fused opacity/RGB/depth residuals;
    - symmetric or enhance-only visibility prevents the one-way deletion bias.
    - optional occupied-neighborhood aggregates provide a cheap substitute for
      sparse 3D convolution context when `spconv` is unavailable.
    """

    IN_FEATURES = SPARSE_VOXEL_FUSION_FEATURES
    OUT_FEATURES = 6

    def __init__(self, hidden: int = 64, layers: int = 3,
                 depth_res_frac: float = 0.05,
                 rgb_res_scale: float = 0.1,
                 opacity_res_scale: float = 0.1,
                 vis_delta: float = 0.5,
                 enhance_only: bool = False,
                 neighbor_radius: int = 0,
                 target_vis_pos_min: float = 0.75,
                 target_vis_neg_max: float = 0.25,
                 target_vis_positive_weight: float = 1.0,
                 target_vis_negative_weight: float = 1.0):
        super().__init__()
        hidden = max(int(hidden), 8)
        layers = max(int(layers), 1)
        self.depth_res_frac = float(depth_res_frac)
        self.rgb_res_scale = float(rgb_res_scale)
        self.opacity_res_scale = float(opacity_res_scale)
        self.vis_delta = float(vis_delta)
        self.enhance_only = bool(enhance_only)
        self.neighbor_radius = max(int(neighbor_radius), 0)
        self.target_vis_pos_min = float(target_vis_pos_min)
        self.target_vis_neg_max = float(target_vis_neg_max)
        self.target_vis_positive_weight = float(target_vis_positive_weight)
        self.target_vis_negative_weight = float(target_vis_negative_weight)
        blocks: list[nn.Module] = []
        cin = self.IN_FEATURES
        if self.neighbor_radius > 0:
            cin = self.IN_FEATURES + DENSE_VOXEL_CONTEXT_EXTRA_FEATURES
        for _ in range(max(layers - 1, 0)):
            blocks.extend([
                nn.Linear(cin, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            ])
            cin = hidden
        self.body = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Linear(cin, self.OUT_FEATURES)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self._last_vis_for_reg: Optional[torch.Tensor] = None
        self._last_op_res_for_reg: Optional[torch.Tensor] = None
        self._last_rgb_res_for_reg: Optional[torch.Tensor] = None
        self._last_support_reg: Optional[torch.Tensor] = None
        self._last_target_vis_loss: Optional[torch.Tensor] = None

    def _cache_support_reg(self, fused: dict, out_opacity: torch.Tensor) -> None:
        support = fused.get("_fusion_support")
        conflict = fused.get("_fusion_conflict")
        if support is None or conflict is None:
            self._last_support_reg = None
            return
        support = support.detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        conflict = conflict.detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        base_opacity = fused["opacity"].detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        support_frac = (support / 2.0).clamp(0.0, 1.0)
        conflict_frac = (conflict / (support + conflict + 1e-6)).clamp(0.0, 1.0)
        weak_or_conflict = (1.0 - support_frac + conflict_frac).clamp(0.0, 1.0)
        strong_support = (support_frac * (1.0 - conflict_frac)).clamp(0.0, 1.0)
        delta = out_opacity - base_opacity
        pos = torch.relu(delta)
        neg = torch.relu(-delta)
        self._last_support_reg = (
            (weak_or_conflict * pos.square()).mean()
            + 0.5 * (strong_support * neg.square()).mean()
        )

    def _cache_target_vis_loss(self, fused: dict, out_opacity: torch.Tensor) -> None:
        support = fused.get("_fusion_target_support")
        conflict = fused.get("_fusion_target_conflict")
        if support is None or conflict is None:
            self._last_target_vis_loss = None
            return
        support = support.detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        conflict = conflict.detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        denom = support + conflict
        valid = denom > 0
        if not bool(valid.any()):
            self._last_target_vis_loss = out_opacity.new_zeros(())
            return
        target_soft = torch.where(valid, support / denom.clamp_min(1.0), support.new_zeros(()))
        pos_min = float(self.target_vis_pos_min)
        neg_max = float(self.target_vis_neg_max)
        if pos_min > 0.0 or neg_max < 1.0:
            pos = valid & (target_soft >= pos_min)
            neg = valid & (target_soft <= neg_max)
            valid = pos | neg
            target = torch.where(pos, target_soft.new_ones(()), target_soft.new_zeros(()))
        else:
            target = target_soft
        if not bool(valid.any()):
            self._last_target_vis_loss = out_opacity.new_zeros(())
            return
        base_opacity = fused["opacity"].detach().to(device=out_opacity.device, dtype=out_opacity.dtype)
        gate = (out_opacity / base_opacity.clamp_min(1e-4)).clamp(1e-4, 1.0 - 1e-4)
        loss_px = nn.functional.binary_cross_entropy(gate, target, reduction="none")
        pos_w = max(float(self.target_vis_positive_weight), 0.0)
        neg_w = max(float(self.target_vis_negative_weight), 0.0)
        weights = target * pos_w + (1.0 - target) * neg_w
        self._last_target_vis_loss = (
            (loss_px[valid] * weights[valid]).sum()
            / weights[valid].sum().clamp_min(1e-6)
        )

    def _decode_outputs(self, raw: torch.Tensor, fused: dict, radius: float) -> dict:
        if self.enhance_only:
            vis_boost = torch.tanh(_positive_zero_centered(raw[:, 0:1]))
            op_boost = torch.tanh(_positive_zero_centered(raw[:, 1:2]))
            vis = 1.0 + vis_boost * (2.0 * self.vis_delta)
            op_res = op_boost * (2.0 * self.opacity_res_scale)
        else:
            vis = 1.0 + torch.tanh(raw[:, 0:1]) * self.vis_delta
            op_res = torch.tanh(raw[:, 1:2]) * self.opacity_res_scale
        rgb_res = torch.tanh(raw[:, 2:5]) * self.rgb_res_scale
        depth_res = torch.tanh(raw[:, 5:6]) * (self.depth_res_frac * radius)

        self._last_vis_for_reg = vis
        self._last_op_res_for_reg = op_res
        self._last_rgb_res_for_reg = rgb_res

        out = dict(fused)
        out["opacity"] = (fused["opacity"] * vis + op_res).clamp(0.0, 1.0)
        self._cache_support_reg(fused, out["opacity"])
        self._cache_target_vis_loss(fused, out["opacity"])
        out["rgb"] = (fused["rgb"] + rgb_res).clamp(0.0, 1.0)
        m = fused["mean"]
        norm = m.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ray_dir = m / norm
        out["mean"] = m + depth_res * ray_dir
        return out

    def vis_reg_loss(self) -> torch.Tensor:
        if self._last_vis_for_reg is None:
            return torch.zeros((), dtype=torch.float32)
        reg = ((self._last_vis_for_reg - 1.0) ** 2).mean()
        if self._last_op_res_for_reg is not None:
            reg = reg + (self._last_op_res_for_reg ** 2).mean()
        if self._last_rgb_res_for_reg is not None:
            reg = reg + 0.25 * (self._last_rgb_res_for_reg ** 2).mean()
        return reg

    def support_reg_loss(self) -> torch.Tensor:
        if self._last_support_reg is None:
            if self._last_vis_for_reg is not None:
                return self._last_vis_for_reg.new_zeros(())
            return torch.zeros((), dtype=torch.float32)
        return self._last_support_reg

    def target_vis_loss(self) -> torch.Tensor:
        if self._last_target_vis_loss is None:
            if self._last_vis_for_reg is not None:
                return self._last_vis_for_reg.new_zeros(())
            return torch.zeros((), dtype=torch.float32)
        return self._last_target_vis_loss

    def refine(self, fused: dict, voxel_size: float, radius: float) -> dict:
        if fused["mean"].shape[0] == 0:
            self._last_vis_for_reg = None
            self._last_support_reg = None
            self._last_target_vis_loss = None
            return fused
        feats = sparse_voxel_fusion_features(fused, radius, voxel_size)
        if self.neighbor_radius > 0:
            feats = dense_voxel_context_features(
                fused, feats, voxel_size, self.neighbor_radius
            )
        raw = self.head(self.body(feats))
        return self._decode_outputs(raw, fused, radius)

    def forward(self, fused: dict, voxel_size: float, radius: float) -> dict:
        return self.refine(fused, voxel_size, radius)


class DenseVoxelMessageFusionMLP(DenseVoxelFusionMLP):
    """Learned local-message fused-splat residual head.

    `DenseVoxelFusionMLP` can see fixed neighbor statistics, but it cannot
    learn how surface evidence should propagate between occupied voxels. This
    head adds sparse-conv-like message passing over the occupied voxel graph
    while keeping the same zero-init output invariant.
    """

    def __init__(self, hidden: int = 128, layers: int = 2,
                 message_radius: int = 1,
                 depth_res_frac: float = 0.05,
                 rgb_res_scale: float = 0.1,
                 opacity_res_scale: float = 0.1,
                 vis_delta: float = 0.5,
                 enhance_only: bool = False,
                 neighbor_radius: int = 0,
                 target_vis_pos_min: float = 0.75,
                 target_vis_neg_max: float = 0.25,
                 target_vis_positive_weight: float = 1.0,
                 target_vis_negative_weight: float = 1.0):
        super().__init__(
            hidden=hidden,
            layers=1,
            depth_res_frac=depth_res_frac,
            rgb_res_scale=rgb_res_scale,
            opacity_res_scale=opacity_res_scale,
            vis_delta=vis_delta,
            enhance_only=enhance_only,
            neighbor_radius=neighbor_radius,
            target_vis_pos_min=target_vis_pos_min,
            target_vis_neg_max=target_vis_neg_max,
            target_vis_positive_weight=target_vis_positive_weight,
            target_vis_negative_weight=target_vis_negative_weight,
        )
        hidden = max(int(hidden), 8)
        layers = max(int(layers), 1)
        self.message_radius = max(int(message_radius), 0)
        self.message_offsets = _message_offsets(self.message_radius)
        cin = self.IN_FEATURES
        if self.neighbor_radius > 0:
            cin = self.IN_FEATURES + DENSE_VOXEL_CONTEXT_EXTRA_FEATURES
        self.input_proj = nn.Sequential(
            nn.Linear(cin, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.message_blocks = nn.ModuleList([
            DenseVoxelMessageBlock(hidden, len(self.message_offsets))
            for _ in range(layers)
        ])
        self.head = nn.Linear(hidden, self.OUT_FEATURES)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def refine(self, fused: dict, voxel_size: float, radius: float) -> dict:
        if fused["mean"].shape[0] == 0:
            self._last_vis_for_reg = None
            self._last_support_reg = None
            self._last_target_vis_loss = None
            return fused
        feats = sparse_voxel_fusion_features(fused, radius, voxel_size)
        if self.neighbor_radius > 0:
            feats = dense_voxel_context_features(
                fused, feats, voxel_size, self.neighbor_radius
            )
        h = self.input_proj(feats)
        pairs = dense_voxel_message_pairs(
            fused, voxel_size, neighbor_radius=self.message_radius
        )
        for block in self.message_blocks:
            h = block(h, pairs)
        raw = self.head(h)
        return self._decode_outputs(raw, fused, radius)
