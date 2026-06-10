"""Small feed-forward conditioning refiners shared by training/eval scripts."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rgb_border_mask(
    frames: torch.Tensor,
    threshold: float = 0.12,
    softness: float = 0.02,
) -> torch.Tensor:
    """Infer a soft foreground mask from decoded RGB on a white/background border.

    ``frames`` has shape ``(V,H,W,3)`` in ``[0,1]``. The function uses only the
    frame pixels, so it is available at inference when dataset masks are not.
    """
    rgb = frames.to(dtype=torch.float32).clamp(0.0, 1.0)
    h, w = rgb.shape[1], rgb.shape[2]
    b = max(2, min(h, w) // 32)
    border = torch.cat([
        rgb[:, :b].reshape(rgb.shape[0], -1, 3),
        rgb[:, -b:].reshape(rgb.shape[0], -1, 3),
        rgb[:, :, :b].reshape(rgb.shape[0], -1, 3),
        rgb[:, :, -b:].reshape(rgb.shape[0], -1, 3),
    ], dim=1)
    bg = border.mean(dim=1).view(rgb.shape[0], 1, 1, 3)
    score = torch.linalg.vector_norm(rgb - bg, dim=-1, keepdim=True) / (3.0 ** 0.5)
    if softness > 0:
        out = torch.sigmoid((score - float(threshold)) / max(float(softness), 1e-6))
    else:
        out = (score > float(threshold)).to(rgb.dtype)
    return out.to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)


class ConditionRGBRefineUNet(torch.nn.Module):
    """Shared RGB residual refiner for latent-decoded conditioning frames.

    The module maps ``[rgb * mask, mask]`` to a bounded RGB residual. The final
    layer is zero-initialized, so enabling the head preserves the original
    conditioning frames until pretrained weights are loaded or it is trained.
    """

    in_channels = 4

    def __init__(self, hidden: int = 32):
        super().__init__()
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = self._block(self.in_channels, hidden)
        self.down1 = self._down(hidden, h2)
        self.down2 = self._down(h2, h4)
        self.mid = self._block(h4, h4)
        self.up1 = self._block(h4 + h2, h2)
        self.up2 = self._block(h2 + hidden, hidden)
        self.head = torch.nn.Conv2d(hidden, 3, kernel_size=3, padding=1)
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
                f"condition-rgb-refine feature bug: got {features.shape[1]} "
                f"channels, expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        e2 = self.down1(e1)
        x = self.down2(e2)
        x = self.mid(x)
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, e1], dim=1))
        return self.head(x)


def apply_rgb_refiner(
    head: ConditionRGBRefineUNet,
    frames: torch.Tensor,
    masks: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    """Apply a bounded foreground RGB residual to ``(V,H,W,3)`` frames."""
    if residual_scale <= 0:
        return frames
    mask = masks.to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)
    feat = torch.cat([
        frames.permute(0, 3, 1, 2).clamp(0.0, 1.0) * mask.permute(0, 3, 1, 2),
        mask.permute(0, 3, 1, 2),
    ], dim=1)
    delta = torch.tanh(head(feat)) * float(residual_scale)
    delta = delta.permute(0, 2, 3, 1) * mask
    return (frames + delta).clamp(0.0, 1.0)


class ConditionRGBDRefineUNet(torch.nn.Module):
    """Shared conditioning refiner for latent-decoded RGB plus depth.

    The input is ``[rgb * mask, mask, depth_frac, depth_valid]`` plus optional
    multi-view support features. The output contains a bounded RGB residual and
    a bounded logit residual on normalized ray-depth. The final layer is
    zero-initialized, so enabling the module preserves both input RGB and input
    depth until trained weights are loaded or learned.
    """

    in_channels = 6

    def __init__(self, hidden: int = 32, in_channels: int | None = None):
        super().__init__()
        self.in_channels = int(in_channels or self.in_channels)
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = ConditionRGBRefineUNet._block(self.in_channels, hidden)
        self.down1 = ConditionRGBRefineUNet._down(hidden, h2)
        self.down2 = ConditionRGBRefineUNet._down(h2, h4)
        self.mid = ConditionRGBRefineUNet._block(h4, h4)
        self.up1 = ConditionRGBRefineUNet._block(h4 + h2, h2)
        self.up2 = ConditionRGBRefineUNet._block(h2 + hidden, hidden)
        self.head = torch.nn.Conv2d(hidden, 4, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"condition-rgbd-refine feature bug: got {features.shape[1]} "
                f"channels, expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        e2 = self.down1(e1)
        x = self.down2(e2)
        x = self.mid(x)
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, e1], dim=1))
        return self.head(x)


class ConditionRGBDViewRefineUNet(torch.nn.Module):
    """RGBD refiner with cross-view context over the conditioning orbit.

    The local U-Net extracts per-view features, then a small transformer mixes
    pooled tokens across the selected conditioning views. The view-context token
    is broadcast back into the decoder path before the final RGB/depth residual
    head. The output head is still zero-initialized, so step 0 preserves the
    deterministic RGBD input exactly.
    """

    in_channels = 6

    def __init__(self, hidden: int = 32, in_channels: int | None = None,
                 max_views: int = 64, context_layers: int = 2,
                 context_heads: int = 4):
        super().__init__()
        self.in_channels = int(in_channels or self.in_channels)
        self.max_views = max(int(max_views), 1)
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = ConditionRGBRefineUNet._block(self.in_channels, hidden)
        self.down1 = ConditionRGBRefineUNet._down(hidden, h2)
        self.down2 = ConditionRGBRefineUNet._down(h2, h4)
        self.mid = ConditionRGBRefineUNet._block(h4, h4)
        heads = max(int(context_heads), 1)
        heads = min(heads, h4)
        while h4 % heads != 0 and heads > 1:
            heads -= 1
        self.view_pos = torch.nn.Parameter(torch.zeros(1, self.max_views, h4))
        if context_layers > 0:
            layer = torch.nn.TransformerEncoderLayer(
                d_model=h4,
                nhead=heads,
                dim_feedforward=h4 * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context = torch.nn.TransformerEncoder(
                layer, num_layers=int(context_layers)
            )
        else:
            self.context = torch.nn.Identity()
        self.context_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(h4),
            torch.nn.Linear(h4, h4),
            torch.nn.GELU(),
        )
        self.context_fuse = ConditionRGBRefineUNet._block(h4 + h4, h4)
        self.up1 = ConditionRGBRefineUNet._block(h4 + h2, h2)
        self.up2 = ConditionRGBRefineUNet._block(h2 + hidden, hidden)
        self.head = torch.nn.Conv2d(hidden, 4, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"condition-rgbd-view-refine feature bug: got {features.shape[1]} "
                f"channels, expected {self.in_channels}"
            )
        if features.shape[0] > self.max_views:
            raise RuntimeError(
                f"condition-rgbd-view-refine got {features.shape[0]} views, "
                f"max_views is {self.max_views}"
            )
        e1 = self.enc1(features)
        e2 = self.down1(e1)
        x = self.down2(e2)
        x = self.mid(x)
        tokens = x.mean(dim=(-2, -1))[None] + self.view_pos[:, :x.shape[0]]
        ctx = self.context(tokens)[0]
        ctx = self.context_proj(ctx)
        ctx_map = ctx[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        x = self.context_fuse(torch.cat([x, ctx_map], dim=1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, e1], dim=1))
        return self.head(x)


class ConditionPoseHead(torch.nn.Module):
    """Predict conditioning-orbit camera pose tokens from RGBD views.

    This is an auxiliary bridge toward generated LTX orbits where camera poses
    are not given at inference. It consumes the same compact RGBD features used
    by the conditioning refiners and predicts per-view
    ``center_dir(3), forward_dir(3), log_dist(1)``.
    """

    in_channels = 6

    def __init__(self, hidden: int = 64, in_channels: int | None = None,
                 max_views: int = 64, context_layers: int = 2,
                 context_heads: int = 4):
        super().__init__()
        self.in_channels = int(in_channels or self.in_channels)
        self.max_views = max(int(max_views), 1)
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = ConditionRGBRefineUNet._block(self.in_channels, hidden)
        self.down1 = ConditionRGBRefineUNet._down(hidden, h2)
        self.down2 = ConditionRGBRefineUNet._down(h2, h4)
        self.mid = ConditionRGBRefineUNet._block(h4, h4)
        heads = max(int(context_heads), 1)
        heads = min(heads, h4)
        while h4 % heads != 0 and heads > 1:
            heads -= 1
        self.view_pos = torch.nn.Parameter(torch.zeros(1, self.max_views, h4))
        torch.nn.init.normal_(self.view_pos, std=0.02)
        if context_layers > 0:
            layer = torch.nn.TransformerEncoderLayer(
                d_model=h4,
                nhead=heads,
                dim_feedforward=h4 * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context = torch.nn.TransformerEncoder(
                layer, num_layers=int(context_layers)
            )
        else:
            self.context = torch.nn.Identity()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(h4),
            torch.nn.Linear(h4, h4),
            torch.nn.GELU(),
            torch.nn.Linear(h4, 7),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"condition-pose feature bug: got {features.shape[1]} "
                f"channels, expected {self.in_channels}"
            )
        if features.shape[0] > self.max_views:
            raise RuntimeError(
                f"condition-pose got {features.shape[0]} views, "
                f"max_views is {self.max_views}"
            )
        x = self.enc1(features)
        x = self.down1(x)
        x = self.down2(x)
        x = self.mid(x)
        tokens = x.mean(dim=(-2, -1))[None] + self.view_pos[:, :x.shape[0]]
        tokens = self.context(tokens)[0]
        return self.head(tokens)


class ConditionDepthAffineHead(torch.nn.Module):
    """Predict a bounded per-conditioning-view affine correction for depth.

    This head is intentionally global: it sees compact RGB/depth/mask summary
    statistics for each conditioning view and predicts ``scale, shift`` on
    normalized ray-depth. The final layer is zero-initialized, so enabling it
    preserves the input depth until the shared weights are trained.
    """

    def __init__(self, in_features: int = 12, hidden: int = 64, layers: int = 3):
        super().__init__()
        in_features = max(int(in_features), 1)
        hidden = max(int(hidden), 8)
        layers = max(int(layers), 1)
        blocks = []
        cin = in_features
        for _ in range(max(layers - 1, 0)):
            blocks.extend([
                torch.nn.Linear(cin, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
            ])
            cin = hidden
        self.body = torch.nn.Sequential(*blocks) if blocks else torch.nn.Identity()
        self.head = torch.nn.Linear(cin, 2)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.in_features = in_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.in_features:
            raise RuntimeError(
                f"condition-depth-affine feature bug: got {features.shape[-1]} "
                f"channels, expected {self.in_features}"
            )
        return self.head(self.body(features))


def apply_rgbd_refiner(
    head: torch.nn.Module,
    frames: torch.Tensor,
    masks: torch.Tensor,
    depth_frac: torch.Tensor,
    depth_valid: torch.Tensor,
    rgb_residual_scale: float,
    depth_delta_scale: float,
    extra_features: torch.Tensor | None = None,
    apply_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply bounded RGB and normalized-depth residuals to conditioning views."""
    mask = masks.to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)
    d_frac = depth_frac.to(device=frames.device, dtype=frames.dtype).clamp(1e-4, 1.0 - 1e-4)
    d_valid = depth_valid.to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)
    feat_parts = [
        frames.permute(0, 3, 1, 2).clamp(0.0, 1.0) * mask.permute(0, 3, 1, 2),
        mask.permute(0, 3, 1, 2),
        d_frac[:, None],
        d_valid[:, None],
    ]
    if extra_features is not None:
        feat_parts.append(extra_features.to(device=frames.device, dtype=frames.dtype))
    raw = head(torch.cat(feat_parts, dim=1))
    if rgb_residual_scale > 0:
        rgb_delta = torch.tanh(raw[:, :3]) * float(rgb_residual_scale)
    else:
        rgb_delta = raw[:, :3] * 0.0
    if depth_delta_scale > 0:
        depth_delta = torch.tanh(raw[:, 3:4]) * float(depth_delta_scale)
    else:
        depth_delta = raw[:, 3:4] * 0.0
    refined_frames = (
        frames + rgb_delta.permute(0, 2, 3, 1) * mask
    ).clamp(0.0, 1.0)
    valid = d_valid if apply_valid is None else apply_valid.to(
        device=frames.device, dtype=frames.dtype
    ).clamp(0.0, 1.0)
    refined_frac = torch.sigmoid(torch.logit(d_frac) + depth_delta[:, 0])
    refined_frac = torch.where(valid > 0.5, refined_frac, d_frac)
    return refined_frames, refined_frac, rgb_delta, depth_delta


class ConditionDepthConfidenceUNet(torch.nn.Module):
    """Shared confidence predictor for estimated conditioning depth.

    The head maps ``[rgb * mask, mask, depth_frac, depth_valid]`` plus optional
    multi-view features to a bounded confidence residual. The final layer is
    zero-initialized, so the initial confidence is controlled entirely by the
    caller's logit bias and can preserve the deterministic prior.
    """

    in_channels = 6

    def __init__(self, hidden: int = 24, in_channels: int | None = None):
        super().__init__()
        self.in_channels = int(in_channels or self.in_channels)
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        self.enc1 = ConditionRGBRefineUNet._block(self.in_channels, hidden)
        self.down1 = ConditionRGBRefineUNet._down(hidden, h2)
        self.mid = ConditionRGBRefineUNet._block(h2, h2)
        self.up1 = ConditionRGBRefineUNet._block(h2 + hidden, hidden)
        self.head = torch.nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"condition-depth-confidence feature bug: got {features.shape[1]} "
                f"channels, expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        x = self.down1(e1)
        x = self.mid(x)
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, e1], dim=1))
        return self.head(x)


def apply_depth_confidence_head(
    head: ConditionDepthConfidenceUNet,
    frames: torch.Tensor,
    masks: torch.Tensor,
    depth_frac: torch.Tensor,
    depth_valid: torch.Tensor,
    init_confidence: float,
    delta_scale: float,
    floor: float,
    extra_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict ``(V,H,W)`` depth confidence in ``[floor,1]``."""
    mask = masks.to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)
    d_frac = depth_frac.to(device=frames.device, dtype=frames.dtype).clamp(1e-4, 1.0 - 1e-4)
    d_valid = depth_valid.to(device=frames.device, dtype=frames.dtype).clamp(0.0, 1.0)
    feat_parts = [
        frames.permute(0, 3, 1, 2).clamp(0.0, 1.0) * mask.permute(0, 3, 1, 2),
        mask.permute(0, 3, 1, 2),
        d_frac[:, None],
        d_valid[:, None],
    ]
    if extra_features is not None:
        feat_parts.append(extra_features.to(device=frames.device, dtype=frames.dtype))
    raw = head(torch.cat(feat_parts, dim=1))
    init = min(max(float(init_confidence), 1e-4), 1.0 - 1e-4)
    bias = torch.logit(raw.new_tensor(init))
    scale = max(float(delta_scale), 0.0)
    delta = scale * torch.tanh(raw) if scale > 0 else raw * 0.0
    prob = torch.sigmoid(bias + delta)[:, 0]
    floor = min(max(float(floor), 0.0), 1.0)
    conf = floor + (1.0 - floor) * prob
    conf = torch.where(d_valid > 0.5, conf, conf.new_zeros(conf.shape))
    return conf.clamp(0.0, 1.0), delta


class ConditionMaskRefineUNet(torch.nn.Module):
    """Shared foreground-mask refiner for latent-decoded conditioning frames.

    The module maps ``[rgb, prior_mask]`` to a bounded residual on the prior
    mask logit. The final layer is zero-initialized, so enabling the head
    preserves the deterministic RGB-derived mask until pretrained weights are
    loaded or it is trained.
    """

    in_channels = 4

    def __init__(self, hidden: int = 32):
        super().__init__()
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        h4 = hidden * 4
        self.enc1 = ConditionRGBRefineUNet._block(self.in_channels, hidden)
        self.down1 = ConditionRGBRefineUNet._down(hidden, h2)
        self.down2 = ConditionRGBRefineUNet._down(h2, h4)
        self.mid = ConditionRGBRefineUNet._block(h4, h4)
        self.up1 = ConditionRGBRefineUNet._block(h4 + h2, h2)
        self.up2 = ConditionRGBRefineUNet._block(h2 + hidden, hidden)
        self.head = torch.nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise RuntimeError(
                f"condition-mask-refine feature bug: got {features.shape[1]} "
                f"channels, expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        e2 = self.down1(e1)
        x = self.down2(e2)
        x = self.mid(x)
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, e1], dim=1))
        return self.head(x)


def apply_mask_refiner(
    head: ConditionMaskRefineUNet,
    frames: torch.Tensor,
    prior_mask: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    """Apply a bounded logit residual to ``(V,H,W,1)`` prior masks."""
    if residual_scale <= 0:
        return prior_mask.clamp(0.0, 1.0)
    prior = prior_mask.to(device=frames.device, dtype=frames.dtype).clamp(1e-4, 1.0 - 1e-4)
    feat = torch.cat([
        frames.permute(0, 3, 1, 2).clamp(0.0, 1.0),
        prior.permute(0, 3, 1, 2),
    ], dim=1)
    delta = torch.tanh(head(feat)) * float(residual_scale)
    refined = torch.sigmoid(torch.logit(prior.permute(0, 3, 1, 2)) + delta)
    return refined.permute(0, 2, 3, 1).clamp(0.0, 1.0)


class OutputAlphaRefineUNet(torch.nn.Module):
    """Target-view alpha cleanup head for rendered RGB/alpha.

    The head predicts a bounded logit residual for an opacity gate. It is
    intentionally one-sided: the applied gate can only reduce the current alpha,
    so it can suppress shell spray without inventing new foreground coverage or
    averaging RGB across candidate views.
    """

    in_channels = 10

    def __init__(self, hidden: int = 16):
        super().__init__()
        hidden = max(int(hidden), 8)
        h2 = hidden * 2
        self.enc1 = self._block(self.in_channels, hidden)
        self.down1 = self._down(hidden, h2)
        self.mid = self._block(h2, h2)
        self.up1 = self._block(h2 + hidden, hidden)
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
                f"output-alpha-refine feature bug: got {features.shape[1]} "
                f"channels, expected {self.in_channels}"
            )
        e1 = self.enc1(features)
        x = self.down1(e1)
        x = self.mid(x)
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, e1], dim=1))
        return self.head(x)


def apply_output_alpha_refiner(
    head: OutputAlphaRefineUNet,
    render: torch.Tensor,
    alpha: torch.Tensor,
    bg: float | torch.Tensor,
    delta_scale: float,
    init: float = 0.995,
    floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a target-view opacity gate to one rendered image.

    ``render`` is assumed to already be composited over scalar background
    ``bg``. The function reconstructs object RGB, gates alpha, and recomposites
    over the same background so object color is unchanged wherever the gate is
    one. Returns ``(render, alpha, gate, delta)``.
    """
    if delta_scale <= 0:
        gate = alpha.new_ones(alpha.shape)
        delta = alpha.new_zeros(alpha.shape)
        return render, alpha, gate, delta
    if render.ndim != 3 or render.shape[-1] != 3:
        raise RuntimeError("render must have shape (H,W,3)")
    if alpha.ndim != 3 or alpha.shape[-1] != 1:
        raise RuntimeError("alpha must have shape (H,W,1)")
    bg_t = torch.as_tensor(bg, device=render.device, dtype=render.dtype)
    h, w = alpha.shape[:2]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=render.device, dtype=render.dtype),
        torch.linspace(-1.0, 1.0, w, device=render.device, dtype=render.dtype),
        indexing="ij",
    )
    alpha_c = alpha.clamp(0.0, 1.0)
    obj = ((render - bg_t * (1.0 - alpha_c)) / alpha_c.clamp_min(1e-4)).clamp(0.0, 1.0)
    feat = torch.cat([
        render.permute(2, 0, 1),
        obj.permute(2, 0, 1),
        alpha_c.permute(2, 0, 1),
        xx[None],
        yy[None],
        bg_t.expand(1, h, w),
    ], dim=0)[None]
    raw = head(feat)
    delta = torch.tanh(raw) * float(delta_scale)
    init_c = min(max(float(init), 1e-4), 1.0 - 1e-4)
    prior = torch.logit(delta.new_tensor(init_c))
    gate = torch.sigmoid(prior + delta).permute(0, 2, 3, 1)[0]
    floor_c = min(max(float(floor), 0.0), 1.0)
    gate = floor_c + (1.0 - floor_c) * gate
    alpha_out = alpha_c * gate
    render_out = obj * alpha_out + bg_t * (1.0 - alpha_out)
    return render_out.clamp(0.0, 1.0), alpha_out.clamp(0.0, 1.0), gate, delta
