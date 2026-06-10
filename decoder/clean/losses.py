"""Loss terms specific to the clean-slate decoder: alpha-mask L1, scale-invariant
reference depth, scale hinge, and an expected-depth render via gsplat."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_alpha_l1(alpha: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """L1 between rendered accumulated alpha and the GT silhouette (both (V,H,W,1))."""
    return F.l1_loss(alpha, gt_mask)


def scale_invariant_depth_loss(pred: torch.Tensor, target: torch.Tensor,
                               valid: torch.Tensor) -> torch.Tensor:
    """Scale-invariant log-depth loss (Eigen) over valid pixels. Robust to a global
    scale; we pair it with a tiny L1 in the trainer for absolute correctness."""
    if valid.sum() == 0:
        return pred.new_zeros(())
    d = torch.log(pred[valid].clamp_min(1e-6)) - torch.log(target[valid].clamp_min(1e-6))
    return (d ** 2).mean() - d.mean() ** 2


def absolute_depth_loss(pred: torch.Tensor, target: torch.Tensor,
                        valid: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    """Huber depth loss over valid pixels.

    The scale-invariant term gives useful opacity gradients but cannot anchor
    absolute shell placement by itself. This small paired term removes that null
    direction when metric depth is available.
    """
    if valid.sum() == 0:
        return pred.new_zeros(())
    return F.huber_loss(pred[valid], target[valid], delta=delta)


def scale_hinge(scale: torch.Tensor, s_min: float, s_max: float) -> torch.Tensor:
    """Penalize Gaussian scales outside [s_min, s_max] (Splatter-Image small/large reg)."""
    return F.relu(s_min - scale).mean() + F.relu(scale - s_max).mean()


class VGGPerceptual(torch.nn.Module):
    """Frozen VGG16 feature-space L1 — the sharpness lever (GS-LRM uses VGG perceptual
    λ≈0.5; Splatter-Image uses LPIPS). L1+SSIM are mean-seeking and tolerate blur;
    matching deep features penalizes the loss of high-frequency structure that PSNR
    misses. Inputs are images (N,H,W,3) in [0,1]."""
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    SLICES = (4, 9, 16, 23)   # relu1_2, relu2_2, relu3_3, relu4_3 in vgg16.features

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.blocks = torch.nn.ModuleList()
        prev = 0
        for s in self.SLICES:
            self.blocks.append(feats[prev:s]); prev = s
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def forward(self, render: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = (render.permute(0, 3, 1, 2) - self.mean) / self.std
        y = (target.permute(0, 3, 1, 2) - self.mean) / self.std
        loss = x.new_zeros(())
        for blk in self.blocks:
            x, y = blk(x), blk(y)
            loss = loss + F.l1_loss(x, y)
        return loss


def opacity_scale_reg(opacity: torch.Tensor, scale: torch.Tensor,
                      fg: torch.Tensor | None = None, masked: bool = True):
    """3DGS-MCMC L1 sparsity on opacity + scale, adapted to a feed-forward decoder
    (no relocation). Returns (opacity_term, scale_term), both scalars.
    If masked and `fg` (N,) in [0,1] given, the opacity term penalizes ONLY
    background-anchored Gaussians (weight = 1 - fg) so FG opacity isn't crushed."""
    if masked and fg is not None:
        bg = (1.0 - fg).clamp(0.0, 1.0)
        op_term = (opacity.squeeze(-1) * bg).sum() / (bg.sum() + 1e-8)
    else:
        op_term = opacity.mean()
    return op_term, scale.mean()


def opacity_entropy(opacity: torch.Tensor) -> torch.Tensor:
    """Binarization prior: minimized when opacity in {0,1}; no net-down bias."""
    op = opacity.clamp(1e-6, 1.0 - 1e-6)
    return (op * (1.0 - op)).mean()


def render_expected_depth(params: dict, w2c: torch.Tensor, K: torch.Tensor,
                          width: int, height: int, mode: str = "ED") -> torch.Tensor:
    """Alpha-weighted expected depth per pixel, (V,H,W,1).

    gsplat supports ``D`` for accumulated camera-z depth and ``ED`` for expected
    camera-z depth. Phase2Dataset's rendered-depth loss uses raw Blender Z-depth,
    so ``ED`` is the default.
    """
    mode = mode.upper()
    if mode not in {"D", "ED"}:
        raise ValueError("depth render mode must be 'D' or 'ED'")
    from gsplat import rasterization
    depth, _, _ = rasterization(
        means=params["mean"], quats=params["quat"], scales=params["scale"],
        opacities=params["opacity"].squeeze(-1), colors=params["rgb"],
        viewmats=w2c, Ks=K, width=width, height=height, render_mode=mode)
    return depth   # (V,H,W,1)
