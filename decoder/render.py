"""gsplat rasterization wrapper + photometric loss.

Renders a single object's Gaussians from all V cameras at once via gsplat's
batched `rasterization`. Background is composited to a configurable color to
match the dataset renders. animals_v1 uses BLACK [0,0,0]; the old
objaverse100_v1 used mid-gray 0.5. Pass `bg=` to match the data.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from gsplat import rasterization

BG_DEFAULT = 0.0  # animals_v1 black background


def render_views(
    params: dict,
    w2c: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    bg: float = BG_DEFAULT,
    eps2d: float = 0.3,
    rasterize_mode: str = "classic",
    sh_degree: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render Gaussians from V cameras.

    params: dict with mean (N,3), quat (N,4), scale (N,3), opacity (N,1), rgb (N,3)
    w2c: (V,4,4)  K: (V,3,3)  bg: background gray level (animals_v1 = 0.0 black)
    returns: (rgb (V,H,W,3) in [0,1], alpha (V,H,W,1) in [0,1])
    """
    use_sh = sh_degree is not None and sh_degree >= 0 and "rgb_sh" in params
    colors_in = params["rgb_sh"] if use_sh else params["rgb"]
    colors, alphas, _ = rasterization(
        means=params["mean"],
        quats=params["quat"],
        scales=params["scale"],
        opacities=params["opacity"].squeeze(-1),
        colors=colors_in,
        viewmats=w2c,
        Ks=K,
        width=width,
        height=height,
        eps2d=eps2d,
        rasterize_mode=rasterize_mode,
        sh_degree=sh_degree if use_sh else None,
        render_mode="RGB",
    )
    # Composite onto the dataset's background color. Done manually (via the
    # returned alpha) rather than a `backgrounds=` kwarg for gsplat-version
    # robustness. bg=0 (black) is a no-op add but kept explicit for clarity.
    out = colors + (1.0 - alphas) * bg
    return out.clamp(0.0, 1.0), alphas


def _ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean SSIM over a batch of (V,H,W,3) images, 11x11 uniform window."""
    a = a.permute(0, 3, 1, 2)
    b = b.permute(0, 3, 1, 2)
    win = 11
    pad = win // 2
    mu_a = F.avg_pool2d(a, win, 1, pad)
    mu_b = F.avg_pool2d(b, win, 1, pad)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.avg_pool2d(a * a, win, 1, pad) - mu_a2
    sb = F.avg_pool2d(b * b, win, 1, pad) - mu_b2
    sab = F.avg_pool2d(a * b, win, 1, pad) - mu_ab
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return ssim_map.mean()


def foreground_mask(target: torch.Tensor, bg: float = BG_DEFAULT, tol: float = 0.04) -> torch.Tensor:
    """(V,H,W,3) target -> (V,H,W,1) foreground mask: 1 where the pixel
    differs from the flat background `bg` by more than `tol` in any channel.
    Free silhouette supervision (the renders are object-on-flat-bg).
    animals_v1 is black bg (bg=0), so any non-near-black pixel is foreground."""
    dev = (target - bg).abs().amax(dim=-1, keepdim=True)
    return (dev > tol).float()


def photometric_loss(
    render: torch.Tensor,
    target: torch.Tensor,
    params: dict,
    alpha: torch.Tensor | None = None,
    bg: float = BG_DEFAULT,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.2,
    mask_weight: float = 1.0,
    scale_reg_weight: float = 0.01,
    fg_weight: float = 1.0,
    opacity_reg_weight: float = 0.0,
    fg_mask: torch.Tensor | None = None,
    opacity_reg_mask: torch.Tensor | None = None,
    loss_type: str = "l1",
) -> tuple[torch.Tensor, dict]:
    """Foreground-weighted L1 + (1-SSIM) + silhouette/alpha loss + scale reg.

    `fg_weight` upweights the per-pixel L1 inside the object silhouette. The
    target is ~96% background (e.g. black) and ~4% object, so a plain mean L1
    is dominated by 'match the background' — the optimizer satisfies it by
    shrinking/dimming all Gaussians until the render is uniform background (the
    trivial PSNR-18.46 collapse, confirmed by the init probe: init geometry was
    fine; training drove Gaussians off-object). fg_weight>1 lets the object
    signal compete. fg_weight=1 recovers the plain mean.
    """
    # foreground silhouette: exact GT mask (Trial R4+) if provided, else the
    # gray/black-threshold heuristic. Drives both the fg-weighted L1 and the BCE.
    fg = fg_mask if fg_mask is not None else foreground_mask(target, bg=bg)   # (V,H,W,1)
    # ``loss_type='l2'`` aligns the optimization target with PSNR exactly
    # (PSNR ≡ −10·log10(MSE)).  L1 has a documented deletion bias at
    # anti-aliased silhouette edges (prefers sharp 0/1 alpha over soft
    # gradients), causing trained heads to drop PSNR even when L1 decreases.
    if loss_type == "l2":
        if fg_weight != 1.0:
            per_px = ((render - target) ** 2).mean(-1, keepdim=True)
            w = 1.0 + (fg_weight - 1.0) * fg
            l1 = (per_px * w).sum() / (w.sum() + 1e-8)
        else:
            l1 = F.mse_loss(render, target)
    elif fg_weight != 1.0:
        per_px = (render - target).abs().mean(-1, keepdim=True)  # (V,H,W,1)
        w = 1.0 + (fg_weight - 1.0) * fg                 # fg pixels weighted up
        l1 = (per_px * w).sum() / (w.sum() + 1e-8)
    else:
        l1 = F.l1_loss(render, target)
    ssim = _ssim(render, target)
    reg = params["scale"].mean()
    # opacity reg: masked (penalize ONLY background-anchored Gaussians) if a
    # per-Gaussian mask is given, else a global L1 on mean opacity.
    if opacity_reg_mask is not None:
        op = params["opacity"].reshape(-1)
        opacity_reg = (op * opacity_reg_mask).sum() / opacity_reg_mask.sum().clamp_min(1.0)
    else:
        opacity_reg = params["opacity"].mean()
    loss = l1_weight * l1 + ssim_weight * (1.0 - ssim) + scale_reg_weight * reg + opacity_reg_weight * opacity_reg
    mask_loss = torch.tensor(0.0, device=render.device)
    if alpha is not None:
        mask_loss = F.binary_cross_entropy(alpha.clamp(1e-6, 1 - 1e-6), fg)
        loss = loss + mask_weight * mask_loss
    mse = F.mse_loss(render, target).detach()
    psnr = -10.0 * torch.log10(mse + 1e-8)
    return loss, {"l1": l1.detach(), "ssim": ssim.detach(), "psnr": psnr,
                  "mask": mask_loss.detach(), "loss": loss.detach()}


def depth_loss(pred_depth: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
               delta: float = 0.1) -> torch.Tensor:
    """Huber loss between the decoder's predicted per-Gaussian ray distance
    (pred_depth, (N,) or (N,1)) and the GT grid target (N,), over valid grid cells
    only (see data.depth_target_on_grid). Zero if no cell is valid."""
    pred = pred_depth.reshape(-1)
    if valid.sum() == 0:
        return pred.new_zeros(())
    return F.huber_loss(pred[valid], target[valid], delta=delta)
