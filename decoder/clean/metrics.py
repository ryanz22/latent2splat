"""Image metrics (device-agnostic) shared by the clean decoder trainer (train_clean.py)
and the ceiling-check fit (fit_freegauss.py), so both report identical numbers:

- fg_masked_psnr: honest foreground-masked PSNR (the per-step random-bg PSNR is noise).
- sharpness_ratio: download-free foreground high-frequency-energy ratio (1.0 = as sharp as
  the target, <1 = blurrier). The blur metric PSNR misses.
"""
from __future__ import annotations

import torch


def fg_masked_psnr(render, target, fg):   # render,target (V,H,W,3); fg (V,H,W,1) bool
    sel = fg.expand_as(render)
    mse = ((render - target) ** 2)[sel].mean()
    return float(-10.0 * torch.log10(mse + 1e-8))


def _hf_energy(img, fgm):   # img (V,H,W,3); fgm (V,H,W) bool -> mean |gradient| in FG
    g = img.mean(-1)                                  # (V,H,W) grayscale
    e = (g[:, :-1, :-1] - g[:, :-1, 1:]).abs() + (g[:, :-1, :-1] - g[:, 1:, :-1]).abs()
    m = fgm[:, :-1, :-1]
    return float(e[m].mean()) if m.any() else 0.0


def sharpness_ratio(render, target, fgm):
    """High-frequency-energy ratio in the foreground. 1.0 = as sharp as the target,
    <1 = blurrier (blur = loss of high frequencies). Download-free perceptual proxy."""
    et = _hf_energy(target, fgm)
    return _hf_energy(render, fgm) / et if et > 0 else 0.0
