"""Pure-CPU image metrics shared by the clean decoder trainer and the ceiling-check
fit: foreground-masked PSNR and a download-free foreground sharpness ratio.
Characterization tests — they pin the behavior currently living in train_clean.py."""
from __future__ import annotations

import torch

from decoder.clean.metrics import fg_masked_psnr, sharpness_ratio, _hf_energy


def test_fg_masked_psnr_equal_is_80():
    x = torch.rand(2, 8, 8, 3)
    fg = torch.ones(2, 8, 8, 1, dtype=torch.bool)
    assert abs(fg_masked_psnr(x, x.clone(), fg) - 80.0) < 1e-4   # mse=0 → -10·log10(1e-8)


def test_fg_masked_psnr_known_value():
    target = torch.zeros(1, 4, 4, 3)
    render = torch.full((1, 4, 4, 3), 0.1)          # constant 0.1 error everywhere
    fg = torch.ones(1, 4, 4, 1, dtype=torch.bool)
    assert abs(fg_masked_psnr(render, target, fg) - 20.0) < 1e-3   # mse=0.01 → 20 dB


def test_fg_masked_psnr_only_counts_foreground():
    target = torch.zeros(1, 2, 2, 3)
    render = torch.zeros(1, 2, 2, 3)
    render[0, 0, 0, :] = 0.1                          # error only at pixel (0,0)
    fg = torch.zeros(1, 2, 2, 1, dtype=torch.bool)
    fg[0, 0, 0, 0] = True                             # mask selects only that pixel
    assert abs(fg_masked_psnr(render, target, fg) - 20.0) < 1e-3   # bg errors ignored


def test_hf_energy_flat_is_zero():
    img = torch.full((1, 5, 5, 3), 0.3)
    fgm = torch.ones(1, 5, 5, dtype=torch.bool)
    assert _hf_energy(img, fgm) == 0.0


def test_hf_energy_width_ramp_is_one():
    # g[v,i,j] = j → |horizontal diff|=1, |vertical diff|=0 → mean energy 1.0
    W = 6
    col = torch.arange(W, dtype=torch.float32)
    img = col.view(1, 1, W, 1).expand(1, 5, W, 3).contiguous()
    fgm = torch.ones(1, 5, W, dtype=torch.bool)
    assert abs(_hf_energy(img, fgm) - 1.0) < 1e-5


def test_hf_energy_empty_mask_is_zero():
    img = torch.rand(1, 5, 5, 3)
    fgm = torch.zeros(1, 5, 5, dtype=torch.bool)
    assert _hf_energy(img, fgm) == 0.0


def test_sharpness_ratio_equal_is_one():
    col = torch.arange(6, dtype=torch.float32)
    img = col.view(1, 1, 6, 1).expand(1, 5, 6, 3).contiguous()
    fgm = torch.ones(1, 5, 6, dtype=torch.bool)
    assert abs(sharpness_ratio(img, img.clone(), fgm) - 1.0) < 1e-5


def test_sharpness_ratio_flat_render_is_zero():
    col = torch.arange(6, dtype=torch.float32)
    target = col.view(1, 1, 6, 1).expand(1, 5, 6, 3).contiguous()
    render = torch.full((1, 5, 6, 3), 0.5)            # flat → no HF energy
    fgm = torch.ones(1, 5, 6, dtype=torch.bool)
    assert sharpness_ratio(render, target, fgm) == 0.0


def test_sharpness_ratio_flat_target_is_zero():
    target = torch.full((1, 5, 6, 3), 0.5)            # et=0 → guarded → 0.0
    render = torch.rand(1, 5, 6, 3)
    fgm = torch.ones(1, 5, 6, dtype=torch.bool)
    assert sharpness_ratio(render, target, fgm) == 0.0
