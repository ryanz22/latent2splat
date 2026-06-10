"""Depth-supervision utils: Z-pass -> ray-distance conversion + grid target.
Pure functions (no dataset/GPU needed)."""
from __future__ import annotations

import math
import torch

from decoder.data import zdepth_to_raydist, depth_target_on_grid


def _K(fx=840.5, fy=840.5, cx=256.0, cy=384.0):
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def test_zdepth_to_raydist_center_is_one_and_corner_is_1p14():
    H, W = 768, 512
    t = zdepth_to_raydist(torch.ones(H, W), _K())
    # center pixel (col=cx, row=cy): factor == 1
    assert abs(t[384, 256].item() - 1.0) < 1e-3
    # corner (0,0): matches analytic factor, ~1.14 at this fov/resolution
    f = math.sqrt((256 / 840.5) ** 2 + (384 / 840.5) ** 2 + 1)
    assert abs(t[0, 0].item() - f) < 1e-3
    assert 1.10 < t[0, 0].item() < 1.18


def test_zdepth_scales_linearly_with_Z():
    K = _K()
    t1 = zdepth_to_raydist(torch.full((768, 512), 2.0), K)
    t2 = zdepth_to_raydist(torch.full((768, 512), 4.0), K)
    assert torch.allclose(t2, 2 * t1, atol=1e-4)


def test_depth_target_on_grid_ignores_background():
    H, W = 768, 512
    z = torch.full((H, W), 1e10)          # all background sentinel
    mask = torch.zeros(H, W)
    z[300:500, 200:300] = 1.5             # a foreground block at Z=1.5
    mask[300:500, 200:300] = 1.0
    target, valid = depth_target_on_grid(z, mask, _K(), grid_h=120, grid_w=80)
    assert target.shape == (120 * 80,) and valid.shape == (120 * 80,)
    assert valid.any() and (~valid).any()              # some fg cells, some bg cells
    tv = target[valid]
    # ray distance >= Z (factor >= 1) and within the modest corner factor
    assert (tv >= 1.5 - 1e-3).all() and (tv < 1.5 * 1.2).all()


def test_depth_target_zero_loss_when_pred_equals_target():
    H, W = 768, 512
    z = torch.full((H, W), 1e10); mask = torch.zeros(H, W)
    z[200:600, 100:400] = 2.0; mask[200:600, 100:400] = 1.0
    target, valid = depth_target_on_grid(z, mask, _K(), 120, 80)
    pred = target.clone()
    loss = torch.nn.functional.huber_loss(pred[valid], target[valid])
    assert loss.item() < 1e-8
