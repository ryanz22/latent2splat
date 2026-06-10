"""Per-attribute activation (GS-LRM/3DGS/Splatter-Image) + opacity pruning.
Pure CPU tensor ops."""
from __future__ import annotations

import torch

from decoder.clean.gaussians import activate, prune_by_opacity, soft_cap_scale, B_ALPHA, SCALE_CAP_FRAC


def _rays(n=100):
    origins = torch.zeros(n, 3)
    dirs = torch.tensor([0.0, 0.0, -1.0]).expand(n, 3).contiguous()
    return origins, dirs


def test_activation_shapes_and_bounds():
    n, radius, dn, df = 100, 1.5, 0.7, 2.3
    raw = torch.zeros(n, 12)                       # raw=0 → test init biases
    origins, dirs = _rays(n)
    p = activate(raw, origins, dirs, dn, df, radius)
    assert p["mean"].shape == (n, 3) and p["quat"].shape == (n, 4)
    assert p["scale"].shape == (n, 3) and p["opacity"].shape == (n, 1) and p["rgb"].shape == (n, 3)
    assert p["depth"].shape == (n, 1)
    assert p["scale_raw"].shape == (n, 3)
    # opacity init: sigmoid(0 + B_ALPHA) ≈ 0.12
    assert abs(float(p["opacity"].mean()) - torch.sigmoid(torch.tensor(B_ALPHA)).item()) < 1e-5
    # scale capped, positive
    assert (p["scale"] > 0).all() and (p["scale"] <= SCALE_CAP_FRAC * radius + 1e-6).all()
    # rgb in [0,1]
    assert (p["rgb"] >= 0).all() and (p["rgb"] <= 1).all()
    # quaternion unit norm
    assert torch.allclose(p["quat"].norm(dim=-1), torch.ones(n), atol=1e-5)
    # depth in [d_near, d_far]; mean = origin + depth*dir
    assert (p["depth"] >= dn - 1e-5).all() and (p["depth"] <= df + 1e-5).all()
    assert torch.allclose(p["mean"], origins + p["depth"] * dirs, atol=1e-5)
    assert torch.allclose(p["mean_offset"], torch.zeros_like(p["mean_offset"]))


def test_soft_scale_cap_keeps_gradient_when_raw_exceeds_cap():
    raw_scale = torch.tensor([10.0], requires_grad=True)
    scale = soft_cap_scale(raw_scale, cap=0.1)
    assert 0.0 < float(scale.detach()) < 0.1
    scale.backward()
    assert raw_scale.grad is not None
    assert float(raw_scale.grad) > 0.0


def test_bounded_mean_offset_relaxes_ray_anchor():
    n, radius, dn, df = 4, 2.0, 0.7, 2.3
    raw = torch.zeros(n, 15)
    raw[:, 12] = 50.0
    origins, dirs = _rays(n)
    p = activate(raw, origins, dirs, dn, df, radius, mean_offset_frac=0.1)
    offset = p["mean"] - p["mean_anchor"]
    assert (offset.abs() <= 0.2 + 1e-6).all()
    assert torch.allclose(offset[:, 0], torch.full((n,), 0.2), atol=1e-5)
    assert not torch.allclose(p["mean"], origins + p["depth"] * dirs)


def test_depth_endpoints():
    origins, dirs = _rays(2)
    raw = torch.zeros(2, 12)
    raw[0, 11] = -50.0   # g_d very negative → t ≈ d_near
    raw[1, 11] = 50.0    # g_d very positive → t ≈ d_far
    p = activate(raw, origins, dirs, 0.7, 2.3, 1.5)
    assert abs(float(p["depth"][0]) - 0.7) < 1e-3
    assert abs(float(p["depth"][1]) - 2.3) < 1e-3


def test_prune_by_opacity():
    p = {"mean": torch.zeros(4, 3), "quat": torch.zeros(4, 4), "scale": torch.ones(4, 3),
         "opacity": torch.tensor([[0.9], [0.001], [0.5], [0.0]]), "rgb": torch.zeros(4, 3),
         "depth": torch.zeros(4, 1)}
    kept = prune_by_opacity(p, thresh=0.005)
    assert kept["opacity"].shape[0] == 2          # only 0.9 and 0.5 survive
    assert torch.allclose(kept["opacity"].flatten(), torch.tensor([0.9, 0.5]))


# --- loss term tests (Task 4 implements decoder/clean/losses.py) ---
from decoder.clean.losses import mask_alpha_l1, scale_invariant_depth_loss, absolute_depth_loss, scale_hinge


def test_mask_alpha_l1_zero_when_equal():
    a = torch.rand(2, 8, 8, 1)
    assert float(mask_alpha_l1(a, a.clone())) == 0.0


def test_scale_invariant_depth_zero_on_shift():
    # SI depth loss is invariant to a global log-scale; equal-up-to-scale → ~0
    pred = torch.tensor([1.0, 2.0, 4.0])
    valid = torch.ones(3, dtype=torch.bool)
    assert scale_invariant_depth_loss(pred * 3.0, pred, valid) < 1e-6


def test_absolute_depth_loss_penalizes_shift():
    pred = torch.tensor([1.0, 2.0, 4.0])
    valid = torch.ones(3, dtype=torch.bool)
    assert absolute_depth_loss(pred * 3.0, pred, valid) > 0


def test_scale_hinge_penalizes_out_of_band():
    s = torch.tensor([[0.001, 0.5, 0.05]])     # one tiny, one huge, one ok
    pen = scale_hinge(s, s_min=0.005, s_max=0.1)
    assert float(pen) > 0
    ok = torch.full((1, 3), 0.05)
    assert float(scale_hinge(ok, s_min=0.005, s_max=0.1)) == 0.0
