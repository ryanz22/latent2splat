"""Method D (GS-LRM activation shifts) tests: shapes, ranges, and crucially the
INIT values — opacity/scale must start at moderate NONZERO values (the whole
point of the anti-collapse shifts)."""
from __future__ import annotations

import torch

from legacy.decoder_d import PixelAlignedDecoderD
from legacy.decoder_a import LATENT_H, LATENT_W
from legacy.tests.test_decoder_c import _frame0_camera


def test_decoder_d_shapes_and_count():
    K, c2w, radius = _frame0_camera()
    dec = PixelAlignedDecoderD(dim=64, depth_layers=2, heads=4, k=27, radius=radius)
    out = dec(torch.randn(1, 128, 2, LATENT_H, LATENT_W), K, c2w)
    n = LATENT_H * LATENT_W * 27  # 10,368
    assert out["mean"].shape == (1, n, 3)
    assert n == 10368


def test_decoder_d_ranges():
    K, c2w, radius = _frame0_camera()
    dec = PixelAlignedDecoderD(dim=64, depth_layers=2, heads=4, k=4, radius=radius)
    out = dec(torch.randn(2, 128, 2, LATENT_H, LATENT_W), K, c2w)
    assert (out["scale"] > 0).all() and (out["scale"] <= 0.3 + 1e-6).all()  # capped
    assert (out["opacity"] >= 0).all() and (out["opacity"] <= 1).all()
    assert (out["rgb"] >= 0).all() and (out["rgb"] <= 1).all()
    q = out["quat"].norm(dim=-1)
    assert torch.allclose(q, torch.ones_like(q), atol=1e-5)
    # means lie within the depth shell along rays
    cam = c2w[:3, 3]
    d = (out["mean"][0] - cam[None]).norm(dim=-1)
    assert (d >= dec.d_near - 1e-3).all() and (d <= dec.d_far + 1e-3).all()


def test_decoder_d_init_is_visible_not_saturated():
    """The anti-collapse property: at init (zero latent -> raw~0 via N(0,.02)
    weights), opacity ~ sigmoid(-2) ~ 0.12 and scale ~ exp(-2.3) ~ 0.10 —
    moderate, nonzero, gradient-live. NOT ~0 (would collapse) or ~1 (saturated)."""
    K, c2w, radius = _frame0_camera()
    dec = PixelAlignedDecoderD(dim=64, depth_layers=2, heads=4, k=4, radius=radius)
    out = dec(torch.zeros(1, 128, 2, LATENT_H, LATENT_W), K, c2w)
    op = out["opacity"].mean().item()
    sc = out["scale"].mean().item()
    assert 0.05 < op < 0.30, f"init opacity {op} should be moderate (~0.12)"
    assert 0.03 < sc < 0.20, f"init scale {sc} should be moderate (~0.10)"
    # rgb starts ~mid-gray (0.5*tanh(~0)+0.5 = 0.5)
    assert abs(out["rgb"].mean().item() - 0.5) < 0.1
