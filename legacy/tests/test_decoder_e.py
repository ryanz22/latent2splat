"""Method E tests: sigmoid (free-opacity) mode shapes/ranges + that opacity starts
low and is free to deactivate (the route depth-PDF lacks), plus a pdf-mode
regression. Fills the gap of decoder_e (the current arch) having had no test."""
from __future__ import annotations

import torch

from legacy.decoder_e import PixelAlignedDecoderE
from legacy.decoder_a import LATENT_H, LATENT_W
from legacy.tests.test_decoder_c import _frame0_camera


def test_decoder_e_sigmoid_shapes_and_ranges():
    K, c2w, radius = _frame0_camera()
    dec = PixelAlignedDecoderE(dim=64, depth_layers=2, heads=4, up=5,
                               radius=radius, opacity_mode="sigmoid")
    out = dec(torch.randn(2, 128, 2, LATENT_H, LATENT_W), K, c2w)
    n = (LATENT_H * 5) * (LATENT_W * 5)   # 9,600 pixel-aligned Gaussians
    assert out["mean"].shape == (2, n, 3)
    assert (out["opacity"] >= 0).all() and (out["opacity"] <= 1).all()
    assert (out["scale"] >= 0.015 - 1e-6).all() and (out["scale"] <= 0.3 + 1e-6).all()
    assert (out["rgb"] >= 0).all() and (out["rgb"] <= 1).all()
    q = out["quat"].norm(dim=-1)
    assert torch.allclose(q, torch.ones_like(q), atol=1e-5)
    # means lie within the depth shell along frame-0 rays
    cam = c2w[:3, 3]
    d = (out["mean"][0] - cam[None]).norm(dim=-1)
    assert (d >= dec.d_near - 1e-3).all() and (d <= dec.d_far + 1e-3).all()


def test_decoder_e_sigmoid_opacity_starts_low_and_free():
    """Free-opacity property (the deactivation route depth-PDF lacks): sigmoid mode
    uses opacity = sigmoid(raw - 2), so at init (~zero raw) opacity starts ~0.12 and
    can go lower toward 0 — unlike pdf-mode's >= 1/32 max-prob floor that saturates."""
    K, c2w, radius = _frame0_camera()
    dec = PixelAlignedDecoderE(dim=64, depth_layers=2, heads=4, up=2,
                               radius=radius, opacity_mode="sigmoid")
    out = dec(torch.zeros(1, 128, 2, LATENT_H, LATENT_W), K, c2w)
    op = out["opacity"].mean().item()
    assert op < 0.2, f"sigmoid init opacity {op} should start low (~0.12), free to deactivate"


def test_decoder_e_pdf_mode_regression():
    """pdf mode (the prior default) still produces valid opacity in [0,1]."""
    K, c2w, radius = _frame0_camera()
    dec = PixelAlignedDecoderE(dim=64, depth_layers=2, heads=4, up=2,
                               radius=radius, opacity_mode="pdf")
    out = dec(torch.randn(1, 128, 2, LATENT_H, LATENT_W), K, c2w)
    n = (LATENT_H * 2) * (LATENT_W * 2)
    assert out["opacity"].shape == (1, n, 1)
    assert (out["opacity"] >= 0).all() and (out["opacity"] <= 1).all()
