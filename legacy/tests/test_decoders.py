"""Decoder output shape + activation-range tests for both architectures."""
from __future__ import annotations

import torch

from legacy.decoder_a import TokenAlignedDecoder, LATENT_H, LATENT_W
from legacy.decoder_b import LearnedQueryDecoder
from legacy.gaussian_head import split_gaussian_params


def _check_param_ranges(p: dict, scale_max: float = 0.3):
    # scale = scale_max * sigmoid(raw) -> bounded in (0, scale_max)
    assert (p["scale"] > 0).all() and (p["scale"] < scale_max + 1e-6).all()
    assert (p["opacity"] >= 0).all() and (p["opacity"] <= 1).all()
    assert (p["rgb"] >= 0).all() and (p["rgb"] <= 1).all()
    norms = p["quat"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_split_gaussian_params_residual_vs_absolute():
    raw = torch.randn(2, 10, 14)
    canon = torch.zeros(2, 10, 3)
    p_res = split_gaussian_params(raw, canonical_mean=canon, offset_max=1.0)
    p_abs = split_gaussian_params(raw, canonical_mean=None, offset_max=1.0)
    # residual to zero canonical == absolute offset
    assert torch.allclose(p_res["mean"], p_abs["mean"])
    assert (p_res["mean"].abs() <= 1.0 + 1e-6).all()
    _check_param_ranges(p_res)


def test_default_opacity_starts_visible():
    """The head's +bias must make a zero-input Gaussian visible (sigmoid>0.5),
    so training does not begin in the opacity-death basin."""
    from legacy.gaussian_head import GaussianHead
    head = GaussianHead(hidden=16, k=4, opacity_bias=2.0)
    raw = head(torch.zeros(1, 16))
    p = split_gaussian_params(raw)
    assert (p["opacity"] > 0.8).all(), p["opacity"]


def test_token_aligned_shapes_and_ranges():
    # v2 default latent_t=2 → tokens = 2·24·16 = 768
    t, k = 2, 8
    dec = TokenAlignedDecoder(dim=64, depth=2, heads=4, k=k, latent_t=t)
    out = dec(torch.randn(2, 128, t, LATENT_H, LATENT_W))
    n = t * LATENT_H * LATENT_W * k
    assert out["mean"].shape == (2, n, 3)
    assert out["opacity"].shape == (2, n, 1)
    _check_param_ranges(out)


def test_token_aligned_handles_v1_temporal_dim():
    # the old 25-frame data gives T=4; the decoder must still build/run
    t, k = 4, 4
    dec = TokenAlignedDecoder(dim=64, depth=2, heads=4, k=k, latent_t=t)
    out = dec(torch.randn(1, 128, t, LATENT_H, LATENT_W))
    assert out["mean"].shape == (1, t * LATENT_H * LATENT_W * k, 3)


def test_learned_query_shapes_and_ranges():
    dec = LearnedQueryDecoder(dim=64, enc_depth=2, dec_depth=2, heads=4,
                              num_queries=256, latent_t=2)
    out = dec(torch.randn(2, 128, 2, LATENT_H, LATENT_W))
    assert out["mean"].shape == (2, 256, 3)
    _check_param_ranges(out)


def test_canonical_grid_within_unit_cube():
    from legacy.decoder_a import _canonical_grid
    g = _canonical_grid(4, 24, 16)
    assert g.shape == (4 * 24 * 16, 3)
    assert g.abs().max() <= 0.5 + 1e-6
