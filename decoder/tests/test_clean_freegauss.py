"""Ceiling-check free-Gaussian fit (Method A): the clean decoder's EXACT output form
with the latent + network removed — a learnable raw map optimized directly. Tests the
free-parameter setup + grad flow on CPU; a GPU render smoke test is guarded."""
from __future__ import annotations

import torch
import pytest

from decoder.clean.fit_freegauss import free_gaussian_map, LH, LW
from decoder.clean.gaussians import activate, B_ALPHA


def _cam():
    K = torch.tensor([[840.0, 0, 256.0], [0, 840.0, 384.0], [0, 0, 1.0]])
    c2w = torch.eye(4); c2w[2, 3] = 1.52
    return K, c2w


def test_free_map_shapes_match_ups_stages():
    K, c2w = _cam()
    raw, (origins, dirs, dn, df) = free_gaussian_map(K, c2w, 1.52, ups_stages=2)
    n = (LH * 4) * (LW * 4)                       # 96*64 = 6144
    assert raw.shape == (n, 12) and raw.requires_grad
    assert origins.shape == (n, 3) and dirs.shape == (n, 3)
    assert 0.0 < dn < df


def test_free_map_ups5_is_393k():
    K, c2w = _cam()
    raw, _ = free_gaussian_map(K, c2w, 1.52, ups_stages=5)
    assert raw.shape[0] == 393_216               # (24*32)*(16*32) = 768*512


def test_free_map_seed_reproducible():
    K, c2w = _cam()
    a, _ = free_gaussian_map(K, c2w, 1.52, ups_stages=2, seed=7)
    b, _ = free_gaussian_map(K, c2w, 1.52, ups_stages=2, seed=7)
    c, _ = free_gaussian_map(K, c2w, 1.52, ups_stages=2, seed=8)
    assert torch.equal(a.detach(), b.detach())
    assert not torch.equal(a.detach(), c.detach())


def test_gradients_flow_to_raw_through_activate():
    K, c2w = _cam()
    raw, (origins, dirs, dn, df) = free_gaussian_map(K, c2w, 1.52, ups_stages=2)
    p = activate(raw, origins, dirs, dn, df, 1.52, 0.012)
    loss = (p["mean"].mean() + p["rgb"].mean() + p["scale"].mean()
            + p["opacity"].mean() + p["quat"].pow(2).mean() + p["depth"].mean())
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert (raw.grad.abs() > 0).any()            # the free params actually receive gradient


def test_init_matches_decoder_start():
    # raw≈0 → activate gives the decoder's init: opacity≈sigmoid(B_ALPHA), rgb≈0.5
    K, c2w = _cam()
    raw, (origins, dirs, dn, df) = free_gaussian_map(K, c2w, 1.52, ups_stages=2, seed=0)
    p = activate(raw.detach(), origins, dirs, dn, df, 1.52, 0.012)
    assert abs(float(p["opacity"].mean()) - torch.sigmoid(torch.tensor(B_ALPHA)).item()) < 0.01
    assert abs(float(p["rgb"].mean()) - 0.5) < 0.02


def test_free_fit_step_runs_on_gpu():
    import os
    pytest.importorskip("gsplat")
    # gsplat is toolkit-disabled (and errors at call time) unless CUDA_HOME is on PATH,
    # so skip cleanly under a plain `pytest`; run this with the gsplat GPU env (CUDA_HOME set).
    if not os.environ.get("CUDA_HOME") or not torch.cuda.is_available():
        pytest.skip("free-fit GPU smoke needs the gsplat CUDA env (CUDA_HOME + CUDA)")
    from decoder.render import render_views
    from decoder.data import opengl_c2w_to_opencv_w2c
    K, c2w = _cam()
    K, c2w = K.cuda(), c2w.cuda()
    raw, (origins, dirs, dn, df) = free_gaussian_map(K, c2w, 1.52, ups_stages=2, device="cuda")
    p = activate(raw, origins, dirs, dn, df, 1.52, 0.012)
    w2c = opengl_c2w_to_opencv_w2c(c2w)[None]
    render, alpha = render_views(p, w2c, K[None], 512, 768, bg=1.0)
    # an L1 to a dummy target exercises the full free-raw → activate → render → grad path
    loss = (render - 0.5).abs().mean() + (alpha - 0.5).abs().mean()
    loss.backward()
    assert torch.isfinite(render).all() and torch.isfinite(alpha).all()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert (raw.grad.abs() > 0).any()
