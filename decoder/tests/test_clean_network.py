"""CleanGSDecoder: shape trace (128,2,24,16) → 98,304 Gaussians, and grad flow.
CPU only (no gsplat)."""
from __future__ import annotations

import torch

from decoder.clean.network import CleanGSDecoder

N_GAUSS = 384 * 256  # 98,304


def _inputs():
    latent = torch.randn(1, 128, 2, 24, 16)
    K = torch.tensor([[840.0, 0, 256.0], [0, 840.0, 384.0], [0, 0, 1.0]])
    c2w = torch.eye(4); c2w[2, 3] = 1.52
    return latent, K, c2w, 1.52


def test_forward_shapes():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8)   # shallow for a fast test
    latent, K, c2w, r = _inputs()
    p = model(latent, K, c2w, r)
    assert p["mean"].shape == (1, N_GAUSS, 3)
    assert p["quat"].shape == (1, N_GAUSS, 4)
    assert p["scale"].shape == (1, N_GAUSS, 3)
    assert p["opacity"].shape == (1, N_GAUSS, 1)
    assert p["rgb"].shape == (1, N_GAUSS, 3)
    assert p["depth"].shape == (1, N_GAUSS, 1)
    assert p["mean_anchor"].shape == (1, N_GAUSS, 3)
    assert p["mean_offset"].shape == (1, N_GAUSS, 3)


def test_optional_mean_offset_head():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, mean_offset_frac=0.05)
    latent, K, c2w, r = _inputs()
    p = model(latent, K, c2w, r)
    assert p["mean"].shape == (1, N_GAUSS, 3)
    assert p["mean_offset"].abs().max() <= 0.05 * r + 1e-6


def test_resize_upsampler_shapes():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, upsample_mode="resize")
    latent, K, c2w, r = _inputs()
    p = model(latent, K, c2w, r)
    assert p["mean"].shape == (1, N_GAUSS, 3)
    assert p["opacity"].shape == (1, N_GAUSS, 1)


def test_latent_skip_shapes():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, latent_skip=True)
    latent, K, c2w, r = _inputs()
    p = model(latent, K, c2w, r)
    assert p["rgb"].shape == (1, N_GAUSS, 3)


def test_coord_inject_shapes():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, coord_inject=True)
    latent, K, c2w, r = _inputs()
    p = model(latent, K, c2w, r)
    assert p["opacity"].shape == (1, N_GAUSS, 1)


def test_image_condition_shapes():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, image_cond_channels=4)
    latent, K, c2w, r = _inputs()
    image_cond = torch.randn(1, 4, 768, 512)
    p = model(latent, K, c2w, r, image_cond=image_cond)
    assert p["rgb"].shape == (1, N_GAUSS, 3)


def test_image_head_skip_shapes():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, image_cond_channels=4,
                           image_head_skip=True, image_scale_frac=0.002,
                           image_geom_residual_scale=0.0)
    latent, K, c2w, r = _inputs()
    image_cond = torch.rand(1, 4, 768, 512)
    p = model(latent, K, c2w, r, image_cond=image_cond)
    assert p["rgb"].shape == (1, N_GAUSS, 3)


def test_image_head_skip_zero_geom_residual_zeros_mean_offsets():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, image_cond_channels=4,
                           image_head_skip=True, mean_offset_frac=0.05,
                           image_geom_residual_scale=0.0)
    latent, K, c2w, r = _inputs()
    image_cond = torch.rand(1, 4, 768, 512)
    p = model(latent, K, c2w, r, image_cond=image_cond)
    assert torch.allclose(p["mean_offset"], torch.zeros_like(p["mean_offset"]), atol=1e-6)
    assert torch.allclose(p["mean"], p["mean_anchor"], atol=1e-6)


def test_explicit_depth_visibility_heads_shapes():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, image_cond_channels=4,
                           image_head_skip=True, explicit_depth_head=True,
                           explicit_visibility_head=True,
                           image_depth_prior_frac=0.25,
                           image_scale_frac=0.001,
                           image_normal_scale_frac=0.0001,
                           image_camera_quat=True,
                           image_geom_residual_scale=0.0)
    latent, K, c2w, r = _inputs()
    image_cond = torch.rand(1, 4, 768, 512)
    p = model(latent, K, c2w, r, image_cond=image_cond)
    assert p["depth"].shape == (1, N_GAUSS, 1)
    assert p["opacity"].shape == (1, N_GAUSS, 1)


def test_image_depth_skip_sets_decoder_depth():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, image_cond_channels=6,
                           image_head_skip=True, image_depth_skip=True,
                           image_depth_residual_scale=0.0,
                           image_geom_residual_scale=0.0)
    latent, K, c2w, r = _inputs()
    image_cond = torch.zeros(1, 6, 768, 512)
    image_cond[:, 3:4] = 1.0       # mask
    image_cond[:, 4:5] = 0.25      # normalized ray depth
    image_cond[:, 5:6] = 1.0       # depth valid
    p = model(latent, K, c2w, r, image_cond=image_cond)
    expected = 0.76 + (2.28 - 0.76) * 0.25
    assert torch.allclose(p["depth"], torch.full_like(p["depth"], expected), atol=1e-4)


def test_image_visibility_skip_sets_decoder_opacity():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, image_cond_channels=7,
                           image_head_skip=True, image_visibility_skip=True,
                           image_opacity_fg=0.8, image_opacity_bg=0.0001,
                           image_opacity_residual_scale=0.0)
    latent, K, c2w, r = _inputs()
    image_cond = torch.zeros(1, 7, 768, 512)
    image_cond[:, :3] = 0.5
    image_cond[:, 3:4] = 1.0       # mask
    image_cond[:, 4:5] = 0.5       # depth placeholder
    image_cond[:, 5:6] = 1.0       # depth valid placeholder
    image_cond[:, 6:7] = 0.25      # visibility
    p = model(latent, K, c2w, r, image_cond=image_cond)
    expected = 0.0001 + (0.8 - 0.0001) * 0.25
    assert torch.allclose(p["opacity"], torch.full_like(p["opacity"], expected), atol=1e-4)


def test_image_normal_quat_sets_surface_orientation():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8, image_cond_channels=9,
                           image_head_skip=True, image_normal_quat=True,
                           image_geom_residual_scale=0.0)
    latent, K, c2w, r = _inputs()
    image_cond = torch.zeros(1, 9, 768, 512)
    image_cond[:, 3:4] = 1.0       # mask
    image_cond[:, 4:5] = 0.5       # depth placeholder
    image_cond[:, 5:6] = 1.0       # depth valid placeholder
    image_cond[:, -1:] = 1.0       # world normal +Z
    p = model(latent, K, c2w, r, image_cond=image_cond)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4)
    assert torch.allclose(p["quat"], expected.expand_as(p["quat"]), atol=1e-4)


def test_gradients_flow():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8)
    latent, K, c2w, r = _inputs()
    p = model(latent, K, c2w, r)
    loss = p["mean"].mean() + p["opacity"].mean() + p["rgb"].mean() + p["scale"].mean()
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert all(g is not None for g in grads)
    assert sum(float((g.abs() > 0).any()) for g in grads) > 0   # some nonzero grad
