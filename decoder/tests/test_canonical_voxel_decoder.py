from __future__ import annotations

import torch

from decoder.clean.canonical_voxel_decoder import CanonicalVoxelDecoder


def _camera(v: int, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    K = torch.tensor(
        [[80.0, 0.0, w / 2.0], [0.0, 80.0, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).repeat(v, 1, 1)
    c2w = torch.eye(4, dtype=torch.float32).repeat(v, 1, 1)
    c2w[:, 2, 3] = 2.0
    for i in range(v):
        c2w[i, 0, 3] = 0.04 * i
    return K, c2w


def test_canonical_voxel_decoder_shapes_and_init():
    torch.manual_seed(2)
    v, h, w = 3, 18, 14
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    masks[:, :4] = 0.0
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = CanonicalVoxelDecoder(
        hidden=64,
        layers=2,
        heads=4,
        grid_h=5,
        grid_w=4,
        latent_pool=1,
        voxel_size_frac=0.02,
        max_voxels=128,
        opacity_init=0.8,
    )

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)

    n = out["mean"].shape[0]
    assert 0 < n <= 128
    assert out["quat"].shape == (n, 4)
    assert out["scale"].shape == (n, 3)
    assert out["opacity"].shape == (n, 1)
    assert out["rgb"].shape == (n, 3)
    assert out["mean_anchor"].shape == (n, 3)
    assert torch.allclose(out["mean"], out["mean_anchor"], atol=1e-6)
    assert torch.allclose(out["mean_offset"], torch.zeros_like(out["mean_offset"]), atol=1e-6)
    assert torch.isfinite(out["opacity"]).all()
    assert out["opacity"].min() >= 0.0
    assert out["opacity"].max() <= 0.8 + 1e-5


def test_canonical_voxel_decoder_has_gradient_path():
    torch.manual_seed(3)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.25)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = CanonicalVoxelDecoder(
        hidden=48,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        latent_pool=1,
        voxel_size_frac=0.03,
        max_voxels=64,
    )

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    loss = out["rgb"].mean() + out["opacity"].mean() + out["mean"].square().mean()
    loss.backward()

    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )


def test_canonical_voxel_decoder_detail_sampling_path():
    torch.manual_seed(4)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.25)
    K, c2w = _camera(v, h, w)
    w2c = torch.eye(4, dtype=torch.float32).repeat(v, 1, 1)
    latent = torch.randn(128, 2, 6, 4)
    model = CanonicalVoxelDecoder(
        hidden=48,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        latent_pool=1,
        voxel_size_frac=0.03,
        max_voxels=64,
        detail_sampling=True,
        detail_chunk=8,
    )

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0, w2c=w2c)
    loss = out["rgb"].mean() + out["opacity"].mean()
    loss.backward()

    assert out["rgb"].shape[0] > 0
    assert model.detail_score.weight.grad is not None
    assert torch.isfinite(model.detail_score.weight.grad).all()


def test_canonical_voxel_decoder_max_learned_paths():
    torch.manual_seed(5)
    v, h, w = 3, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.25)
    K, c2w = _camera(v, h, w)
    w2c = torch.inverse(c2w)
    latent = torch.randn(128, 2, 6, 4)
    model = CanonicalVoxelDecoder(
        hidden=64,
        layers=2,
        heads=4,
        latent_layers=1,
        scene_slots=6,
        grid_h=4,
        grid_w=3,
        latent_pool=1,
        voxel_size_frac=0.03,
        max_voxels=64,
        gaussians_per_voxel=3,
        detail_sampling=True,
        detail_chunk=8,
        view_feature_channels=12,
        view_feature_scale=0.5,
        opacity_prior_weight=0.5,
    )

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0, w2c=w2c)
    loss = out["rgb"].mean() + out["opacity"].mean() + out["mean"].square().mean()
    loss.backward()

    assert out["mean"].shape[0] > 0
    assert out["mean"].shape[0] % 3 == 0
    assert model.scene_tokens.grad is not None
    assert model.view_feature_encoder.in_proj[0].weight.grad is not None


def test_canonical_source_consistency_refiner_is_trainable_zero_init():
    torch.manual_seed(6)
    v, h, w = 3, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.25)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = CanonicalVoxelDecoder(
        hidden=48,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        latent_pool=1,
        voxel_size_frac=0.03,
        max_voxels=64,
        source_consistency_refine=True,
        source_consistency_hidden=32,
    )

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    n = out["opacity"].shape[0]
    scored = dict(out)
    scored["_fusion_support"] = torch.full((n, 1), 2.0)
    scored["_fusion_conflict"] = torch.zeros(n, 1)
    scored["_fusion_coverage"] = torch.full((n, 1), 3.0)
    refined = model.refine_source_consistency(scored, radius=2.0)

    assert torch.allclose(refined["opacity"], out["opacity"], atol=1e-6)
    assert torch.allclose(refined["rgb"], out["rgb"], atol=1e-6)
    loss = refined["opacity"].mean() + refined["rgb"].mean() + refined["scale"].mean()
    loss.backward()
    assert model.source_consistency_refine[-1].weight.grad is not None
    assert torch.isfinite(model.source_consistency_refine[-1].weight.grad).all()
