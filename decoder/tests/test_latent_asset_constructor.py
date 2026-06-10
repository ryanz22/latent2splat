from __future__ import annotations

import torch

from decoder.clean.latent_asset_constructor import (
    LatentSourcePredictor,
    camera_token_features,
)


def _cameras(batch: int, views: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.eye(3).view(1, 1, 3, 3).repeat(batch, views, 1, 1)
    K[..., 0, 0] = 700.0
    K[..., 1, 1] = 700.0
    K[..., 0, 2] = 384.0
    K[..., 1, 2] = 256.0
    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(batch, views, 1, 1)
    c2w[..., 0, 3] = torch.linspace(-1.0, 1.0, views).view(1, views)
    c2w[..., 2, 3] = 3.0
    radius = torch.ones(batch)
    return K, c2w, radius


def test_camera_token_features_shape_and_finite():
    K, c2w, radius = _cameras(batch=2, views=3)

    feat = camera_token_features(K, c2w, radius)

    assert feat.shape == (2, 3, 16)
    assert torch.isfinite(feat).all()


def test_latent_source_predictor_shapes_and_gradients():
    latent = torch.randn(2, 128, 2, 6, 4, requires_grad=True)
    K, c2w, radius = _cameras(batch=2, views=3)
    model = LatentSourcePredictor(
        hidden=16,
        out_scale=2,
        depth_bins=8,
        feature_channels=5,
        blocks=1,
    )

    out = model(latent, K, c2w, radius)

    assert out["rgb_delta"].shape == (2, 3, 3, 12, 8)
    assert out["mask_logit"].shape == (2, 3, 1, 12, 8)
    assert out["depth_logits"].shape == (2, 3, 8, 12, 8)
    assert out["confidence_logit"].shape == (2, 3, 1, 12, 8)
    assert out["features"].shape == (2, 3, 5, 12, 8)

    loss = sum(v.square().mean() for v in out.values())
    loss.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_latent_source_predictor_zero_init_outputs_are_zero():
    latent = torch.randn(1, 128, 2, 6, 4)
    K, c2w, radius = _cameras(batch=1, views=2)
    model = LatentSourcePredictor(
        hidden=16,
        out_scale=2,
        depth_bins=4,
        feature_channels=3,
        blocks=1,
    )

    out = model(latent, K, c2w, radius)

    for value in out.values():
        assert torch.count_nonzero(value.detach()) == 0

