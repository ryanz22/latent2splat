import torch

from decoder.clean.condition_refine import (
    ConditionDepthConfidenceUNet,
    ConditionRGBDRefineUNet,
    ConditionRGBDViewRefineUNet,
    OutputAlphaRefineUNet,
    apply_depth_confidence_head,
    apply_output_alpha_refiner,
    apply_rgbd_refiner,
)


def test_output_alpha_refiner_zero_head_applies_init_gate_and_preserves_color():
    head = OutputAlphaRefineUNet(hidden=8)
    render = torch.rand(6, 5, 3)
    alpha = torch.rand(6, 5, 1).clamp_min(0.1)
    bg = 0.25

    out, alpha_out, gate, delta = apply_output_alpha_refiner(
        head, render, alpha, bg, delta_scale=10.0, init=0.995
    )

    assert torch.allclose(delta, torch.zeros_like(delta))
    assert torch.allclose(gate, torch.full_like(gate, 0.995), atol=1e-6)
    assert torch.allclose(alpha_out, alpha * 0.995, atol=1e-6)

    obj_in = ((render - bg * (1.0 - alpha)) / alpha).clamp(0.0, 1.0)
    obj_out = ((out - bg * (1.0 - alpha_out)) / alpha_out.clamp_min(1e-4)).clamp(0.0, 1.0)
    assert torch.allclose(obj_out, obj_in, atol=1e-5)


def test_output_alpha_refiner_disabled_is_identity():
    head = OutputAlphaRefineUNet(hidden=8)
    render = torch.rand(4, 3, 3)
    alpha = torch.rand(4, 3, 1)

    out, alpha_out, gate, delta = apply_output_alpha_refiner(
        head, render, alpha, bg=1.0, delta_scale=0.0
    )

    assert torch.equal(out, render)
    assert torch.equal(alpha_out, alpha)
    assert torch.equal(gate, torch.ones_like(alpha))
    assert torch.equal(delta, torch.zeros_like(alpha))


def test_condition_rgbd_refiner_zero_head_preserves_rgb_and_depth():
    head = ConditionRGBDRefineUNet(hidden=8)
    frames = torch.rand(2, 7, 5, 3)
    masks = torch.rand(2, 7, 5, 1)
    depth_frac = torch.rand(2, 7, 5).clamp(0.05, 0.95)
    depth_valid = torch.ones(2, 7, 5)

    rgb, depth, rgb_delta, depth_delta = apply_rgbd_refiner(
        head,
        frames,
        masks,
        depth_frac,
        depth_valid,
        rgb_residual_scale=0.2,
        depth_delta_scale=0.5,
    )

    assert torch.allclose(rgb, frames)
    assert torch.allclose(depth, depth_frac)
    assert torch.count_nonzero(rgb_delta) == 0
    assert torch.count_nonzero(depth_delta) == 0


def test_condition_rgbd_refiner_checks_feature_count_with_extra_features():
    head = ConditionRGBDRefineUNet(hidden=8, in_channels=10)
    frames = torch.rand(2, 7, 5, 3)
    masks = torch.rand(2, 7, 5, 1)
    depth_frac = torch.rand(2, 7, 5).clamp(0.05, 0.95)
    depth_valid = torch.ones(2, 7, 5)
    extra = torch.rand(2, 4, 7, 5)

    rgb, depth, _, _ = apply_rgbd_refiner(
        head,
        frames,
        masks,
        depth_frac,
        depth_valid,
        rgb_residual_scale=0.2,
        depth_delta_scale=0.5,
        extra_features=extra,
    )

    assert rgb.shape == frames.shape
    assert depth.shape == depth_frac.shape


def test_condition_rgbd_view_refiner_zero_head_preserves_rgb_and_depth():
    head = ConditionRGBDViewRefineUNet(
        hidden=8, in_channels=10, max_views=4, context_layers=1, context_heads=2
    )
    frames = torch.rand(3, 7, 5, 3)
    masks = torch.rand(3, 7, 5, 1)
    depth_frac = torch.rand(3, 7, 5).clamp(0.05, 0.95)
    depth_valid = torch.ones(3, 7, 5)
    extra = torch.rand(3, 4, 7, 5)

    rgb, depth, rgb_delta, depth_delta = apply_rgbd_refiner(
        head,
        frames,
        masks,
        depth_frac,
        depth_valid,
        rgb_residual_scale=0.2,
        depth_delta_scale=0.5,
        extra_features=extra,
    )

    assert torch.allclose(rgb, frames)
    assert torch.allclose(depth, depth_frac)
    assert torch.count_nonzero(rgb_delta) == 0
    assert torch.count_nonzero(depth_delta) == 0


def test_condition_rgbd_view_refiner_rejects_too_many_views():
    head = ConditionRGBDViewRefineUNet(hidden=8, max_views=1)
    x = torch.rand(2, ConditionRGBDViewRefineUNet.in_channels, 7, 5)

    try:
        head(x)
    except RuntimeError as ex:
        assert "max_views" in str(ex)
    else:
        raise AssertionError("expected max_views error")


def test_depth_confidence_head_zero_init_uses_requested_prior_and_valid_mask():
    head = ConditionDepthConfidenceUNet(hidden=8)
    frames = torch.rand(2, 7, 5, 3)
    masks = torch.ones(2, 7, 5, 1)
    depth_frac = torch.rand(2, 7, 5).clamp(0.05, 0.95)
    depth_valid = torch.ones(2, 7, 5)
    depth_valid[1, :2] = 0.0

    conf, delta = apply_depth_confidence_head(
        head,
        frames,
        masks,
        depth_frac,
        depth_valid,
        init_confidence=0.8,
        delta_scale=6.0,
        floor=0.25,
    )

    expected = 0.25 + 0.75 * 0.8
    assert torch.allclose(conf[0], torch.full_like(conf[0], expected), atol=1e-6)
    assert torch.count_nonzero(conf[1, :2]) == 0
    assert torch.count_nonzero(delta) == 0


def test_depth_confidence_head_checks_extra_feature_count():
    head = ConditionDepthConfidenceUNet(hidden=8, in_channels=10)
    frames = torch.rand(2, 7, 5, 3)
    masks = torch.ones(2, 7, 5, 1)
    depth_frac = torch.rand(2, 7, 5).clamp(0.05, 0.95)
    depth_valid = torch.ones(2, 7, 5)
    extra = torch.rand(2, 4, 7, 5)

    conf, _ = apply_depth_confidence_head(
        head,
        frames,
        masks,
        depth_frac,
        depth_valid,
        init_confidence=0.9,
        delta_scale=4.0,
        floor=0.1,
        extra_features=extra,
    )

    assert conf.shape == depth_frac.shape
