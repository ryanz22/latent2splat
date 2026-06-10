from __future__ import annotations

import torch

from decoder.clean.condition_refine import ConditionDepthAffineHead
from decoder.clean.train_phase2 import (
    AdaptiveLossBalancer,
    DepthRefineUNet,
    FUSION_CANDIDATE_COORD_FEATURE_CHANNELS,
    FUSION_CANDIDATE_FEATURE_CHANNELS,
    FUSION_CANDIDATE_NEIGHBOR_FEATURE_CHANNELS,
    FUSION_CANDIDATE_RICH_FEATURE_CHANNELS,
    FUSION_CANDIDATE_VOXEL_FEATURE_CHANNELS,
    FusionCandidateGate,
    LearnedIblendFillBlend,
    LearnedIblendFillUNet,
    SurfaceConfidenceUNet,
    SurfaceRefineUNet,
    _alpha_mask_stats,
    _alpha_anti_lattice_loss,
    _blend_static_fill,
    _depth_confidence_targets,
    _compose_iblend_anchor,
    _depth_multiview_support_maps,
    _erode_mask_2d,
    _fusion_candidate_features,
    _learned_iblend_feature_channels,
    _local_valid_median_map,
    _normalize_confidence_map,
    _sample_depth_support_window,
    _select_iblend_alpha,
    _select_iblend_object_color,
    _surface_confidence_protect_mask,
)


def test_erode_mask_2d_preserves_interior_and_removes_boundary():
    mask = torch.zeros(1, 5, 5)
    mask[:, 1:4, 1:4] = 1.0

    eroded = _erode_mask_2d(mask, 1)

    assert eroded.sum().item() == 1.0
    assert eroded[0, 2, 2].item() == 1.0
    assert _erode_mask_2d(mask, 0).equal(mask)


def test_adaptive_loss_balancer_preserves_fixed_weight_at_init():
    balancer = AdaptiveLossBalancer(["term"])
    raw = torch.tensor(2.0)

    weighted = balancer("term", raw, base_weight=0.25)
    activated_from_zero = balancer("term", raw, base_weight=0.0)

    assert torch.allclose(weighted, torch.tensor(0.5))
    assert torch.allclose(activated_from_zero, raw)


def test_alpha_anti_lattice_penalizes_smooth_foreground_dots_and_has_gradients():
    alpha = torch.full((1, 10, 10, 1), 0.8, requires_grad=True)
    rgb = torch.full((1, 10, 10, 3), 0.4)
    fg = torch.ones(1, 10, 10, 1)

    flat = _alpha_anti_lattice_loss(
        alpha,
        rgb,
        fg,
        blur_px=1,
        edge_band_px=0,
        detail_edge_thresh=0.05,
    )
    assert torch.allclose(flat, torch.zeros_like(flat), atol=1e-7)

    dotted = alpha.detach().clone()
    dotted[:, ::2, ::2, :] = 0.1
    dotted.requires_grad_(True)
    loss = _alpha_anti_lattice_loss(
        dotted,
        rgb,
        fg,
        blur_px=1,
        edge_band_px=0,
        detail_edge_thresh=0.05,
    )
    assert loss > 0
    loss.backward()
    assert dotted.grad is not None
    assert torch.isfinite(dotted.grad).all()
    assert dotted.grad.abs().sum() > 0


def test_alpha_anti_lattice_protects_target_rgb_detail():
    alpha = torch.full((1, 8, 8, 1), 0.8)
    alpha[:, ::2, ::2, :] = 0.1
    fg = torch.ones(1, 8, 8, 1)
    smooth_rgb = torch.full((1, 8, 8, 3), 0.5)
    detailed_rgb = smooth_rgb.clone()
    detailed_rgb[:, ::2, :, :] = 0.0
    detailed_rgb[:, 1::2, :, :] = 1.0

    smooth_loss = _alpha_anti_lattice_loss(
        alpha,
        smooth_rgb,
        fg,
        blur_px=1,
        edge_band_px=0,
        detail_edge_thresh=0.05,
    )
    detailed_loss = _alpha_anti_lattice_loss(
        alpha,
        detailed_rgb,
        fg,
        blur_px=1,
        edge_band_px=0,
        detail_edge_thresh=0.05,
    )

    assert smooth_loss > 0
    assert detailed_loss < smooth_loss


def test_learned_iblend_head_zero_initializes_to_no_residual():
    topk = 2
    head = LearnedIblendFillBlend(topk=topk, hidden=16, layers=2)
    x = torch.randn(1, _learned_iblend_feature_channels(topk), 6, 5)

    cand_delta, fill_delta, rgb_delta = head(x)

    assert cand_delta.shape == (1, topk, 6, 5)
    assert fill_delta.shape == (1, 1, 6, 5)
    assert rgb_delta is None
    assert torch.count_nonzero(cand_delta) == 0
    assert torch.count_nonzero(fill_delta) == 0


def test_learned_iblend_head_can_emit_rgb_residual():
    topk = 3
    head = LearnedIblendFillBlend(topk=topk, hidden=16, layers=1, rgb_residual_scale=0.1)
    x = torch.randn(1, _learned_iblend_feature_channels(topk), 4, 4)

    cand_delta, fill_delta, rgb_delta = head(x)

    assert cand_delta.shape == (1, topk, 4, 4)
    assert fill_delta.shape == (1, 1, 4, 4)
    assert rgb_delta is not None
    assert rgb_delta.shape == (1, 3, 4, 4)
    assert torch.count_nonzero(rgb_delta) == 0


def test_learned_iblend_unet_zero_initializes_to_no_residual_odd_size():
    topk = 2
    head = LearnedIblendFillUNet(topk=topk, hidden=8)
    x = torch.randn(1, _learned_iblend_feature_channels(topk), 7, 5)

    cand_delta, fill_delta, rgb_delta = head(x)

    assert cand_delta.shape == (1, topk, 7, 5)
    assert fill_delta.shape == (1, 1, 7, 5)
    assert rgb_delta is None
    assert torch.count_nonzero(cand_delta) == 0
    assert torch.count_nonzero(fill_delta) == 0


def test_depth_refine_unet_zero_initializes_to_no_residual_odd_size():
    head = DepthRefineUNet(hidden=8)
    x = torch.randn(2, DepthRefineUNet.in_channels, 7, 5)

    delta = head(x)

    assert delta.shape == (2, 1, 7, 5)
    assert torch.count_nonzero(delta) == 0


def test_depth_refine_unet_accepts_extra_feature_channels():
    head = DepthRefineUNet(hidden=8, in_channels=10)
    x = torch.randn(2, 10, 7, 5)

    delta = head(x)

    assert delta.shape == (2, 1, 7, 5)
    assert torch.count_nonzero(delta) == 0


def test_condition_depth_affine_head_zero_initializes_to_identity_residuals():
    head = ConditionDepthAffineHead(in_features=12, hidden=16, layers=2)
    x = torch.randn(4, 12)

    raw = head(x)

    assert raw.shape == (4, 2)
    assert torch.count_nonzero(raw) == 0


def test_surface_confidence_unet_zero_initializes_to_no_residual_odd_size():
    head = SurfaceConfidenceUNet(hidden=8)
    x = torch.randn(2, SurfaceConfidenceUNet.in_channels, 7, 5)

    delta = head(x)

    assert delta.shape == (2, 1, 7, 5)
    assert torch.count_nonzero(delta) == 0


def test_surface_refine_unet_zero_initializes_to_no_residual_odd_size():
    head = SurfaceRefineUNet(hidden=8)
    x = torch.randn(2, SurfaceRefineUNet.in_channels, 7, 5)

    delta = head(x)

    assert delta.shape == (2, 5, 7, 5)
    assert torch.count_nonzero(delta) == 0


def test_fusion_candidate_gate_zero_initializes_to_identity_residuals():
    head = FusionCandidateGate(hidden=8, layers=2)
    x = torch.randn(5, FUSION_CANDIDATE_FEATURE_CHANNELS)

    score_delta, opacity_delta = head(x)

    assert score_delta.shape == (5, 1)
    assert opacity_delta.shape == (5, 1)
    assert torch.count_nonzero(score_delta) == 0
    assert torch.count_nonzero(opacity_delta) == 0


def test_fusion_candidate_features_shape_and_defaults():
    params = {
        "rgb": torch.rand(4, 3),
        "opacity": torch.rand(4, 1),
        "depth": torch.linspace(1.0, 4.0, 4)[:, None],
        "scale": torch.full((4, 3), 0.001),
        "_fusion_score": torch.tensor([[0.0], [1.0], [2.0], [-1.0]]),
        "_fusion_support": torch.tensor([[0.0], [1.0], [2.0], [0.0]]),
        "_fusion_conflict": torch.tensor([[1.0], [0.0], [0.0], [2.0]]),
        "_fusion_coverage": torch.ones(4, 1),
    }

    feat = _fusion_candidate_features(params, radius=2.0, ref_count=4)

    assert feat.shape == (4, FUSION_CANDIDATE_FEATURE_CHANNELS)
    assert torch.isfinite(feat).all()


def test_fusion_candidate_features_optional_coords():
    params = {
        "rgb": torch.rand(4, 3),
        "opacity": torch.rand(4, 1),
        "mean": torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        "_fusion_source": torch.arange(4)[:, None],
    }

    feat = _fusion_candidate_features(params, radius=2.0, ref_count=4, include_coords=True)

    assert feat.shape == (
        4,
        FUSION_CANDIDATE_FEATURE_CHANNELS + FUSION_CANDIDATE_COORD_FEATURE_CHANNELS,
    )
    assert torch.isfinite(feat).all()


def test_fusion_candidate_features_optional_rich_features():
    params = {
        "rgb": torch.rand(4, 3),
        "opacity": torch.rand(4, 1),
        "depth": torch.linspace(1.0, 4.0, 4)[:, None],
        "scale": torch.full((4, 3), 0.001),
        "_fusion_score": torch.tensor([[0.0], [1.0], [2.0], [-1.0]]),
        "_fusion_support": torch.tensor([[0.0], [1.0], [2.0], [0.0]]),
        "_fusion_conflict": torch.tensor([[1.0], [0.0], [0.0], [2.0]]),
        "_fusion_coverage": torch.tensor([[1.0], [2.0], [4.0], [4.0]]),
        "_fusion_color_support": torch.tensor([[0.0], [0.5], [1.5], [0.0]]),
        "_fusion_depth_error": torch.tensor([[1.0], [0.0], [2.0], [4.0]]),
        "_fusion_color_error": torch.tensor([[0.1], [0.0], [0.2], [0.4]]),
        "_fusion_front_conflict": torch.tensor([[0.0], [0.0], [1.0], [1.0]]),
        "_fusion_silhouette_conflict": torch.tensor([[1.0], [0.0], [0.0], [1.0]]),
    }

    feat = _fusion_candidate_features(params, radius=2.0, ref_count=4, include_rich=True)

    assert feat.shape == (
        4,
        FUSION_CANDIDATE_FEATURE_CHANNELS + FUSION_CANDIDATE_RICH_FEATURE_CHANNELS,
    )
    assert torch.isfinite(feat).all()


def test_fusion_candidate_features_optional_voxel_context():
    params = {
        "rgb": torch.rand(5, 3),
        "opacity": torch.tensor([[0.8], [0.5], [0.2], [0.7], [0.1]]),
        "mean": torch.tensor([
            [0.00, 0.00, 0.00],
            [0.10, 0.05, 0.00],
            [0.80, 0.00, 0.00],
            [0.90, 0.10, 0.00],
            [1.60, 0.00, 0.00],
        ]),
        "_fusion_score": torch.tensor([[4.0], [1.0], [2.0], [5.0], [0.0]]),
        "_fusion_support": torch.tensor([[4.0], [1.0], [2.0], [3.0], [0.0]]),
        "_fusion_conflict": torch.tensor([[0.0], [2.0], [1.0], [0.0], [4.0]]),
    }

    feat = _fusion_candidate_features(
        params,
        radius=2.0,
        ref_count=4,
        include_voxel=True,
        voxel_size=0.5,
    )

    assert feat.shape == (
        5,
        FUSION_CANDIDATE_FEATURE_CHANNELS + FUSION_CANDIDATE_VOXEL_FEATURE_CHANNELS,
    )
    assert torch.isfinite(feat).all()


def test_fusion_candidate_features_optional_neighbor_context():
    params = {
        "rgb": torch.rand(5, 3),
        "opacity": torch.tensor([[0.8], [0.5], [0.2], [0.7], [0.1]]),
        "depth": torch.linspace(1.0, 4.0, 5)[:, None],
        "scale": torch.full((5, 3), 0.001),
        "mean": torch.tensor([
            [0.00, 0.00, 0.00],
            [0.10, 0.05, 0.00],
            [0.80, 0.00, 0.00],
            [0.90, 0.10, 0.00],
            [1.60, 0.00, 0.00],
        ]),
        "_fusion_score": torch.tensor([[4.0], [1.0], [2.0], [5.0], [0.0]]),
        "_fusion_support": torch.tensor([[4.0], [1.0], [2.0], [3.0], [0.0]]),
        "_fusion_conflict": torch.tensor([[0.0], [2.0], [1.0], [0.0], [4.0]]),
        "_fusion_coverage": torch.tensor([[4.0], [4.0], [3.0], [3.0], [1.0]]),
        "_fusion_color_support": torch.tensor([[3.0], [1.0], [1.0], [2.0], [0.0]]),
        "_fusion_depth_error": torch.tensor([[0.0], [1.0], [2.0], [0.0], [4.0]]),
        "_fusion_color_error": torch.tensor([[0.0], [0.2], [0.4], [0.1], [0.8]]),
    }

    feat = _fusion_candidate_features(
        params,
        radius=2.0,
        ref_count=4,
        include_neighbor=True,
        neighbor_radius=1,
        voxel_size=0.5,
    )

    assert feat.shape == (
        5,
        FUSION_CANDIDATE_FEATURE_CHANNELS
        + FUSION_CANDIDATE_NEIGHBOR_FEATURE_CHANNELS,
    )
    assert torch.isfinite(feat).all()


def test_normalize_confidence_map_preserves_constant_prior():
    conf = torch.full((3, 4), 0.995)
    valid = torch.ones_like(conf, dtype=torch.bool)

    norm = _normalize_confidence_map(conf, valid)

    assert torch.allclose(norm, conf)


def test_normalize_confidence_map_uses_quantiles_when_nonconstant():
    conf = torch.tensor([[0.0, 0.25], [0.75, 1.0]])
    valid = torch.ones_like(conf, dtype=torch.bool)

    norm = _normalize_confidence_map(conf, valid, q_low=0.0, q_high=1.0)

    assert torch.allclose(norm, conf)


def test_depth_confidence_targets_ignore_ambiguous_band():
    prior = torch.tensor([0.10, 0.12, 0.16, 0.30])
    target = torch.full_like(prior, 0.10)
    valid = torch.tensor([True, True, True, True])

    labels, label_valid = _depth_confidence_targets(
        prior,
        target,
        valid,
        positive_tol_frac=0.02,
        negative_tol_frac=0.08,
    )

    assert torch.equal(labels, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(label_valid, torch.tensor([True, True, False, True]))


def test_alpha_mask_stats_reports_background_leak_and_missing_fg():
    alpha = torch.tensor([
        [[[[1.0], [0.2]], [[0.0], [0.6]]]],
    ]).reshape(1, 2, 2, 1)
    mask = torch.tensor([
        [[[[1.0], [0.0]], [[1.0], [0.0]]]],
    ]).reshape(1, 2, 2, 1)

    stats = _alpha_mask_stats(alpha, mask)

    assert abs(stats["alpha_l1"] - 0.45) < 1e-6
    assert abs(stats["alpha_bg_mean"] - 0.4) < 1e-6
    assert abs(stats["alpha_fg_miss"] - 0.5) < 1e-6
    assert stats["alpha_fp_gt_0_1"] == 1.0
    assert stats["alpha_fp_gt_0_5"] == 0.5
    assert stats["alpha_fn_le_0_5"] == 0.5
    assert abs(stats["alpha_iou_0_5"] - (1.0 / 3.0)) < 1e-6


def test_depth_support_window_radius_tolerates_one_pixel_projection_error():
    fg = torch.tensor([
        [False, False, False],
        [False, True, False],
        [False, False, False],
    ])
    depth = torch.full((3, 3), 1e6)
    depth[1, 1] = 2.0
    u = torch.tensor([2.0])
    v = torch.tensor([1.0])
    z = torch.tensor([2.01])

    fg0, match0, front0, _ = _sample_depth_support_window(
        fg, depth, u, v, z, tol=0.05, radius_px=0
    )
    fg1, match1, front1, _ = _sample_depth_support_window(
        fg, depth, u, v, z, tol=0.05, radius_px=1
    )

    assert not fg0.item()
    assert not match0.item()
    assert not front0.item()
    assert fg1.item()
    assert match1.item()
    assert not front1.item()


def test_local_valid_median_map_ignores_invalid_outlier():
    values = torch.tensor([[
        [0.20, 0.21, 0.22],
        [0.19, 0.99, 0.23],
        [0.18, 0.24, 0.25],
    ]])
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[:, 1, 1] = False

    median, median_valid = _local_valid_median_map(values, valid, radius_px=1)

    assert median_valid[:, 1, 1].item()
    assert torch.allclose(median[:, 1, 1], torch.tensor([0.21]))


def test_depth_multiview_support_maps_identifies_support_and_conflict():
    depths = torch.full((2, 3, 3), 1e6)
    depths[:, 1, 1] = 2.0
    fg = torch.zeros(2, 3, 3, 1)
    fg[:, 1, 1, 0] = 1.0
    k = torch.eye(3).repeat(2, 1, 1)
    k[:, 0, 0] = 1.0
    k[:, 1, 1] = 1.0
    k[:, 0, 2] = 1.0
    k[:, 1, 2] = 1.0
    c2w = torch.eye(4).repeat(2, 1, 1)

    support = _depth_multiview_support_maps(
        depths, fg, k, c2w, radius=1.0, tol_frac=0.05, max_refs=1
    )

    assert support.shape == (2, 4, 3, 3)
    assert torch.allclose(support[:, 0, 1, 1], torch.ones(2))
    assert torch.allclose(support[:, 1, 1, 1], torch.zeros(2))
    assert torch.allclose(support[:, 3, 1, 1], torch.ones(2))

    depths[1, 1, 1] = 3.0
    conflict = _depth_multiview_support_maps(
        depths, fg, k, c2w, radius=1.0, tol_frac=0.05, max_refs=1
    )

    assert conflict[0, 0, 1, 1] == 0
    assert conflict[0, 1, 1, 1] == 1
    assert conflict[0, 2, 1, 1] == 1


def test_surface_confidence_protect_mask_keeps_only_well_supported_pixels():
    mv = torch.zeros(4, 2, 2)
    mv[0] = torch.tensor([[1.0, 0.8], [1.0, 0.2]])  # support
    mv[1] = torch.tensor([[0.0, 0.4], [0.7, 0.0]])  # conflict
    mv[3] = torch.tensor([[1.0, 1.0], [1.0, 0.3]])  # coverage

    mask = _surface_confidence_protect_mask(
        mv,
        support_min=0.75,
        conflict_max=0.5,
        coverage_min=0.5,
    )

    expected = torch.tensor([[True, True], [False, False]])
    assert torch.equal(mask, expected)


def test_iblend_color_mode_average_matches_weighted_mean():
    obj = torch.tensor([
        [[[1.0, 0.0, 0.0]]],
        [[[0.0, 1.0, 0.0]]],
    ])
    weights = torch.tensor([
        [[[0.75]]],
        [[[0.25]]],
    ])

    out = _select_iblend_object_color(obj, weights, "average")

    assert torch.allclose(out, torch.tensor([[[0.75, 0.25, 0.0]]]))


def test_iblend_color_mode_nearest_keeps_first_candidate_color():
    obj = torch.tensor([
        [[[1.0, 0.0, 0.0]]],
        [[[0.0, 1.0, 0.0]]],
    ])
    weights = torch.tensor([
        [[[0.1]]],
        [[[0.9]]],
    ])

    out = _select_iblend_object_color(obj, weights, "nearest")

    assert torch.allclose(out, torch.tensor([[[1.0, 0.0, 0.0]]]))


def test_iblend_color_mode_maxweight_picks_per_pixel_candidate():
    obj = torch.tensor([
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        [[[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
    ])
    weights = torch.tensor([
        [[[0.8], [0.2]]],
        [[[0.2], [0.8]]],
    ])

    out = _select_iblend_object_color(obj, weights, "maxweight")

    expected = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    assert torch.allclose(out, expected)


def test_iblend_color_mode_maxweight_st_matches_forward_and_has_weight_grad():
    obj = torch.tensor([
        [[[1.0, 0.0, 0.0]]],
        [[[0.0, 1.0, 0.0]]],
    ])
    weights = torch.tensor([
        [[[0.25]]],
        [[[0.75]]],
    ], requires_grad=True)

    out = _select_iblend_object_color(obj, weights, "maxweight_st")

    assert torch.allclose(out.detach(), torch.tensor([[[0.0, 1.0, 0.0]]]))
    out[..., 0].sum().backward()
    assert weights.grad is not None
    assert torch.count_nonzero(weights.grad) > 0


def test_iblend_alpha_mode_maxweight_st_matches_forward_and_has_weight_grad():
    alpha = torch.tensor([
        [[[0.25]]],
        [[[0.75]]],
    ])
    weights = torch.tensor([
        [[[0.25]]],
        [[[0.75]]],
    ], requires_grad=True)

    out = _select_iblend_alpha(alpha, weights, "maxweight_st")

    assert torch.allclose(out.detach(), torch.tensor([[[0.75]]]))
    out.sum().backward()
    assert weights.grad is not None
    assert torch.count_nonzero(weights.grad) > 0


def test_iblend_alpha_mode_nearest_keeps_first_candidate_alpha():
    alpha = torch.tensor([
        [[[0.25]]],
        [[[0.75]]],
    ])
    weights = torch.tensor([
        [[[0.1]]],
        [[[0.9]]],
    ])

    out = _select_iblend_alpha(alpha, weights, "nearest")

    assert torch.allclose(out, torch.tensor([[[0.25]]]))


def test_iblend_alpha_mode_nearest_average_blends_first_and_weighted_alpha():
    alpha = torch.tensor([
        [[[0.25]]],
        [[[0.75]]],
    ])
    weights = torch.tensor([
        [[[0.25]]],
        [[[0.75]]],
    ])

    out = _select_iblend_alpha(alpha, weights, "nearest_average")

    assert torch.allclose(out, torch.tensor([[[0.4375]]]))


def test_compose_iblend_anchor_uses_requested_color_mode():
    obj = torch.tensor([
        [[[1.0, 0.0, 0.0]]],
        [[[0.0, 1.0, 0.0]]],
    ])
    alpha = torch.tensor([
        [[[0.5]]],
        [[[0.5]]],
    ])
    weights = torch.tensor([
        [[[0.25]]],
        [[[0.75]]],
    ])

    _, alpha_out, rgb = _compose_iblend_anchor(
        obj, alpha, weights, color_mode="maxweight", alpha_mode="average", bg=1.0
    )

    assert torch.allclose(alpha_out, torch.tensor([[[0.5]]]))
    assert torch.allclose(rgb, torch.tensor([[[0.5, 1.0, 0.5]]]))


def test_compose_iblend_anchor_can_hard_select_alpha():
    obj = torch.tensor([
        [[[1.0, 0.0, 0.0]]],
        [[[0.0, 1.0, 0.0]]],
    ])
    alpha = torch.tensor([
        [[[0.25]]],
        [[[0.75]]],
    ])
    weights = torch.tensor([
        [[[0.75]]],
        [[[0.25]]],
    ])

    _, alpha_out, rgb = _compose_iblend_anchor(
        obj, alpha, weights, color_mode="maxweight", alpha_mode="maxweight", bg=1.0
    )

    assert torch.allclose(alpha_out, torch.tensor([[[0.25]]]))
    assert torch.allclose(rgb, torch.tensor([[[1.0, 0.75, 0.75]]]))


def test_static_fill_blend_default_matches_alpha_power_rule():
    primary_rgb = torch.tensor([[[0.2, 0.2, 0.2]]])
    primary_alpha = torch.tensor([[[0.25]]])
    static_rgb = torch.tensor([[[1.0, 0.0, 0.0]]])
    static_alpha = torch.tensor([[[0.5]]])

    rgb, alpha = _blend_static_fill(
        primary_rgb, primary_alpha, static_rgb, static_alpha, fill_alpha_power=1.0
    )

    assert torch.allclose(rgb, torch.tensor([[[0.8, 0.05, 0.05]]]))
    assert torch.allclose(alpha, torch.tensor([[[0.4375]]]))


def test_static_fill_alpha_min_suppresses_low_confidence_static_fill():
    primary_rgb = torch.tensor([[[0.2, 0.2, 0.2]]])
    primary_alpha = torch.tensor([[[0.25]]])
    static_rgb = torch.tensor([[[1.0, 0.0, 0.0]]])
    static_alpha = torch.tensor([[[0.1]]])

    rgb, alpha = _blend_static_fill(
        primary_rgb, primary_alpha, static_rgb, static_alpha,
        fill_alpha_power=1.0,
        static_alpha_min=0.2,
        static_alpha_softness=0.0,
    )

    assert torch.allclose(rgb, primary_rgb)
    assert torch.allclose(alpha, primary_alpha)


def test_static_fill_alpha_softness_partially_keeps_static_fill():
    primary_rgb = torch.tensor([[[0.0, 0.0, 0.0]]])
    primary_alpha = torch.tensor([[[0.0]]])
    static_rgb = torch.tensor([[[1.0, 0.0, 0.0]]])
    static_alpha = torch.tensor([[[0.3]]])

    rgb, alpha = _blend_static_fill(
        primary_rgb, primary_alpha, static_rgb, static_alpha,
        fill_alpha_power=1.0,
        static_alpha_min=0.2,
        static_alpha_softness=0.2,
    )

    assert torch.allclose(rgb, torch.tensor([[[0.5, 0.0, 0.0]]]), atol=1e-6)
    assert torch.allclose(alpha, torch.tensor([[[0.15]]]), atol=1e-6)
