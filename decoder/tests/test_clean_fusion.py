from __future__ import annotations

import torch

from decoder.clean.fusion import (
    _sh_bases,
    rgbd_fit_sh_colors,
    rgbd_target_view_surface,
    rgbd_target_view_surface_splat,
    rgbd_tsdf_filter_params,
    rgbd_tsdf_fuse,
    voxel_fuse_params,
)


def _params(mean: torch.Tensor, opacity: torch.Tensor) -> dict:
    n = mean.shape[0]
    return {
        "mean": mean,
        "opacity": opacity.reshape(n, 1),
        "rgb": torch.arange(n * 3, dtype=mean.dtype).reshape(n, 3),
        "scale": torch.ones(n, 3, dtype=mean.dtype),
        "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=mean.dtype).repeat(n, 1),
        "depth": torch.ones(n, 1, dtype=mean.dtype),
        "scale_raw": torch.ones(n, 3, dtype=mean.dtype),
        "mean_anchor": mean.clone(),
        "mean_offset": torch.zeros_like(mean),
    }


def test_voxel_fuse_keeps_highest_opacity_per_cell():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.04, 0.04, 0.04],
        [0.25, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.2, 0.8, 0.4]))
    fused, stats = voxel_fuse_params(p, voxel_size=0.1)
    assert stats["input"] == 3
    assert stats["output"] == 2
    assert torch.allclose(fused["mean"][0], mean[1])
    assert torch.allclose(fused["mean"][1], mean[2])


def test_voxel_fuse_min_count_drops_singletons():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.04, 0.04, 0.04],
        [0.25, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.2, 0.8, 0.4]))
    fused, stats = voxel_fuse_params(p, voxel_size=0.1, min_count=2)
    assert stats["output"] == 1
    assert stats["dropped_low_support"] == 1
    assert torch.allclose(fused["mean"][0], mean[1])


def test_voxel_fuse_can_keep_multiple_representatives_per_cell():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.02, 0.00, 0.00],
        [0.04, 0.00, 0.00],
        [0.30, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.2, 0.8, 0.4, 0.6]))
    fused, stats = voxel_fuse_params(p, voxel_size=0.1, max_per_voxel=2)
    assert stats["output"] == 3
    assert torch.allclose(fused["mean"][:2], mean[:2])
    assert torch.allclose(fused["mean"][2], mean[3])


def test_voxel_fuse_average_collapses_cell_and_sets_scale_floor():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.40, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.5, 0.5]))
    p["scale"] = torch.full((3, 3), 0.01)
    fused, stats = voxel_fuse_params(
        p, voxel_size=0.25, mode="average", scale_floor=0.08
    )
    assert stats["output"] == 2
    x = fused["mean"][:, 0].sort().values
    assert torch.allclose(x, torch.tensor([0.05, 0.4]))
    assert torch.all(fused["scale"] >= 0.08)


def test_voxel_fuse_average_can_downweight_spatial_outlier():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.02, 0.00, 0.00],
        [0.45, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.5, 0.5]))

    plain, _ = voxel_fuse_params(
        p, voxel_size=0.5, mode="average",
    )
    robust, _ = voxel_fuse_params(
        p, voxel_size=0.5, mode="average", average_dist_decay=20.0,
    )

    assert torch.allclose(plain["mean"][0, 0], torch.tensor(0.1567), atol=1e-4)
    assert robust["mean"][0, 0] < 0.08


def test_voxel_fuse_detail_can_reduce_scale_floor():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.40, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.5, 0.5]))
    p["scale"] = torch.full((3, 3), 0.01)
    p["_fusion_detail"] = torch.tensor([[1.0], [1.0], [0.0]])

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="average", scale_floor=0.08,
        scale_floor_detail_key="_fusion_detail", scale_floor_detail_min=0.5,
    )

    order = fused["mean"][:, 0].argsort()
    assert torch.allclose(fused["scale"][order[0]], torch.full((3,), 0.04))
    assert torch.allclose(fused["scale"][order[1]], torch.full((3,), 0.08))


def test_voxel_fuse_scale_floor_can_keep_normal_axis_thin():
    mean = torch.tensor([[0.00, 0.00, 0.00], [0.10, 0.00, 0.00]])
    p = _params(mean, torch.tensor([0.5, 0.5]))
    p["scale"] = torch.tensor([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]])

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="average", scale_floor=0.08,
        scale_floor_z_mult=0.25,
    )

    assert torch.allclose(fused["scale"][0], torch.tensor([0.08, 0.08, 0.02]))


def test_voxel_fuse_average_can_select_representative_rgb():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.9]))
    p["rgb"] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="average", color_mode="select"
    )

    assert torch.allclose(fused["mean"][0], torch.tensor([0.0643, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(fused["rgb"][0], p["rgb"][1])


def test_voxel_fuse_can_select_medoid_representative_rgb():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.11, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.9, 0.5, 0.5]))
    p["rgb"] = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    fused_op, stats_op = voxel_fuse_params(
        p, voxel_size=0.25, mode="average", color_mode="select",
        representative_mode="opacity",
    )
    fused_med, stats_med = voxel_fuse_params(
        p, voxel_size=0.25, mode="average", color_mode="select",
        representative_mode="medoid",
    )

    assert stats_med["representative_mode"] == "medoid"
    assert torch.allclose(fused_op["rgb"][0], p["rgb"][0])
    assert torch.allclose(fused_med["rgb"][0], p["rgb"][1])
    assert torch.allclose(fused_med["mean"][0], fused_op["mean"][0])


def test_voxel_fuse_can_select_score_representative():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.11, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.9, 0.5, 0.5]))
    p["_fusion_score"] = torch.tensor([[0.1], [2.0], [1.0]])

    fused, stats = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", representative_mode="score",
    )

    assert stats["representative_mode"] == "score"
    assert torch.allclose(fused["mean"][0], mean[1])


def test_voxel_fuse_can_use_score_color_with_medoid_geometry():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.11, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.9, 0.5, 0.5]))
    p["rgb"] = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    p["_fusion_score"] = torch.tensor([[0.1], [0.2], [3.0]])

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", representative_mode="medoid",
        color_mode="score_select",
    )

    assert torch.allclose(fused["mean"][0], mean[1])
    assert torch.allclose(fused["rgb"][0], p["rgb"][2])


def test_voxel_fuse_score_soft_color_is_differentiable():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.11, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.5, 0.5]))
    p["rgb"] = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    score = torch.tensor([[0.0], [1.0], [2.0]], requires_grad=True)
    p["_fusion_score"] = score

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", representative_mode="medoid",
        color_mode="score_soft", score_softmax_temp=0.5,
    )

    expected_w = torch.softmax(score.detach().reshape(-1) / 0.5, dim=0)
    expected = (p["rgb"] * expected_w[:, None]).sum(dim=0)
    assert torch.allclose(fused["mean"][0], mean[1])
    assert torch.allclose(fused["rgb"][0], expected, atol=1e-6)
    fused["rgb"][0, 1].backward()
    assert score.grad is not None
    assert score.grad.abs().sum() > 0


def test_voxel_fuse_score_soft_opacity_mix_is_differentiable():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.11, 0.00, 0.00],
    ])
    opacity = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)
    p = _params(mean, opacity)
    score = torch.tensor([[0.0], [1.0], [2.0]], requires_grad=True)
    p["_fusion_score"] = score

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", representative_mode="medoid",
        score_soft_opacity_mix=1.0, score_softmax_temp=0.5,
    )

    expected_w = torch.softmax(score.detach().reshape(-1) / 0.5, dim=0)
    expected = (opacity.detach() * expected_w).sum()
    assert torch.allclose(fused["opacity"][0, 0], expected, atol=1e-6)
    fused["opacity"][0, 0].backward()
    assert score.grad is not None
    assert score.grad.abs().sum() > 0
    assert opacity.grad is not None
    assert opacity.grad.abs().sum() > 0


def test_voxel_fuse_score_soft_geometry_mix_is_differentiable():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.11, 0.00, 0.00],
    ], requires_grad=True)
    p = _params(mean, torch.tensor([0.5, 0.5, 0.5]))
    score = torch.tensor([[0.0], [1.0], [2.0]], requires_grad=True)
    p["_fusion_score"] = score

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", representative_mode="medoid",
        score_soft_geometry_mix=1.0, score_softmax_temp=0.5,
    )

    expected_w = torch.softmax(score.detach().reshape(-1) / 0.5, dim=0)
    expected = (mean.detach() * expected_w[:, None]).sum(dim=0)
    assert torch.allclose(fused["mean"][0], expected, atol=1e-6)
    fused["mean"][0, 0].backward()
    assert score.grad is not None
    assert score.grad.abs().sum() > 0
    assert mean.grad is not None
    assert mean.grad.abs().sum() > 0


def test_voxel_fuse_can_keep_topk_score_representatives():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.11, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.9, 0.5, 0.5]))
    p["_fusion_score"] = torch.tensor([[0.1], [2.0], [1.0]])

    fused, stats = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", representative_mode="score",
        max_per_voxel=2,
    )

    assert stats["output"] == 2
    assert torch.allclose(fused["mean"][0], mean[1])
    assert torch.allclose(fused["mean"][1], mean[2])


def test_voxel_fuse_can_blend_average_and_representative_rgb():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.5]))
    p["rgb"] = torch.tensor([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="average", color_mode="select",
        color_select_mix=0.25,
    )

    expected_avg = torch.tensor([0.5, 0.5, 0.5])
    expected = expected_avg * 0.75 + p["rgb"][0] * 0.25
    assert torch.allclose(fused["rgb"][0], expected)


def test_voxel_fuse_select_sets_scale_floor():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.5]))
    p["scale"] = torch.full((2, 3), 0.01)
    fused, stats = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", scale_floor=0.08
    )
    assert stats["output"] == 1
    assert torch.all(fused["scale"] >= 0.08)


def test_voxel_fuse_select_can_append_average_coverage_splat():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.5]))
    p["rgb"] = torch.tensor([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
    p["scale"] = torch.full((2, 3), 0.01)

    fused, stats = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", representative_mode="medoid",
        scale_floor=0.08, coverage_opacity_mult=0.25, coverage_scale_mult=1.5,
    )

    assert stats["output"] == 2
    assert stats["coverage_added"] == 1
    assert torch.allclose(fused["mean"][0], mean[0])
    assert torch.allclose(fused["mean"][1], torch.tensor([0.05, 0.0, 0.0]))
    assert torch.allclose(fused["rgb"][1], torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(fused["opacity"][1, 0], fused["opacity"][0, 0] * 0.25)
    assert torch.allclose(fused["scale"][1], torch.full((3,), 0.12))


def test_voxel_fuse_select_can_average_color_without_moving_geometry():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.5]))
    p["rgb"] = torch.tensor([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", color_select_mix=0.0,
        representative_mode="medoid",
    )

    assert torch.allclose(fused["mean"][0], p["mean"][0])
    assert torch.allclose(fused["rgb"][0], torch.tensor([0.5, 0.5, 0.5]))


def test_voxel_fuse_select_can_blend_average_and_representative_color():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.5, 0.5]))
    p["rgb"] = torch.tensor([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])

    fused, _ = voxel_fuse_params(
        p, voxel_size=0.25, mode="select", color_select_mix=0.25,
        representative_mode="medoid",
    )

    expected = torch.tensor([0.5, 0.5, 0.5]) * 0.75 + p["rgb"][0] * 0.25
    assert torch.allclose(fused["mean"][0], p["mean"][0])
    assert torch.allclose(fused["rgb"][0], expected)


def test_voxel_fuse_min_count_can_count_distinct_sources():
    mean = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.02, 0.00, 0.00],
        [0.04, 0.00, 0.00],
        [0.30, 0.00, 0.00],
        [0.32, 0.00, 0.00],
    ])
    p = _params(mean, torch.tensor([0.2, 0.8, 0.4, 0.6, 0.7]))
    p["_fusion_source"] = torch.tensor([[0], [0], [0], [0], [1]])
    fused, stats = voxel_fuse_params(
        p, voxel_size=0.1, min_count=2, support_key="_fusion_source"
    )
    assert stats["output"] == 1
    assert torch.allclose(fused["mean"][0], mean[4])


def test_voxel_fuse_noop_for_nonpositive_voxel_size():
    mean = torch.randn(4, 3)
    p = _params(mean, torch.rand(4))
    fused, stats = voxel_fuse_params(p, voxel_size=0.0)
    assert stats["output"] == 4
    assert fused["mean"].data_ptr() == p["mean"].data_ptr()


def test_voxel_fuse_soft_low_support_keeps_dim_splats():
    p = _params(
        torch.tensor([[0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([0.8, 0.8]),
    )
    p["_fusion_source"] = torch.tensor([[0], [0]])

    fused, stats = voxel_fuse_params(
        p, voxel_size=0.5, min_count=2, mode="average",
        support_key="_fusion_source", low_support_opacity_decay=1.0,
    )

    assert stats["output"] == 2
    assert torch.all(fused["opacity"] < 0.8)
    assert torch.all(fused["opacity"] > 0.0)


def test_voxel_fuse_low_support_scale_floor_can_stay_thin():
    p = _params(
        torch.tensor([
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [1.00, 0.0, 0.0],
        ]),
        torch.tensor([0.8, 0.8, 0.8]),
    )
    p["scale"] = torch.full((3, 3), 0.01)
    p["_fusion_source"] = torch.tensor([[0], [1], [0]])

    fused, stats = voxel_fuse_params(
        p, voxel_size=0.25, min_count=2, mode="average",
        scale_floor=0.08, support_key="_fusion_source",
        low_support_opacity_decay=1.0, low_support_scale_floor_mult=0.25,
    )

    order = fused["mean"][:, 0].argsort()
    assert stats["output"] == 2
    assert torch.allclose(fused["scale"][order[0]], torch.full((3,), 0.08))
    assert torch.allclose(fused["scale"][order[1]], torch.full((3,), 0.02))


def test_voxel_fuse_neighbor_decay_suppresses_isolated_low_support():
    p = _params(
        torch.tensor([
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.30, 0.0, 0.0],
            [1.00, 0.0, 0.0],
        ]),
        torch.tensor([0.8, 0.8, 0.8, 0.8]),
    )
    p["_fusion_source"] = torch.tensor([[0], [1], [0], [0]])

    fused, stats = voxel_fuse_params(
        p, voxel_size=0.25, min_count=2, mode="average",
        support_key="_fusion_source", low_support_opacity_decay=0.1,
        neighbor_support_min=1, neighbor_opacity_decay=2.0,
    )

    order = fused["mean"][:, 0].argsort()
    op_sorted = fused["opacity"][order, 0]
    assert stats["neighbor_dropped"] == 1
    assert op_sorted[-1] < op_sorted[-2] * 0.5


def test_voxel_fuse_support_propagation_softly_suppresses_disconnected_low_support():
    p = _params(
        torch.tensor([
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.30, 0.0, 0.0],
            [1.00, 0.0, 0.0],
        ]),
        torch.tensor([0.8, 0.8, 0.8, 0.8]),
    )
    p["_fusion_source"] = torch.tensor([[0], [1], [0], [0]])

    fused, stats = voxel_fuse_params(
        p, voxel_size=0.25, min_count=2, mode="average",
        support_key="_fusion_source", low_support_opacity_decay=0.1,
        support_propagation_steps=1, support_propagation_radius=1,
        support_propagation_opacity_decay=2.0,
    )

    order = fused["mean"][:, 0].argsort()
    op_sorted = fused["opacity"][order, 0]
    assert stats["propagation_dropped"] == 1
    assert stats["output"] == 3
    assert op_sorted[1] > op_sorted[2] * 3.0


def test_voxel_fuse_support_propagation_can_remove_disconnected_low_support():
    p = _params(
        torch.tensor([
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.30, 0.0, 0.0],
            [1.00, 0.0, 0.0],
        ]),
        torch.tensor([0.8, 0.8, 0.8, 0.8]),
    )
    p["_fusion_source"] = torch.tensor([[0], [1], [0], [0]])

    fused, stats = voxel_fuse_params(
        p, voxel_size=0.25, min_count=2, mode="average",
        support_key="_fusion_source", low_support_opacity_decay=0.1,
        support_propagation_steps=1, support_propagation_radius=1,
        support_propagation_opacity_decay=0.0,
    )

    x = fused["mean"][:, 0].sort().values
    assert stats["propagation_dropped"] == 1
    assert stats["output"] == 2
    assert torch.all(x < 0.5)


def test_rgbd_tsdf_fuse_extracts_constant_depth_plane():
    h, w = 16, 16
    frames = torch.ones(1, h, w, 3) * 0.25
    masks = torch.ones(1, h, w, 1)
    depths = torch.ones(1, h, w) * 2.0
    K = torch.eye(3).repeat(1, 1, 1)
    K[:, 0, 0] = 20.0
    K[:, 1, 1] = 20.0
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    w2c = torch.eye(4).repeat(1, 1, 1)
    c2w = torch.eye(4).repeat(1, 1, 1)

    params, stats = rgbd_tsdf_fuse(
        frames, masks, depths, K, w2c, c2w,
        voxel_size=0.1, min_weight=1, max_voxels=100000, max_points=1000,
    )

    assert stats["output"] > 32
    assert params["mean"].shape[1] == 3
    assert torch.allclose(params["mean"][:, 2].median(), torch.tensor(2.0), atol=0.11)
    assert torch.allclose(params["rgb"].mean(), torch.tensor(0.25), atol=1e-4)
    assert torch.all(params["opacity"] > 0.9)


def test_rgbd_target_view_surface_keeps_frontmost_color():
    h, w = 8, 8
    frames = torch.zeros(2, h, w, 3)
    frames[0, ..., 0] = 1.0
    frames[1, ..., 2] = 1.0
    masks = torch.ones(2, h, w, 1)
    depths = torch.ones(2, h, w) * 2.0
    depths[0] = 1.5
    K = torch.eye(3).repeat(2, 1, 1)
    K[:, 0, 0] = 20.0
    K[:, 1, 1] = 20.0
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[:, 1, 1] = -1.0
    c2w[:, 2, 2] = -1.0
    w2c = torch.eye(4).repeat(2, 1, 1)

    params, stats = rgbd_target_view_surface(
        frames, masks, depths, K, w2c, c2w, K[0], w2c[0], c2w[0],
        w, h, radius=2.0, scale_frac=0.01, normal_scale_frac=0.002,
        opacity=0.9, depth_tol=0.05,
    )

    assert stats["valid_pixels"] > h * w // 2
    assert params["rgb"][:, 0].mean() > 0.95
    assert params["rgb"][:, 2].mean() < 0.05
    assert torch.allclose(params["opacity"].mean(), torch.tensor(0.9), atol=1e-5)


def test_rgbd_target_view_surface_splat_keeps_frontmost_color_and_fills_more():
    h, w = 8, 8
    frames = torch.zeros(2, h, w, 3)
    frames[0, ..., 0] = 1.0
    frames[1, ..., 2] = 1.0
    masks = torch.ones(2, h, w, 1)
    depths = torch.ones(2, h, w) * 2.0
    depths[0] = 1.5
    K = torch.eye(3).repeat(2, 1, 1)
    K[:, 0, 0] = 20.0
    K[:, 1, 1] = 20.0
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[:, 1, 1] = -1.0
    c2w[:, 2, 2] = -1.0
    w2c = torch.eye(4).repeat(2, 1, 1)

    nearest, nearest_stats = rgbd_target_view_surface(
        frames, masks, depths, K, w2c, c2w, K[0], w2c[0], c2w[0],
        w, h, radius=2.0, scale_frac=0.01, normal_scale_frac=0.002,
        opacity=0.9, depth_tol=0.05,
    )
    splat, splat_stats = rgbd_target_view_surface_splat(
        frames, masks, depths, K, w2c, c2w, K[0], w2c[0], c2w[0],
        w, h, radius=2.0, scale_frac=0.01, normal_scale_frac=0.002,
        opacity=0.9, depth_tol=0.05,
    )

    assert splat_stats["valid_pixels"] >= nearest_stats["valid_pixels"]
    assert splat["rgb"][:, 0].mean() > 0.95
    assert splat["rgb"][:, 2].mean() < 0.05
    assert torch.allclose(splat["opacity"].mean(), torch.tensor(0.9), atol=1e-5)
    assert splat["mean"].shape[0] == splat_stats["valid_pixels"]


def test_rgbd_target_view_surface_splat_can_require_multiview_support():
    h, w = 8, 8
    frames = torch.zeros(2, h, w, 3)
    frames[0, ..., 0] = 1.0
    masks = torch.ones(2, h, w, 1)
    masks[1] = 0.0
    depths = torch.ones(2, h, w) * 2.0
    K = torch.eye(3).repeat(2, 1, 1)
    K[:, 0, 0] = 20.0
    K[:, 1, 1] = 20.0
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[:, 1, 1] = -1.0
    c2w[:, 2, 2] = -1.0
    w2c = torch.eye(4).repeat(2, 1, 1)

    params, stats = rgbd_target_view_surface_splat(
        frames, masks, depths, K, w2c, c2w, K[0], w2c[0], c2w[0],
        w, h, radius=2.0, scale_frac=0.01, normal_scale_frac=0.002,
        opacity=0.9, depth_tol=0.05, min_support=2,
    )

    assert stats["valid_pixels"] == 0
    assert params["mean"].numel() == 0


def test_rgbd_target_view_surface_splat_support_handles_out_of_bounds_points():
    h, w = 8, 8
    frames = torch.ones(2, h, w, 3) * 0.5
    masks = torch.ones(2, h, w, 1)
    depths = torch.ones(2, h, w) * 2.0
    K = torch.eye(3).repeat(2, 1, 1)
    K[:, 0, 0] = 20.0
    K[:, 1, 1] = 20.0
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[:, 1, 1] = -1.0
    c2w[:, 2, 2] = -1.0
    w2c = torch.eye(4).repeat(2, 1, 1)
    target_w2c = w2c[0].clone()
    target_w2c[0, 3] = 0.6

    params, stats = rgbd_target_view_surface_splat(
        frames, masks, depths, K, w2c, c2w, K[0], target_w2c, c2w[0],
        w, h, radius=2.0, scale_frac=0.01, normal_scale_frac=0.002,
        opacity=0.9, depth_tol=0.05, min_support=2,
    )

    assert stats["valid_pixels"] > 0
    assert params["mean"].shape[0] == stats["valid_pixels"]


def test_rgbd_tsdf_edge_mode_extracts_constant_depth_plane():
    h, w = 16, 16
    frames = torch.ones(1, h, w, 3) * 0.5
    masks = torch.ones(1, h, w, 1)
    depths = torch.ones(1, h, w) * 2.0
    K = torch.eye(3).repeat(1, 1, 1)
    K[:, 0, 0] = 20.0
    K[:, 1, 1] = 20.0
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    w2c = torch.eye(4).repeat(1, 1, 1)
    c2w = torch.eye(4).repeat(1, 1, 1)

    params, stats = rgbd_tsdf_fuse(
        frames, masks, depths, K, w2c, c2w,
        voxel_size=0.1, min_weight=1, max_voxels=100000, max_points=1000,
        surface_mode="edges",
    )

    assert stats["surface_mode"] == "edges"
    assert stats["output"] > 32
    assert torch.allclose(params["mean"][:, 2].median(), torch.tensor(2.0), atol=0.06)


def test_rgbd_tsdf_filter_downweights_off_surface_splats():
    h, w = 16, 16
    frames = torch.ones(1, h, w, 3) * 0.25
    masks = torch.ones(1, h, w, 1)
    depths = torch.ones(1, h, w) * 2.0
    K = torch.eye(3).repeat(1, 1, 1)
    K[:, 0, 0] = 20.0
    K[:, 1, 1] = 20.0
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    w2c = torch.eye(4).repeat(1, 1, 1)
    c2w = torch.eye(4).repeat(1, 1, 1)
    params = _params(
        torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]),
        torch.tensor([1.0, 1.0]),
    )

    filtered, stats = rgbd_tsdf_filter_params(
        params, frames, masks, depths, K, w2c, c2w,
        voxel_size=0.1, min_weight=1, band=0.2, max_voxels=100000,
    )

    assert stats["filtered"] >= 1
    assert filtered["opacity"][0, 0] > 0.9
    assert filtered["opacity"][1, 0] < 0.5


def test_rgbd_fit_sh_colors_adds_coefficients_for_visible_points():
    h, w = 16, 16
    frames = torch.ones(1, h, w, 3) * 0.25
    masks = torch.ones(1, h, w, 1)
    depths = torch.ones(1, h, w) * 2.0
    K = torch.eye(3).repeat(1, 1, 1)
    K[:, 0, 0] = 20.0
    K[:, 1, 1] = 20.0
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    w2c = torch.eye(4).repeat(1, 1, 1)
    params = _params(
        torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        torch.tensor([1.0, 1.0]),
    )
    params["rgb"] = torch.ones(2, 3) * 0.5

    out, stats = rgbd_fit_sh_colors(
        params, frames, masks, depths, K, w2c,
        degree=1, depth_tol=0.1, min_obs=1,
    )

    assert out["rgb_sh"].shape == (2, 4, 3)
    assert stats["fitted"] == 2
    bases = _sh_bases(params["mean"], degree=1)
    approx_rgb = (bases[..., None] * out["rgb_sh"]).sum(dim=1) + 0.5
    assert torch.allclose(approx_rgb.mean(), torch.tensor(0.25), atol=0.02)


def test_rgbd_fit_sh_colors_can_blend_with_base_color():
    h, w = 16, 16
    frames = torch.ones(1, h, w, 3) * 0.25
    masks = torch.ones(1, h, w, 1)
    depths = torch.ones(1, h, w) * 2.0
    K = torch.eye(3).repeat(1, 1, 1)
    K[:, 0, 0] = 20.0
    K[:, 1, 1] = 20.0
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    w2c = torch.eye(4).repeat(1, 1, 1)
    params = _params(torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([1.0]))
    params["rgb"] = torch.ones(1, 3) * 0.5

    out, stats = rgbd_fit_sh_colors(
        params, frames, masks, depths, K, w2c,
        degree=1, depth_tol=0.1, min_obs=1, mix=0.5,
    )

    bases = _sh_bases(params["mean"], degree=1)
    approx_rgb = (bases[..., None] * out["rgb_sh"]).sum(dim=1) + 0.5
    assert stats["mix"] == 0.5
    assert torch.allclose(approx_rgb.mean(), torch.tensor(0.375), atol=0.03)
