from __future__ import annotations

import torch

from decoder.clean.sparse_voxel_fusion import (
    DENSE_VOXEL_CONTEXT_EXTRA_FEATURES,
    DenseVoxelFusionMLP,
    DenseVoxelMessageFusionMLP,
    SPARSE_VOXEL_FUSION_FEATURES,
    _positive_zero_centered,
    dense_voxel_context_features,
    dense_voxel_message_pairs,
    sparse_voxel_fusion_features,
)


def test_sparse_voxel_fusion_features_are_finite_and_inference_only():
    fused = {
        "rgb": torch.rand(4, 3),
        "opacity": torch.tensor([[0.8], [0.5], [0.2], [0.7]]),
        "scale": torch.full((4, 3), 0.001),
        "mean": torch.tensor([
            [0.00, 0.00, 0.00],
            [0.10, 0.05, 0.00],
            [0.80, 0.00, 0.00],
            [0.90, 0.10, 0.00],
        ]),
        "_fusion_score": torch.tensor([[4.0], [1.0], [2.0], [5.0]]),
        "_fusion_support": torch.tensor([[4.0], [1.0], [2.0], [3.0]]),
        "_fusion_conflict": torch.tensor([[0.0], [2.0], [1.0], [0.0]]),
        "_fusion_coverage": torch.tensor([[4.0], [4.0], [3.0], [3.0]]),
        "_fusion_color_support": torch.tensor([[3.0], [1.0], [1.0], [2.0]]),
        "_fusion_depth_error": torch.tensor([[0.0], [1.0], [2.0], [0.0]]),
        "_fusion_color_error": torch.tensor([[0.0], [0.2], [0.4], [0.1]]),
        "_fusion_front_conflict": torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
        "_fusion_silhouette_conflict": torch.tensor([[0.0], [1.0], [1.0], [0.0]]),
        "_fusion_detail": torch.tensor([[0.0], [0.5], [1.0], [0.25]]),
    }

    feat = sparse_voxel_fusion_features(fused, radius=2.0, voxel_size=0.5)

    assert feat.shape == (4, SPARSE_VOXEL_FUSION_FEATURES)
    assert torch.isfinite(feat).all()


def test_positive_zero_centered_has_boost_gradient_at_identity():
    raw = torch.tensor([[-1.0], [0.0], [1.0]], requires_grad=True)

    out = _positive_zero_centered(raw)
    out.sum().backward()

    assert torch.allclose(out.detach().squeeze(-1), torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(raw.grad.squeeze(-1), torch.tensor([0.0, 1.0, 1.0]))


def test_dense_voxel_fusion_mlp_is_identity_at_init():
    fused = {
        "rgb": torch.rand(8, 3),
        "opacity": torch.rand(8, 1),
        "scale": torch.full((8, 3), 0.001),
        "mean": torch.randn(8, 3) * 0.1 + 0.5,
    }
    head = DenseVoxelFusionMLP(hidden=16, layers=2)

    out = head.refine(fused, voxel_size=0.05, radius=2.0)

    assert torch.allclose(out["rgb"], fused["rgb"])
    assert torch.allclose(out["opacity"], fused["opacity"])
    assert torch.allclose(out["mean"], fused["mean"])


def test_dense_voxel_context_features_are_finite_and_identity_safe():
    fused = {
        "rgb": torch.rand(5, 3),
        "opacity": torch.rand(5, 1),
        "scale": torch.full((5, 3), 0.001),
        "mean": torch.tensor([
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.50, 0.00, 0.00],
            [0.55, 0.00, 0.00],
        ]),
    }
    base = sparse_voxel_fusion_features(fused, radius=2.0, voxel_size=0.05)

    feat = dense_voxel_context_features(
        fused, base, voxel_size=0.05, neighbor_radius=1
    )
    head = DenseVoxelFusionMLP(hidden=16, layers=2, neighbor_radius=1)
    out = head.refine(fused, voxel_size=0.05, radius=2.0)

    assert feat.shape == (
        5,
        SPARSE_VOXEL_FUSION_FEATURES + DENSE_VOXEL_CONTEXT_EXTRA_FEATURES,
    )
    assert torch.isfinite(feat).all()
    assert torch.allclose(out["rgb"], fused["rgb"])
    assert torch.allclose(out["opacity"], fused["opacity"])
    assert torch.allclose(out["mean"], fused["mean"])


def test_dense_voxel_message_pairs_connect_occupied_neighbors():
    fused = {
        "mean": torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]),
    }

    pairs = dense_voxel_message_pairs(fused, voxel_size=1.0, neighbor_radius=1)
    connected = set()
    for src, dst in pairs:
        connected.update(zip(src.tolist(), dst.tolist()))

    assert (0, 0) in connected
    assert (1, 1) in connected
    assert (2, 2) in connected
    assert (0, 1) in connected
    assert (1, 0) in connected
    assert (1, 2) not in connected
    assert (2, 1) not in connected


def test_dense_voxel_message_fusion_is_identity_at_init():
    fused = {
        "rgb": torch.rand(6, 3),
        "opacity": torch.rand(6, 1),
        "scale": torch.full((6, 3), 0.001),
        "mean": torch.tensor([
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.50, 0.00, 0.00],
            [0.55, 0.00, 0.00],
            [1.50, 0.00, 0.00],
        ]),
    }
    head = DenseVoxelMessageFusionMLP(
        hidden=16, layers=2, message_radius=1, neighbor_radius=1
    )

    out = head.refine(fused, voxel_size=0.05, radius=2.0)

    assert torch.allclose(out["rgb"], fused["rgb"])
    assert torch.allclose(out["opacity"], fused["opacity"])
    assert torch.allclose(out["mean"], fused["mean"])


def test_dense_voxel_support_reg_penalizes_weak_opacity_boosts():
    fused = {
        "rgb": torch.rand(3, 3),
        "opacity": torch.full((3, 1), 0.4),
        "scale": torch.full((3, 3), 0.001),
        "mean": torch.randn(3, 3) * 0.1 + 0.5,
        "_fusion_support": torch.tensor([[0.0], [1.0], [3.0]]),
        "_fusion_conflict": torch.tensor([[2.0], [1.0], [0.0]]),
    }
    head = DenseVoxelFusionMLP(
        hidden=16,
        layers=2,
        rgb_res_scale=0.0,
        depth_res_frac=0.0,
        opacity_res_scale=0.1,
        vis_delta=0.0,
    )

    raw_identity = torch.zeros(3, 6)
    head._decode_outputs(raw_identity, fused, radius=2.0)
    assert float(head.support_reg_loss()) == 0.0

    raw_boost = torch.zeros(3, 6)
    raw_boost[:, 1] = 1.0
    head._decode_outputs(raw_boost, fused, radius=2.0)
    assert float(head.support_reg_loss()) > 0.0


def test_dense_voxel_target_visibility_loss_uses_depth_labels():
    fused_pos = {
        "rgb": torch.rand(2, 3),
        "opacity": torch.full((2, 1), 0.4),
        "scale": torch.full((2, 3), 0.001),
        "mean": torch.randn(2, 3) * 0.1 + 0.5,
        "_fusion_target_support": torch.tensor([[2.0], [3.0]]),
        "_fusion_target_conflict": torch.tensor([[0.0], [0.0]]),
    }
    fused_neg = {
        **{k: v.clone() for k, v in fused_pos.items()
           if isinstance(v, torch.Tensor)},
        "_fusion_target_support": torch.tensor([[0.0], [0.0]]),
        "_fusion_target_conflict": torch.tensor([[2.0], [3.0]]),
    }
    head = DenseVoxelFusionMLP(
        hidden=16,
        layers=2,
        rgb_res_scale=0.0,
        depth_res_frac=0.0,
        opacity_res_scale=0.3,
        vis_delta=0.0,
    )

    raw_identity = torch.zeros(2, 6)
    head._decode_outputs(raw_identity, fused_pos, radius=2.0)
    assert float(head.target_vis_loss()) < 1e-3

    head._decode_outputs(raw_identity, fused_neg, radius=2.0)
    identity_loss = float(head.target_vis_loss())
    assert identity_loss > 1.0

    raw_drop = torch.zeros(2, 6)
    raw_drop[:, 1] = -4.0
    head._decode_outputs(raw_drop, fused_neg, radius=2.0)
    assert float(head.target_vis_loss()) < identity_loss
