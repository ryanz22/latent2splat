from __future__ import annotations

import torch

from decoder.clean.train_phase2 import (
    _surface_token_proposal_losses,
    _surface_token_source_policy_losses,
)
from decoder.clean.surface_token_decoder import (
    RGBDSurfaceTokenDecoder,
    SurfaceTokenViewSelector,
    _LatentSlotBlock,
    _SlotSelfBlock,
    build_rgbd_surface_tokens,
)


def _camera(v: int, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    K = torch.tensor(
        [[80.0, 0.0, w / 2.0], [0.0, 80.0, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).repeat(v, 1, 1)
    c2w = torch.eye(4, dtype=torch.float32).repeat(v, 1, 1)
    c2w[:, 2, 3] = 2.0
    return K, c2w


def test_surface_tokens_shape_and_masking():
    torch.manual_seed(0)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    masks[:, :4] = 0.0
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)

    features, base = build_rgbd_surface_tokens(
        frames, masks, depths, K, c2w, radius=2.0, grid_h=4, grid_w=3
    )

    assert features.shape == (v * 4 * 3, RGBDSurfaceTokenDecoder.FEATURE_DIM)
    assert base["mean"].shape == (v * 4 * 3, 3)
    assert base["rgb"].shape == (v * 4 * 3, 3)
    assert base["valid"].min() >= 0.0
    assert base["valid"].max() <= 1.0


def test_surface_tokens_can_use_depth_normals_for_quaternions():
    torch.manual_seed(20)
    v, h, w = 1, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    x = torch.linspace(-0.2, 0.2, w).reshape(1, 1, w)
    depths = 1.5 + x.expand(v, h, w)
    K, c2w = _camera(v, h, w)

    _, ray_base = build_rgbd_surface_tokens(
        frames, masks, depths, K, c2w, radius=2.0, grid_h=4, grid_w=3,
        use_depth_normals=False,
    )
    _, normal_base = build_rgbd_surface_tokens(
        frames, masks, depths, K, c2w, radius=2.0, grid_h=4, grid_w=3,
        use_depth_normals=True,
    )
    _, ray_blend_base = build_rgbd_surface_tokens(
        frames, masks, depths, K, c2w, radius=2.0, grid_h=4, grid_w=3,
        use_depth_normals=True, depth_normal_blend=0.0,
    )

    def local_z(q: torch.Tensor) -> torch.Tensor:
        wq, xq, yq, zq = q.unbind(dim=-1)
        return torch.stack([
            2.0 * (xq * zq + wq * yq),
            2.0 * (yq * zq - wq * xq),
            1.0 - 2.0 * (xq.square() + yq.square()),
        ], dim=-1)

    ray_z = local_z(ray_base["quat"])
    ray_blend_z = local_z(ray_blend_base["quat"])
    normal_z = local_z(normal_base["quat"])
    assert torch.isfinite(normal_z).all()
    assert torch.allclose(ray_z, ray_blend_z, atol=5e-4)
    assert (normal_z - ray_z).abs().mean() > 1e-3


def test_learned_depth_normal_blend_has_gradients():
    torch.manual_seed(21)
    v, h, w = 1, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    x = torch.linspace(-0.2, 0.2, w).reshape(1, 1, w)
    depths = 1.5 + x.expand(v, h, w)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        depth_normal_quat=True,
        depth_normal_blend=0.25,
        learned_depth_normal_blend=True,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    loss = out["quat"][:, 1].sum()
    loss.backward()

    assert model.depth_normal_blend_logit is not None
    assert model.depth_normal_blend_logit.grad is not None
    assert torch.isfinite(model.depth_normal_blend_logit.grad)


def test_learned_depth_normal_blend_head_preserves_init_and_has_gradients():
    torch.manual_seed(22)
    v, h, w = 1, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    x = torch.linspace(-0.2, 0.2, w).reshape(1, 1, w)
    depths = 1.5 + x.expand(v, h, w)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        depth_normal_quat=True,
        depth_normal_blend=0.25,
        learned_depth_normal_blend_head=True,
        depth_normal_blend_head_scale=2.0,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    blend = out["_surface_token_depth_normal_blend"]
    assert torch.allclose(blend, torch.full_like(blend, 0.25), atol=1e-6)

    loss = out["quat"][:, 1].sum()
    loss.backward()

    assert model.depth_normal_blend_head is not None
    head_weight = model.depth_normal_blend_head[-1].weight
    assert head_weight.grad is not None
    assert torch.isfinite(head_weight.grad).all()
    assert head_weight.grad.abs().sum() > 0


def test_surface_token_decoder_identity_init_and_grad_flow():
    torch.manual_seed(1)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    masks[:, :4] = 0.0
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 4, 3)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=16,
        layers=2,
        heads=4,
        grid_h=4,
        grid_w=3,
        opacity_init=0.8,
    )

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)

    n = v * 4 * 3
    assert out["mean"].shape == (n, 3)
    assert out["quat"].shape == (n, 4)
    assert out["scale"].shape == (n, 3)
    assert out["opacity"].shape == (n, 1)
    assert out["rgb"].shape == (n, 3)
    assert torch.allclose(out["mean"], out["mean_anchor"], atol=1e-6)
    assert torch.allclose(out["mean_offset"], torch.zeros_like(out["mean_offset"]), atol=1e-6)
    assert out["opacity"].min() >= 0.0
    assert out["opacity"].max() <= 0.8 + 1e-5

    loss = out["rgb"].mean() + out["opacity"].mean() + out["mean"].square().mean()
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )


def test_surface_token_decoder_dense_latent_context_shape_and_grad():
    torch.manual_seed(2)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        latent_layers=2,
        latent_pool=2,
        latent_gate_init=0.05,
        grid_h=4,
        grid_w=3,
        source_rgb_dropout_prob=1.0,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)

    assert out["rgb"].shape == (v * 4 * 3, 3)
    assert out["opacity"].shape == (v * 4 * 3, 1)
    loss = out["rgb"].mean() + out["opacity"].mean()
    loss.backward()
    assert model.latent_gate_logit is not None
    assert model.latent_gate_logit.grad is not None
    assert torch.isfinite(model.latent_gate_logit.grad).all()


def test_surface_token_decoder_learned_scale_opacity_and_slot_refine():
    torch.manual_seed(3)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        latent_layers=1,
        slot_refine_layers=2,
        slot_refine_mlp_ratio=3,
        grid_h=4,
        grid_w=3,
        learned_scale_head=True,
        learned_scale_min_frac=1e-5,
        learned_scale_max_frac=1e-2,
        learned_opacity_prior=True,
        learned_output_scales=True,
        learned_color_affine=True,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)

    assert out["scale"].shape == (v * 4 * 3, 3)
    assert out["scale"].min() >= 2.0e-5 * 0.99
    assert out["scale"].max() <= 2.0e-2 * 1.01
    assert out["opacity"].min() >= 0.0
    assert out["opacity"].max() <= 1.0
    loss = out["scale"].mean() + out["opacity"].mean() + out["rgb"].mean()
    loss.backward()
    assert model.slot_refine_gate_logit is not None
    assert model.slot_refine_gate_logit.grad is not None
    assert torch.isfinite(model.slot_refine_gate_logit.grad).all()
    assert model.log_output_scales is not None
    assert model.log_output_scales.grad is not None
    assert torch.isfinite(model.log_output_scales.grad).all()
    assert model.color_affine is not None
    assert model.color_affine[-1].weight.grad is not None
    assert torch.isfinite(model.color_affine[-1].weight.grad).all()


def test_new_refinement_blocks_are_identity_at_init():
    torch.manual_seed(4)
    slots = torch.randn(2, 5, 32)
    latent_tokens = torch.randn(2, 7, 32)

    slot_block = _SlotSelfBlock(hidden=32, heads=4, mlp_ratio=2)
    latent_block = _LatentSlotBlock(hidden=32, heads=4, mlp_ratio=2)

    assert torch.allclose(slot_block(slots), slots, atol=1e-6)
    assert torch.allclose(latent_block(slots, latent_tokens), slots, atol=1e-6)


def test_disable_new_capacity_matches_identity_init():
    torch.manual_seed(5)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        latent_layers=1,
        slot_refine_layers=2,
        learned_scale_base=True,
        learned_opacity_bias=True,
        learned_opacity_prior=True,
        grid_h=4,
        grid_w=3,
    )
    model.eval()

    out_live = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    out_scaffold = model(
        latent, frames, masks, depths, K, c2w, radius=2.0,
        disable_new_capacity=True,
    )

    for key in ["mean", "rgb", "scale", "opacity", "quat"]:
        assert torch.allclose(out_live[key], out_scaffold[key], atol=1e-6)


def test_disable_new_capacity_excludes_detail_layer():
    torch.manual_seed(6)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        detail_layer=1,
        detail_opacity_init=0.005,
    )
    model.eval()

    out_live = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    out_scaffold = model(
        latent, frames, masks, depths, K, c2w, radius=2.0,
        disable_new_capacity=True,
    )

    n = v * 4 * 3
    assert out_live["mean"].shape[0] == 2 * n
    assert out_scaffold["mean"].shape[0] == n
    assert "_surface_token_detail" in out_live
    assert "_surface_token_detail" not in out_scaffold


def test_policy_head_identity_and_grad_flow():
    torch.manual_seed(7)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        learned_policy_head=True,
        policy_depth_res_frac=0.02,
        policy_move_res_frac=0.02,
        policy_scale_res_scale=0.5,
        policy_opacity_res_scale=0.5,
        policy_view_res_scale=0.5,
        policy_confidence_res_scale=0.5,
        policy_keep_res_scale=0.5,
        policy_coverage_scale_res_scale=0.5,
        policy_birth_res_scale=0.5,
    )
    model.train()

    out_live = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    out_scaffold = model(
        latent, frames, masks, depths, K, c2w, radius=2.0,
        disable_new_capacity=True,
    )

    for key in ["mean", "rgb", "scale", "opacity", "quat"]:
        assert torch.allclose(out_live[key], out_scaffold[key], atol=1e-6)
    assert "_surface_token_policy_view_gate" in out_live
    assert torch.allclose(
        out_live["_surface_token_policy_view_gate"],
        torch.ones_like(out_live["_surface_token_policy_view_gate"]),
        atol=1e-6,
    )

    loss = (
        out_live["mean"].square().mean()
        + out_live["scale"].mean()
        + out_live["opacity"].mean()
    )
    loss.backward()
    assert model.policy_head is not None
    assert model.policy_head[-1].weight.grad is not None
    assert torch.isfinite(model.policy_head[-1].weight.grad).all()
    assert model.policy_head[-1].weight.grad.abs().sum() > 0


def test_learned_policy_output_scales_preserve_init_and_have_gradients():
    torch.manual_seed(21)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        learned_policy_head=True,
        policy_depth_res_frac=0.006,
        policy_move_res_frac=0.004,
        policy_scale_res_scale=0.25,
        policy_opacity_res_scale=0.35,
        policy_view_res_scale=0.35,
        policy_confidence_res_scale=0.35,
        policy_keep_res_scale=0.35,
        policy_coverage_scale_res_scale=0.35,
        policy_birth_res_scale=0.35,
        learned_policy_output_scales=True,
    )
    model.train()

    out_live = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    out_scaffold = model(
        latent, frames, masks, depths, K, c2w, radius=2.0,
        disable_new_capacity=True,
    )
    for key in ["mean", "rgb", "scale", "opacity", "quat"]:
        assert torch.allclose(out_live[key], out_scaffold[key], atol=1e-6)
    assert model.policy_log_output_scales is not None
    init = model.policy_log_output_scales.detach().exp()
    expected = torch.tensor([0.006, 0.004, 0.25, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35])
    assert torch.allclose(init.cpu(), expected, atol=1e-6)

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.policy_head[-1].bias[0] = 0.5
        model.policy_head[-1].bias[4:13] = 0.5
    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    loss = out["mean"].square().mean() + out["scale"].mean() + out["opacity"].mean()
    loss.backward()
    assert model.policy_log_output_scales.grad is not None
    assert torch.isfinite(model.policy_log_output_scales.grad).all()
    assert model.policy_log_output_scales.grad.abs().sum() > 0


def test_learned_source_depth_confidence_preserves_init_and_has_gradients():
    torch.manual_seed(22)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        learned_source_depth_confidence_head=True,
        source_depth_res_frac=0.004,
        source_confidence_res_scale=0.35,
        learned_source_depth_confidence_scales=True,
    )
    model.train()

    out_live = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    out_scaffold = model(
        latent, frames, masks, depths, K, c2w, radius=2.0,
        disable_new_capacity=True,
    )
    for key in ["mean", "rgb", "scale", "opacity", "quat"]:
        assert torch.allclose(out_live[key], out_scaffold[key], atol=1e-6)
    assert "_surface_token_source_depth_res" in out_live
    assert "_surface_token_source_confidence_gate" in out_live
    assert torch.allclose(
        out_live["_surface_token_source_depth_res"],
        torch.zeros_like(out_live["_surface_token_source_depth_res"]),
        atol=1e-6,
    )
    assert torch.allclose(
        out_live["_surface_token_source_confidence_gate"],
        torch.ones_like(out_live["_surface_token_source_confidence_gate"]),
        atol=1e-6,
    )
    assert model.source_depth_confidence_log_scales is not None
    init = model.source_depth_confidence_log_scales.detach().exp()
    assert torch.allclose(init.cpu(), torch.tensor([0.004, 0.35]), atol=1e-6)

    loss = out_live["mean"].square().mean() + out_live["opacity"].mean()
    loss.backward()
    assert model.source_depth_confidence_head is not None
    assert model.source_depth_confidence_head[-1].weight.grad is not None
    assert torch.isfinite(model.source_depth_confidence_head[-1].weight.grad).all()
    assert model.source_depth_confidence_head[-1].weight.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.source_depth_confidence_head[-1].bias[:] = torch.tensor([0.5, 0.5])
    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    loss = out["mean"].square().mean() + out["opacity"].mean()
    loss.backward()
    assert model.source_depth_confidence_log_scales.grad is not None
    assert torch.isfinite(model.source_depth_confidence_log_scales.grad).all()
    assert model.source_depth_confidence_log_scales.grad.abs().sum() > 0


def test_view_selector_preserves_prior_and_gate_grad_flow():
    torch.manual_seed(8)
    v, h, w = 6, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    masks[1::2, :4] = 0.0
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    prior_ids = torch.tensor([0, 2, 5])
    selector = SurfaceTokenViewSelector(
        hidden=32,
        score_scale=0.5,
        gate_scale=0.75,
    )
    selector.train()

    picked = selector.select(
        latent, frames, masks, depths, K, c2w, radius=2.0,
        k=3, prior_ids=prior_ids,
    )

    assert torch.equal(picked["ids"].cpu(), prior_ids)
    assert torch.allclose(picked["gates"], torch.ones_like(picked["gates"]), atol=1e-6)
    loss = picked["gates"].square().mean()
    loss.backward()
    assert selector.head[-1].weight.grad is not None
    assert torch.isfinite(selector.head[-1].weight.grad).all()
    assert selector.head[-1].weight.grad.abs().sum() > 0


def test_learned_proposals_preserve_scaffold_and_train_opacity():
    torch.manual_seed(9)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    proposal_count = 5
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=proposal_count,
        proposal_opacity_init=5e-4,
        detail_layer=1,
        detail_opacity_init=0.005,
    )
    model.train()

    out_live = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    out_scaffold = model(
        latent, frames, masks, depths, K, c2w, radius=2.0,
        disable_new_capacity=True,
    )

    base_n = out_scaffold["mean"].shape[0]
    assert out_live["mean"].shape[0] == base_n * 2 + proposal_count
    for key in ["mean", "rgb", "scale", "opacity", "quat"]:
        assert torch.allclose(out_live[key][:base_n], out_scaffold[key], atol=1e-6)
    assert "_surface_token_proposal" in out_live
    assert out_live["_surface_token_proposal"].shape[0] == out_live["mean"].shape[0]
    assert torch.all(out_live["_surface_token_proposal"][-proposal_count:] > 0.5)
    prop_opacity = out_live["opacity"][-proposal_count:]
    assert torch.all(prop_opacity > 0.0)
    assert torch.all(prop_opacity < 1e-3)

    prop_opacity.sum().backward()
    assert model.proposal_head is not None
    assert model.proposal_head[-1].weight.grad is not None
    assert torch.isfinite(model.proposal_head[-1].weight.grad).all()
    assert model.proposal_head[-1].weight.grad.abs().sum() > 0
    assert model.proposal_opacity_logit_bias.grad is not None
    assert model.proposal_opacity_logit_bias.grad.abs().sum() > 0


def test_surface_token_proposal_losses_train_mean_opacity_and_rgb():
    torch.manual_seed(10)
    v, h, w = 1, 8, 8
    frames = torch.zeros(v, h, w, 3)
    frames[..., 0] = 1.0
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 2.0)
    K, c2w = _camera(v, h, w)
    c2w[:, 2, 3] = 0.0
    mean = torch.tensor(
        [[0.0, 0.0, -2.0], [1.0, 1.0, -2.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    opacity = torch.full((2, 1), 0.01, dtype=torch.float32, requires_grad=True)
    rgb = torch.full((2, 3), 0.25, dtype=torch.float32, requires_grad=True)
    keep_gate = torch.ones((2, 1), dtype=torch.float32, requires_grad=True)
    confidence_gate = torch.ones((2, 1), dtype=torch.float32, requires_grad=True)
    coverage_gate = torch.ones((2, 1), dtype=torch.float32, requires_grad=True)
    params = {
        "mean": mean,
        "opacity": opacity,
        "rgb": rgb,
        "_surface_token_proposal": torch.ones(2, 1),
        "_surface_token_proposal_policy_keep_gate": keep_gate,
        "_surface_token_proposal_policy_confidence_gate": confidence_gate,
        "_surface_token_proposal_policy_coverage_mult": coverage_gate,
    }

    losses = _surface_token_proposal_losses(
        params,
        frames,
        masks,
        depths,
        K,
        c2w,
        radius=2.0,
        cover_points=64,
        depth_tol_frac=0.25,
        fg_threshold=0.5,
    )
    total = (
        losses["cover"]
        + losses["surface"]
        + losses["opacity"]
        + losses["rgb"]
        + losses["policy_keep"]
        + losses["policy_confidence"]
        + losses["policy_coverage"]
    )
    total.backward()

    assert losses["cover"] > 0
    assert losses["surface"] > 0
    assert losses["opacity"] > 0
    assert losses["rgb"] > 0
    assert losses["policy_keep"] > 0
    assert losses["policy_confidence"] > 0
    assert losses["policy_coverage"] > 0
    assert losses["policy_keep_target_mean"] > 0
    assert losses["policy_confidence_target_mean"] > 0
    assert losses["policy_coverage_target_mean"] > 0
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert mean.grad.abs().sum() > 0
    assert opacity.grad is not None
    assert torch.isfinite(opacity.grad).all()
    assert opacity.grad.abs().sum() > 0
    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()
    assert rgb.grad.abs().sum() > 0
    assert keep_gate.grad is not None
    assert torch.isfinite(keep_gate.grad).all()
    assert keep_gate.grad.abs().sum() > 0
    assert confidence_gate.grad is not None
    assert torch.isfinite(confidence_gate.grad).all()
    assert confidence_gate.grad.abs().sum() > 0
    assert coverage_gate.grad is not None
    assert torch.isfinite(coverage_gate.grad).all()
    assert coverage_gate.grad.abs().sum() > 0


def test_surface_token_source_policy_losses_train_confidence_and_depth():
    torch.manual_seed(11)
    v, h, w = 1, 8, 8
    frames = torch.zeros(v, h, w, 3)
    frames[..., 0] = 1.0
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 2.0)
    K, c2w = _camera(v, h, w)
    c2w[:, 2, 3] = 0.0
    mean = torch.tensor(
        [[0.0, 0.0, -1.8], [1.0, 1.0, -2.5]],
        dtype=torch.float32,
    )
    source_depth_res = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)
    source_confidence_gate = torch.ones((2, 1), dtype=torch.float32, requires_grad=True)
    params = {
        "mean": mean,
        "opacity": torch.ones(2, 1),
        "rgb": torch.ones(2, 3),
        "_surface_token_valid": torch.ones(2, 1),
        "_surface_token_mask": torch.ones(2, 1),
        "_surface_token_source_depth_res": source_depth_res,
        "_surface_token_source_confidence_gate": source_confidence_gate,
        "_surface_token_source_base_mean": mean.detach(),
        "_surface_token_source_direction": torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
        ),
        "_surface_token_proposal": torch.zeros(2, 1),
    }

    losses = _surface_token_source_policy_losses(
        params,
        frames,
        masks,
        depths,
        K,
        c2w,
        radius=2.0,
        source_points=64,
        target_points=64,
        depth_tol_frac=0.25,
        fg_threshold=0.5,
    )
    total = losses["confidence"] + losses["depth"]
    total.backward()

    assert losses["support_mean"] > 0
    assert losses["confidence"] > 0
    assert losses["depth"] > 0
    assert (losses["confidence_target_mean"] - 1.0).abs() > 1e-3
    assert losses["depth_target_abs_frac"] > 0
    assert source_confidence_gate.grad is not None
    assert torch.isfinite(source_confidence_gate.grad).all()
    assert source_confidence_gate.grad.abs().sum() > 0
    assert source_depth_res.grad is not None
    assert torch.isfinite(source_depth_res.grad).all()
    assert source_depth_res.grad.abs().sum() > 0


def test_surface_token_source_policy_projective_support_trains_confidence():
    torch.manual_seed(12)
    v, h, w = 1, 8, 8
    frames = torch.zeros(v, h, w, 3)
    frames[..., 0] = 1.0
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 2.0)
    K, c2w = _camera(v, h, w)
    c2w[:, 2, 3] = 0.0
    mean = torch.tensor(
        [[0.0, 0.0, -2.0], [0.0, 0.0, -1.0]],
        dtype=torch.float32,
    )
    source_confidence_gate = torch.ones((2, 1), dtype=torch.float32, requires_grad=True)
    params = {
        "mean": mean,
        "opacity": torch.ones(2, 1),
        "rgb": torch.ones(2, 3),
        "_surface_token_valid": torch.ones(2, 1),
        "_surface_token_mask": torch.ones(2, 1),
        "_surface_token_source_depth_res": torch.zeros(2, 1, requires_grad=True),
        "_surface_token_source_confidence_gate": source_confidence_gate,
        "_surface_token_source_base_mean": mean.detach(),
        "_surface_token_source_direction": torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
        ),
        "_surface_token_proposal": torch.zeros(2, 1),
    }

    losses = _surface_token_source_policy_losses(
        params,
        frames,
        masks,
        depths,
        K,
        c2w,
        radius=2.0,
        source_points=64,
        target_points=64,
        depth_tol_frac=0.1,
        fg_threshold=0.5,
        support_mode="projective",
    )
    losses["confidence"].backward()

    assert 0.0 < losses["support_mean"] < 1.0
    assert losses["confidence"] > 0
    assert source_confidence_gate.grad is not None
    assert torch.isfinite(source_confidence_gate.grad).all()
    assert source_confidence_gate.grad.abs().sum() > 0


def test_surface_token_proposal_policy_target_modes():
    torch.manual_seed(15)
    v, h, w = 1, 8, 8
    frames = torch.zeros(v, h, w, 3)
    frames[..., 1] = 1.0
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 2.0)
    K, c2w = _camera(v, h, w)
    c2w[:, 2, 3] = 0.0
    base_params = {
        "mean": torch.tensor([[0.0, 0.0, -2.0], [0.8, 0.8, -2.3]], dtype=torch.float32),
        "opacity": torch.full((2, 1), 0.5, dtype=torch.float32),
        "rgb": torch.full((2, 3), 0.25, dtype=torch.float32),
        "_surface_token_proposal": torch.ones(2, 1),
    }
    none_params = dict(base_params)
    none_params.update({
        "_surface_token_proposal_policy_keep_gate": torch.full((2, 1), 1.2, requires_grad=True),
        "_surface_token_proposal_policy_confidence_gate": torch.full((2, 1), 0.8, requires_grad=True),
        "_surface_token_proposal_policy_coverage_mult": torch.full((2, 1), 1.1, requires_grad=True),
    })

    none_losses = _surface_token_proposal_losses(
        none_params,
        frames,
        masks,
        depths,
        K,
        c2w,
        radius=2.0,
        cover_points=64,
        depth_tol_frac=0.25,
        fg_threshold=0.5,
        policy_target_mode="none",
    )

    assert none_losses["policy_keep"] == 0
    assert none_losses["policy_confidence"] == 0
    assert none_losses["policy_coverage"] == 0
    assert torch.allclose(none_losses["policy_keep_target_mean"], torch.tensor(1.0))
    assert torch.allclose(none_losses["policy_coverage_target_mean"], torch.tensor(1.0))

    identity_params = dict(base_params)
    keep_gate = torch.full((2, 1), 1.2, requires_grad=True)
    confidence_gate = torch.full((2, 1), 0.8, requires_grad=True)
    coverage_gate = torch.full((2, 1), 1.1, requires_grad=True)
    identity_params.update({
        "_surface_token_proposal_policy_keep_gate": keep_gate,
        "_surface_token_proposal_policy_confidence_gate": confidence_gate,
        "_surface_token_proposal_policy_coverage_mult": coverage_gate,
    })

    identity_losses = _surface_token_proposal_losses(
        identity_params,
        frames,
        masks,
        depths,
        K,
        c2w,
        radius=2.0,
        cover_points=64,
        depth_tol_frac=0.25,
        fg_threshold=0.5,
        policy_target_mode="identity",
    )
    policy_loss = (
        identity_losses["policy_keep"]
        + identity_losses["policy_confidence"]
        + identity_losses["policy_coverage"]
    )
    policy_loss.backward()

    assert identity_losses["policy_keep"] > 0
    assert identity_losses["policy_confidence"] > 0
    assert identity_losses["policy_coverage"] > 0
    assert torch.allclose(identity_losses["policy_keep_target_mean"], torch.tensor(1.0))
    assert keep_gate.grad is not None and keep_gate.grad.abs().sum() > 0
    assert confidence_gate.grad is not None and confidence_gate.grad.abs().sum() > 0
    assert coverage_gate.grad is not None and coverage_gate.grad.abs().sum() > 0


def test_surface_token_proposal_detail_cover_targets_rgb_edges():
    torch.manual_seed(16)
    v, h, w = 1, 8, 8
    frames = torch.zeros(v, h, w, 3)
    frames[:, :, : w // 2, 0] = 1.0
    frames[:, :, w // 2 :, 1] = 1.0
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 2.0)
    K, c2w = _camera(v, h, w)
    c2w[:, 2, 3] = 0.0
    mean = torch.tensor(
        [[0.7, 0.0, -2.0], [0.9, 0.4, -2.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    params = {
        "mean": mean,
        "opacity": torch.full((2, 1), 0.5, dtype=torch.float32),
        "rgb": torch.full((2, 3), 0.25, dtype=torch.float32),
        "_surface_token_proposal": torch.ones(2, 1),
    }

    losses = _surface_token_proposal_losses(
        params,
        frames,
        masks,
        depths,
        K,
        c2w,
        radius=2.0,
        cover_points=64,
        depth_tol_frac=0.25,
        fg_threshold=0.5,
        policy_target_mode="none",
        detail_edge_thresh=0.01,
    )
    losses["detail_cover"].backward()

    assert losses["detail_mean"] > 0
    assert losses["detail_cover"] > 0
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert mean.grad.abs().sum() > 0


def test_surface_seeded_proposals_are_anchored_to_rgbd_surface():
    torch.manual_seed(11)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    proposal_count = 6
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=proposal_count,
        proposal_seed_surface=True,
        proposal_seed_pool=8,
        proposal_opacity_init=5e-4,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    scaffold = model(
        latent, frames, masks, depths, K, c2w, radius=2.0,
        disable_new_capacity=True,
    )
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5

    assert prop.sum().item() == proposal_count
    assert "_surface_token_proposal_surface_seed" in out
    assert torch.all(out["_surface_token_proposal_surface_seed"][prop] > 0.5)
    anchor = out["mean_anchor"][prop]
    lo = scaffold["mean"].min(dim=0).values - 1e-4
    hi = scaffold["mean"].max(dim=0).values + 1e-4
    assert torch.all(anchor >= lo)
    assert torch.all(anchor <= hi)
    loss = out["mean"][prop].square().mean() + out["rgb"][prop].mean()
    loss.backward()
    assert model.proposal_from_surface is not None
    assert model.proposal_from_surface.in_proj_weight.grad is not None
    assert torch.isfinite(model.proposal_from_surface.in_proj_weight.grad).all()
    assert model.proposal_from_surface.in_proj_weight.grad.abs().sum() > 0


def test_learned_surface_anchor_selector_has_st_gradients():
    torch.manual_seed(12)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    proposal_count = 6
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=proposal_count,
        proposal_seed_surface=True,
        proposal_seed_pool=8,
        proposal_anchor_mode="learned_st",
        proposal_anchor_gate_init=0.5,
        proposal_anchor_mix_res_scale=2.0,
        proposal_opacity_init=5e-4,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5

    assert prop.sum().item() == proposal_count
    assert "_surface_token_proposal_anchor_mix" in out
    assert "_surface_token_proposal_anchor_entropy" in out
    assert "_surface_token_proposal_anchor_entropy_loss" in out
    assert "_surface_token_proposal_anchor_usage_loss" in out
    assert "_surface_token_proposal_anchor_usage_perplexity" in out
    assert "_surface_token_proposal_anchor_unique_frac" in out
    assert "_surface_token_proposal_anchor_collision_loss" in out
    assert "_surface_token_proposal_anchor_collision_frac" in out
    mix = out["_surface_token_proposal_anchor_mix"][prop]
    entropy = out["_surface_token_proposal_anchor_entropy"][prop]
    assert torch.allclose(mix, torch.full_like(mix, 0.5), atol=1e-6)
    assert torch.all(entropy >= 0.0)
    assert torch.all(entropy <= 1.0 + 1e-5)
    assert torch.isfinite(out["_surface_token_proposal_anchor_entropy_loss"])
    assert torch.isfinite(out["_surface_token_proposal_anchor_usage_loss"])
    assert 0.0 <= float(out["_surface_token_proposal_anchor_usage_perplexity"].detach()) <= 1.0 + 1e-5
    assert 0.0 <= float(out["_surface_token_proposal_anchor_unique_frac"].detach()) <= 1.0
    assert float(out["_surface_token_proposal_anchor_collision_loss"].detach()) >= 0.0
    assert 0.0 <= float(out["_surface_token_proposal_anchor_collision_frac"].detach()) <= 1.0

    target = out["mean_anchor"][prop].detach().roll(1, dims=0)
    loss = (out["mean_anchor"][prop] - target).square().mean()
    loss.backward()
    assert model.proposal_anchor_q is not None
    assert model.proposal_anchor_k is not None
    q_weight = model.proposal_anchor_q[1].weight
    k_weight = model.proposal_anchor_k[1].weight
    assert q_weight.grad is not None
    assert k_weight.grad is not None
    assert torch.isfinite(q_weight.grad).all()
    assert torch.isfinite(k_weight.grad).all()
    assert q_weight.grad.abs().sum() > 0
    assert k_weight.grad.abs().sum() > 0
    assert model.proposal_anchor_mix_logit is not None
    assert model.proposal_anchor_mix_logit.grad is not None
    assert model.proposal_anchor_mix_head is not None
    mix_weight = model.proposal_anchor_mix_head[-1].weight
    assert mix_weight.grad is not None
    assert torch.isfinite(mix_weight.grad).all()
    assert mix_weight.grad.abs().sum() > 0


def test_learned_surface_anchor_even_prior_preserves_initial_coverage():
    torch.manual_seed(13)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    proposal_count = 6
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=proposal_count,
        proposal_seed_surface=True,
        proposal_seed_pool=8,
        proposal_anchor_mode="learned_st",
        proposal_anchor_gate_init=0.5,
        proposal_anchor_mix_res_scale=2.0,
        proposal_anchor_even_prior=8.0,
        proposal_opacity_init=5e-4,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5

    assert prop.sum().item() == proposal_count
    assert float(out["_surface_token_proposal_anchor_unique_frac"].detach()) == 1.0
    assert float(out["_surface_token_proposal_anchor_entropy"].detach()[prop].mean()) < 1e-3


def test_learned_local_surface_anchor_preserves_stratified_coverage_without_global_prior():
    torch.manual_seed(15)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    proposal_count = 6
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=proposal_count,
        proposal_seed_surface=True,
        proposal_seed_pool=24,
        proposal_anchor_mode="learned_local_st",
        proposal_anchor_local_window=3,
        proposal_anchor_gate_init=1.0 - 1e-4,
        proposal_anchor_even_prior=0.0,
        proposal_opacity_init=5e-4,
    )
    with torch.no_grad():
        model.proposal_anchor_q[1].weight.zero_()
        model.proposal_anchor_q[1].bias.zero_()
        model.proposal_anchor_k[1].weight.zero_()
        model.proposal_anchor_k[1].bias.zero_()
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5

    assert prop.sum().item() == proposal_count
    assert float(out["_surface_token_proposal_anchor_unique_frac"].detach()) == 1.0
    entropy = out["_surface_token_proposal_anchor_entropy"].detach()[prop]
    assert torch.all(entropy > 0.0)
    assert torch.all(entropy <= 1.0 + 1e-5)


def test_learned_local_surface_anchor_has_st_gradients():
    torch.manual_seed(16)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    proposal_count = 6
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=proposal_count,
        proposal_seed_surface=True,
        proposal_seed_pool=24,
        proposal_anchor_mode="learned_local_st",
        proposal_anchor_local_window=5,
        proposal_anchor_gate_init=0.5,
        proposal_anchor_mix_res_scale=2.0,
        proposal_anchor_even_prior=0.0,
        proposal_opacity_init=5e-4,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5
    target = out["mean_anchor"][prop].detach().roll(1, dims=0)
    loss = (out["mean_anchor"][prop] - target).square().mean()
    loss.backward()

    assert model.proposal_anchor_q is not None
    assert model.proposal_anchor_k is not None
    q_weight = model.proposal_anchor_q[1].weight
    k_weight = model.proposal_anchor_k[1].weight
    assert q_weight.grad is not None
    assert k_weight.grad is not None
    assert torch.isfinite(q_weight.grad).all()
    assert torch.isfinite(k_weight.grad).all()
    assert q_weight.grad.abs().sum() > 0
    assert k_weight.grad.abs().sum() > 0


def test_learned_local_unique_surface_anchor_preserves_center_scaffold_at_tie():
    torch.manual_seed(18)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    proposal_count = 6
    seed_pool = 24
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=proposal_count,
        proposal_seed_surface=True,
        proposal_seed_pool=seed_pool,
        proposal_anchor_mode="learned_local_unique_st",
        proposal_anchor_local_window=5,
        proposal_anchor_gate_init=1.0 - 1e-4,
        proposal_anchor_even_prior=0.0,
        proposal_opacity_init=5e-4,
    )
    with torch.no_grad():
        model.proposal_anchor_q[1].weight.zero_()
        model.proposal_anchor_q[1].bias.zero_()
        model.proposal_anchor_k[1].weight.zero_()
        model.proposal_anchor_k[1].bias.zero_()
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5
    _, base = build_rgbd_surface_tokens(
        frames, masks, depths, K, c2w, radius=2.0, grid_h=4, grid_w=3
    )
    idx = ((base["valid"] * base["mask"]).reshape(-1) > 0.5).nonzero(
        as_tuple=False
    ).reshape(-1)
    take = torch.linspace(0, idx.numel() - 1, seed_pool).round().to(dtype=torch.long)
    cand_idx = idx.index_select(0, take)
    pick = torch.linspace(0, cand_idx.numel() - 1, proposal_count).round().to(dtype=torch.long)
    expected = base["mean"].index_select(0, cand_idx.index_select(0, pick))

    assert prop.sum().item() == proposal_count
    assert torch.allclose(out["mean_anchor"][prop], expected, atol=1e-4)
    assert float(out["_surface_token_proposal_anchor_unique_frac"].detach()) == 1.0
    assert float(out["_surface_token_proposal_anchor_collision_frac"].detach()) == 0.0


def test_learned_local_unique_surface_anchor_has_st_gradients():
    torch.manual_seed(19)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=6,
        proposal_seed_surface=True,
        proposal_seed_pool=24,
        proposal_anchor_mode="learned_local_unique_st",
        proposal_anchor_local_window=5,
        proposal_anchor_gate_init=0.5,
        proposal_anchor_mix_res_scale=2.0,
        proposal_anchor_even_prior=0.0,
        proposal_opacity_init=5e-4,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5
    target = out["mean_anchor"][prop].detach().roll(1, dims=0)
    loss = (out["mean_anchor"][prop] - target).square().mean()
    loss.backward()

    assert model.proposal_anchor_q is not None
    assert model.proposal_anchor_k is not None
    q_weight = model.proposal_anchor_q[1].weight
    k_weight = model.proposal_anchor_k[1].weight
    assert q_weight.grad is not None
    assert k_weight.grad is not None
    assert torch.isfinite(q_weight.grad).all()
    assert torch.isfinite(k_weight.grad).all()
    assert q_weight.grad.abs().sum() > 0
    assert k_weight.grad.abs().sum() > 0


def test_learned_surface_anchor_collision_signal_detects_overfull_anchor_pool():
    torch.manual_seed(17)
    v, h, w = 1, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=8,
        proposal_seed_surface=True,
        proposal_seed_pool=4,
        proposal_anchor_mode="learned_st",
        proposal_anchor_gate_init=0.5,
        proposal_anchor_even_prior=0.0,
        proposal_opacity_init=5e-4,
    )
    with torch.no_grad():
        model.proposal_anchor_q[1].weight.zero_()
        model.proposal_anchor_q[1].bias.zero_()
        model.proposal_anchor_k[1].weight.zero_()
        model.proposal_anchor_k[1].bias.zero_()
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)

    assert float(out["_surface_token_proposal_anchor_collision_loss"].detach()) > 0.0
    assert float(out["_surface_token_proposal_anchor_collision_frac"].detach()) > 0.0


def test_proposal_policy_head_zero_init_noop_and_has_gradients():
    torch.manual_seed(14)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    proposal_count = 6
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=proposal_count,
        proposal_seed_surface=True,
        proposal_seed_pool=8,
        proposal_anchor_mode="learned_st",
        proposal_anchor_gate_init=0.5,
        proposal_anchor_even_prior=8.0,
        learned_proposal_policy_head=True,
        proposal_policy_keep_res_scale=0.75,
        proposal_policy_confidence_res_scale=0.75,
        proposal_policy_coverage_res_scale=0.75,
        proposal_opacity_init=5e-4,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5

    assert prop.sum().item() == proposal_count
    keep = out["_surface_token_proposal_policy_keep_gate"][prop]
    conf = out["_surface_token_proposal_policy_confidence_gate"][prop]
    cov = out["_surface_token_proposal_policy_coverage_mult"][prop]
    assert torch.allclose(keep, torch.ones_like(keep))
    assert torch.allclose(conf, torch.ones_like(conf))
    assert torch.allclose(cov, torch.ones_like(cov))

    loss = out["opacity"][prop].mean() + out["scale"][prop].mean()
    loss.backward()

    assert model.proposal_policy_head is not None
    policy_weight = model.proposal_policy_head[-1].weight
    assert policy_weight.grad is not None
    assert torch.isfinite(policy_weight.grad).all()
    assert policy_weight.grad.abs().sum() > 0


def test_learned_proposal_scale_base_preserves_init_and_has_gradients():
    torch.manual_seed(20)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    scale_frac = 0.0012
    normal_frac = 0.00012
    radius = 2.0
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=6,
        proposal_seed_surface=True,
        proposal_seed_pool=8,
        proposal_scale_frac=scale_frac,
        proposal_normal_scale_frac=normal_frac,
        learned_proposal_scale_base=True,
        proposal_opacity_init=5e-4,
    )
    with torch.no_grad():
        model.proposal_head[-1].weight.zero_()
        model.proposal_head[-1].bias.zero_()
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=radius)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5
    expected = torch.tensor(
        [scale_frac * radius, scale_frac * radius, normal_frac * radius],
        dtype=out["scale"].dtype,
    ).reshape(1, 3)

    assert model.proposal_log_scale_base is not None
    assert torch.allclose(out["scale"][prop], expected.expand_as(out["scale"][prop]), atol=1e-7)
    loss = out["scale"][prop].sum()
    loss.backward()
    assert model.proposal_log_scale_base.grad is not None
    assert torch.isfinite(model.proposal_log_scale_base.grad).all()
    assert model.proposal_log_scale_base.grad.abs().sum() > 0


def test_learned_proposal_scale_head_preserves_init_and_has_gradients():
    torch.manual_seed(21)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    scale_frac = 0.0012
    normal_frac = 0.00012
    radius = 2.0
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=6,
        proposal_seed_surface=True,
        proposal_seed_pool=8,
        proposal_scale_frac=scale_frac,
        proposal_normal_scale_frac=normal_frac,
        learned_proposal_scale_head=True,
        learned_proposal_scale_min_frac=1e-5,
        learned_proposal_scale_max_frac=3e-2,
        proposal_opacity_init=5e-4,
    )
    with torch.no_grad():
        model.proposal_head[-1].weight.zero_()
        keep_bias = model.proposal_head[-1].bias.detach().clone()
        model.proposal_head[-1].bias.zero_()
        model.proposal_head[-1].bias[6:9] = keep_bias[6:9]
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=radius)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5
    expected = torch.tensor(
        [scale_frac * radius, scale_frac * radius, normal_frac * radius],
        dtype=out["scale"].dtype,
    ).reshape(1, 3)

    assert torch.allclose(out["scale"][prop], expected.expand_as(out["scale"][prop]), atol=1e-6)
    loss = out["scale"][prop].sum()
    loss.backward()
    scale_bias_grad = model.proposal_head[-1].bias.grad[6:9]
    assert torch.isfinite(scale_bias_grad).all()
    assert scale_bias_grad.abs().sum() > 0


def test_free_proposals_report_not_surface_seeded():
    torch.manual_seed(22)
    v, h, w = 2, 16, 12
    frames = torch.rand(v, h, w, 3)
    masks = torch.ones(v, h, w, 1)
    depths = torch.full((v, h, w), 1.5)
    K, c2w = _camera(v, h, w)
    latent = torch.randn(128, 2, 6, 4)
    model = RGBDSurfaceTokenDecoder(
        hidden=64,
        slots=12,
        layers=1,
        heads=4,
        grid_h=4,
        grid_w=3,
        proposal_count=6,
        proposal_seed_surface=False,
        proposal_opacity_init=5e-4,
    )
    model.train()

    out = model(latent, frames, masks, depths, K, c2w, radius=2.0)
    prop = out["_surface_token_proposal"].reshape(-1) > 0.5

    assert prop.sum().item() == 6
    assert torch.all(out["_surface_token_proposal_surface_seed"][prop] == 0)
