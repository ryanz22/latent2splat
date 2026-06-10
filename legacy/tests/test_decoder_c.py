"""Method C (pixel-aligned depth) unprojection + shape tests.

The unprojection is correctness-critical (like the camera conversion): if rays
or depths are wrong, every Gaussian is misplaced. We verify against the
documented orbit geometry (radius 0.95, fov_y 49.1, 512x768) WITHOUT needing
the dataset.
"""
from __future__ import annotations

import math

import torch

from legacy.decoder_c import PixelAlignedDecoder, ray_dirs_world
from legacy.decoder_a import LATENT_H, LATENT_W


def _frame0_camera():
    """Frame-0 OpenGL camera from the documented orbit: azimuth 0, elev 15,
    radius 0.95, fov_y 49.1, 512x768."""
    width, height, fov_y, radius, elev = 512, 768, 49.1, 0.95, 15.0
    az, el = 0.0, math.radians(elev)
    eye = torch.tensor([radius * math.cos(el) * math.cos(az),
                        radius * math.sin(el),
                        radius * math.cos(el) * math.sin(az)])
    fwd = -eye / eye.norm()
    up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.linalg.cross(fwd, up); right = right / right.norm()
    true_up = torch.linalg.cross(right, fwd)
    c2w = torch.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -fwd
    c2w[:3, 3] = eye
    fy = (height / 2) / math.tan(math.radians(fov_y) / 2)
    K = torch.tensor([[fy, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1.0]])
    return K, c2w, radius


def test_center_ray_hits_origin():
    """The ray through the latent-grid cell nearest image center, at
    depth=radius, must land near the world origin (the object center)."""
    K, c2w, radius = _frame0_camera()
    dirs = ray_dirs_world(K, c2w, LATENT_H, LATENT_W)  # (H*W,3)
    cam_center = c2w[:3, 3]
    hits = cam_center[None] + radius * dirs            # (H*W,3) at depth=radius
    # the closest hit to origin should be very close (grid won't sample the
    # exact center pixel, but one cell is near it)
    min_dist = hits.norm(dim=-1).min()
    assert min_dist < 0.15, f"no ray lands near origin at depth=radius; min={min_dist}"


def test_rays_are_unit_and_point_inward():
    K, c2w, radius = _frame0_camera()
    dirs = ray_dirs_world(K, c2w, LATENT_H, LATENT_W)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(dirs.shape[0]), atol=1e-5)
    # rays must point from the camera toward the scene (origin side): the dot
    # of each ray with (origin - cam_center) should be positive on average.
    to_origin = (-c2w[:3, 3]) / c2w[:3, 3].norm()
    assert (dirs @ to_origin).mean() > 0.8


def test_decoder_c_shapes_and_means_on_rays():
    K, c2w, radius = _frame0_camera()
    dec = PixelAlignedDecoder(dim=64, depth_layers=2, heads=4, k=2,
                              radius=radius, depth_halfrange=0.6, scale_max=0.05)
    out = dec(torch.randn(1, 128, 2, LATENT_H, LATENT_W), K, c2w)
    n = LATENT_H * LATENT_W * 2
    assert out["mean"].shape == (1, n, 3)
    assert out["scale"].shape == (1, n, 3)
    # every mean must lie within depth_halfrange of the radius shell along a
    # ray from the camera: its distance from cam_center is in [r-hr, r+hr].
    cam_center = c2w[:3, 3]
    d = (out["mean"][0] - cam_center[None]).norm(dim=-1)
    assert (d >= radius - 0.6 - 1e-4).all() and (d <= radius + 0.6 + 1e-4).all()
    assert (out["scale"] > 0).all() and (out["scale"] < 0.05 + 1e-6).all()
    assert (out["opacity"] >= 0).all() and (out["opacity"] <= 1).all()


def test_decoder_c_opacity_starts_visible():
    K, c2w, radius = _frame0_camera()
    dec = PixelAlignedDecoder(dim=64, depth_layers=2, heads=4, k=2, radius=radius)
    # zero latent -> head bias dominates -> opacity should start high
    out = dec(torch.zeros(1, 128, 2, LATENT_H, LATENT_W), K, c2w)
    assert out["opacity"].mean() > 0.8


def test_decoder_c_handles_either_temporal_dim():
    """Method C mean-pools over T, so it must accept v2 (T=2) and v1 (T=4)."""
    K, c2w, radius = _frame0_camera()
    dec = PixelAlignedDecoder(dim=64, depth_layers=2, heads=4, k=2, radius=radius)
    out4 = dec(torch.randn(1, 128, 4, LATENT_H, LATENT_W), K, c2w)
    out2 = dec(torch.randn(1, 128, 2, LATENT_H, LATENT_W), K, c2w)
    assert out4["mean"].shape == out2["mean"].shape == (1, LATENT_H * LATENT_W * 2, 3)
