"""Geometry utils for the clean-slate decoder: depth bounds + Plücker embedding.
Pure CPU math, no gsplat/CUDA needed."""
from __future__ import annotations

import math
import torch

from decoder.clean.geometry import depth_bounds, plucker_embedding, ray_dirs_world


def _c2w(radius=1.52):
    # camera on +Z axis at distance `radius`, looking down -Z (OpenGL), +Y up
    c2w = torch.eye(4)
    c2w[2, 3] = radius
    return c2w


def _K():
    return torch.tensor([[840.0, 0, 256.0], [0, 840.0, 384.0], [0, 0, 1.0]])


def test_depth_bounds_brackets_object_shell():
    dn, df = depth_bounds(_c2w(1.52), radius=1.52, half_frac=0.5)
    assert dn < 1.18 and df > 1.74          # contains the lion's observed ref-depth range
    assert dn > 0.0                          # never at/behind the camera
    assert math.isclose(df - dn, 1.52, rel_tol=1e-4)  # width == 2*half_frac*radius


def test_plucker_shape_and_direction_matches_ray_dirs():
    K, c2w = _K(), _c2w()
    pl = plucker_embedding(K, c2w, 24, 16)
    assert pl.shape == (24 * 16, 6)
    dirs = ray_dirs_world(K, c2w, 24, 16)
    assert torch.allclose(pl[:, :3], dirs, atol=1e-5)            # first 3 = unit direction
    o = c2w[:3, 3]
    assert torch.allclose(pl[:, 3:], torch.cross(o.expand_as(dirs), dirs, dim=-1), atol=1e-5)


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
    """The ray through the grid cell nearest image center, at depth=radius,
    must land near the world origin (the object center)."""
    K, c2w, radius = _frame0_camera()
    dirs = ray_dirs_world(K, c2w, 24, 16)
    cam_center = c2w[:3, 3]
    hits = cam_center[None] + radius * dirs            # (H*W,3) at depth=radius
    min_dist = hits.norm(dim=-1).min()
    assert min_dist < 0.15, f"no ray lands near origin at depth=radius; min={min_dist}"


def test_rays_are_unit_and_point_inward():
    K, c2w, _radius = _frame0_camera()
    dirs = ray_dirs_world(K, c2w, 24, 16)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(dirs.shape[0]), atol=1e-5)
    # rays must point from the camera toward the scene (origin side): the dot
    # of each ray with (origin - cam_center) should be positive on average.
    to_origin = (-c2w[:3, 3]) / c2w[:3, 3].norm()
    assert (dirs @ to_origin).mean() > 0.8
