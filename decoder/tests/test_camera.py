"""Camera conversion correctness — the highest-risk math in the pipeline.

These tests do NOT require the dataset; they synthesize an orbit from the
documented convention (elevation 15°, radius 0.95, fov_y 49.1°, 512x768) and
verify that the OpenGL→OpenCV→projection round-trip lands the world origin at
the image principal point in every view.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from decoder.data import opengl_c2w_to_opencv_w2c


def _orbit_c2w_opengl(azimuth_deg: float, elevation_deg: float, radius: float) -> torch.Tensor:
    """Build an OpenGL c2w looking at the origin, +Y up — mirrors the renderer."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = np.array([
        radius * math.cos(el) * math.cos(az),
        radius * math.sin(el),
        radius * math.cos(el) * math.sin(az),
    ])
    fwd = -eye / np.linalg.norm(eye)            # camera looks toward origin
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    # OpenGL camera basis: +X right, +Y up, +Z = -forward (looks down -Z)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -fwd
    c2w[:3, 3] = eye
    return torch.tensor(c2w, dtype=torch.float32)


def _intrinsics(width: int, height: int, fov_y_deg: float) -> torch.Tensor:
    fy = (height / 2) / math.tan(math.radians(fov_y_deg) / 2)
    K = torch.eye(3)
    K[0, 0] = fy   # square pixels: fx = fy
    K[1, 1] = fy
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    return K


def test_origin_projects_to_principal_point():
    """World origin (the orbit target) must project to (cx,cy) in every view."""
    width, height, fov_y, radius, elev = 512, 768, 49.1, 0.95, 15.0
    K = _intrinsics(width, height, fov_y)
    for az in np.linspace(0, 360, 25, endpoint=False):
        c2w_gl = _orbit_c2w_opengl(az, elev, radius)
        w2c = opengl_c2w_to_opencv_w2c(c2w_gl)
        p_cam = (w2c @ torch.tensor([0.0, 0.0, 0.0, 1.0]))[:3]
        assert p_cam[2] > 0, f"origin behind camera at az={az}: z={p_cam[2]}"
        uv = (K @ (p_cam / p_cam[2]))[:2]
        assert torch.allclose(uv, torch.tensor([width / 2, height / 2]), atol=1.0), uv


def test_point_above_origin_projects_higher():
    """A point at +Y (world up) projects ABOVE center (smaller pixel-y) in OpenCV."""
    width, height, fov_y, radius, elev = 512, 768, 49.1, 0.95, 15.0
    K = _intrinsics(width, height, fov_y)
    c2w_gl = _orbit_c2w_opengl(0.0, elev, radius)
    w2c = opengl_c2w_to_opencv_w2c(c2w_gl)
    p = (w2c @ torch.tensor([0.0, 0.3, 0.0, 1.0]))[:3]
    uv = (K @ (p / p[2]))[:2]
    assert uv[1] < height / 2, f"point above origin should map to smaller y, got {uv[1]}"


def test_w2c_is_rigid():
    """Converted w2c must be a rigid transform (orthonormal rotation, det +1)."""
    c2w_gl = _orbit_c2w_opengl(123.0, 15.0, 0.95)
    w2c = opengl_c2w_to_opencv_w2c(c2w_gl)
    R = w2c[:3, :3]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0), atol=1e-5)
