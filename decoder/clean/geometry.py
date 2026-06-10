"""Camera geometry for the clean-slate decoder: per-pixel world rays,
radius-relative depth bounds, and Plücker ray embeddings. `ray_dirs_world` is
the repo's single tested ray builder so the OpenGL→world convention stays
consistent (failure mode C5)."""
from __future__ import annotations

import torch


def ray_dirs_world(K: torch.Tensor, c2w_opengl: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Per-pixel world-space ray directions for an OpenGL camera.

    K: (3,3) intrinsics for the FULL-RES image. h,w: the latent grid size
    (we sample pixel centers of the h×w grid mapped onto the full image).
    Returns (h*w, 3) unit ray directions in world space, and the implied
    pixel centers are the grid-cell centers of the full image.

    OpenGL camera: looks down -Z, +Y up. Camera-space ray for pixel (u,v):
        d_cam = ((u-cx)/fx, -(v-cy)/fy, -1)
    rotate by the c2w rotation (no translation) to get world direction.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    full_w, full_h = 2 * cx, 2 * cy  # principal point is image center here
    # pixel centers of an h×w grid spanning the full image
    us = (torch.arange(w, dtype=K.dtype, device=K.device) + 0.5) * (full_w / w)
    vs = (torch.arange(h, dtype=K.dtype, device=K.device) + 0.5) * (full_h / h)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")  # (h,w)
    d_cam = torch.stack([(uu - cx) / fx, -(vv - cy) / fy, -torch.ones_like(uu)], dim=-1)
    d_cam = d_cam.reshape(-1, 3)                       # (h*w, 3)
    R = c2w_opengl[:3, :3]
    d_world = d_cam @ R.T                              # rotate cam->world
    return d_world / d_world.norm(dim=-1, keepdim=True)


def depth_bounds(c2w_opengl: torch.Tensor, radius: float, half_frac: float = 0.5):
    """Sigmoid-depth shell around the object, in world units.

    NOTE: in our data `radius` is the CAMERA-ORBIT distance (‖cam center‖), not
    the object bounding radius. We bracket [d_cam - half_frac*radius,
    d_cam + half_frac*radius]; half_frac=0.5 covers the observed object depths.
    Returns python floats for normal dataset cameras. If `c2w_opengl` requires
    gradients, returns scalar tensors so predicted-camera training keeps the
    camera-distance gradient instead of detaching through `float(...)`.
    """
    d_cam = c2w_opengl[:3, 3].norm()
    radius_t = torch.as_tensor(radius, device=d_cam.device, dtype=d_cam.dtype)
    half = float(half_frac) * radius_t
    d_near = torch.maximum(d_cam - half, 0.05 * d_cam)
    d_far = d_cam + half
    if c2w_opengl.requires_grad:
        return d_near, d_far
    return float(d_near), float(d_far)


def plucker_embedding(K: torch.Tensor, c2w_opengl: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(h*w, 6) Plücker coordinates [direction(3), moment = origin × direction(3)]
    for the h×w pixel-center grid of one OpenGL camera. Direction matches
    `ray_dirs_world` exactly."""
    dirs = ray_dirs_world(K, c2w_opengl, h, w)          # (h*w, 3) unit, world
    origin = c2w_opengl[:3, 3]                          # (3,)
    moment = torch.cross(origin.expand_as(dirs), dirs, dim=-1)
    return torch.cat([dirs, moment], dim=-1)            # (h*w, 6)
