"""Render gate: the clean decoder's Gaussians rasterize without NaN, and a
Gaussian forced to the origin projects to the image center (coordinate-convention
gate, design failure mode C5). Skipped off-GPU."""
from __future__ import annotations

import math
import torch
import pytest

gsplat = pytest.importorskip("gsplat")
if not torch.cuda.is_available():
    pytest.skip("gsplat needs CUDA", allow_module_level=True)

from decoder.clean.network import CleanGSDecoder
from decoder.clean.gaussians import activate
from decoder.clean.geometry import depth_bounds
from decoder.clean.losses import render_expected_depth
from decoder.data import opengl_c2w_to_opencv_w2c
from decoder.render import render_views


def _cam(radius=1.52, device="cuda"):
    K = torch.tensor([[840.0, 0, 256.0], [0, 840.0, 384.0], [0, 0, 1.0]], device=device)
    c2w = torch.eye(4, device=device); c2w[2, 3] = radius
    return K, c2w


def test_decoder_output_renders_without_nan():
    torch.manual_seed(0)
    model = CleanGSDecoder(dim=768, depth=2, heads=8).cuda()
    latent = torch.randn(1, 128, 2, 24, 16, device="cuda")
    K, c2w = _cam()
    p = {k: v[0] for k, v in model(latent, K, c2w, 1.52).items()}
    w2c = opengl_c2w_to_opencv_w2c(c2w)[None]
    rgb, alpha = render_views(p, w2c, K[None], 512, 768, bg=1.0)
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    assert rgb.shape == (1, 768, 512, 3) and (alpha >= 0).all() and (alpha <= 1).all()


def test_origin_gaussian_projects_to_image_center():
    # One opaque Gaussian at the world origin → should render at the principal point.
    K, c2w = _cam()
    origins = c2w[:3, 3][None]                       # camera center
    dirs = torch.nn.functional.normalize(-origins, dim=-1)   # ray toward origin
    raw = torch.zeros(1, 12, device="cuda")
    raw[0, 10] = 20.0                                # opacity → ~1
    raw[0, 3:6] = -1.0                               # modest scale
    dn, df = depth_bounds(c2w, 1.52, 0.5)
    # set dist channel so t == d_cam (=‖origin‖); invert the sigmoid mapping
    d_cam = float(origins.norm())
    frac = (d_cam - dn) / (df - dn)
    raw[0, 11] = math.log(frac / (1 - frac))
    p = activate(raw, origins, dirs, dn, df, 1.52)
    assert torch.allclose(p["mean"], torch.zeros(1, 3, device="cuda"), atol=1e-3)
    w2c = opengl_c2w_to_opencv_w2c(c2w)[None]
    rgb, alpha = render_views(p, w2c, K[None], 512, 768, bg=0.0)
    # most-opaque pixel near the principal point (col 256, row 384)
    yx = (alpha[0, :, :, 0]).flatten().argmax()
    row, col = int(yx // 512), int(yx % 512)
    assert abs(col - 256) < 30 and abs(row - 384) < 30


def test_expected_depth_render_mode_is_gsplat_compatible():
    K, c2w = _cam()
    origins = c2w[:3, 3][None]
    dirs = torch.nn.functional.normalize(-origins, dim=-1)
    raw = torch.zeros(1, 12, device="cuda")
    raw[0, 10] = 20.0
    dn, df = depth_bounds(c2w, 1.52, 0.5)
    p = activate(raw, origins, dirs, dn, df, 1.52)
    w2c = opengl_c2w_to_opencv_w2c(c2w)[None]
    d = render_expected_depth(p, w2c, K[None], 512, 768, mode="Ed")
    assert d.shape == (1, 768, 512, 1)
    assert torch.isfinite(d).all()
