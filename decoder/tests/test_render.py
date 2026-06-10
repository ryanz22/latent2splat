"""gsplat render sanity. Skipped if gsplat/CUDA unavailable (i.e. off-GPU)."""
from __future__ import annotations

import math

import pytest
import torch

gsplat = pytest.importorskip("gsplat")
if not torch.cuda.is_available():
    pytest.skip("gsplat rasterization needs CUDA", allow_module_level=True)

from decoder.render import render_views, photometric_loss


def _single_camera(width=512, height=768, fov_y=49.1, radius=0.95):
    """Return (w2c (1,4,4), K (1,3,3), width, height) for one OpenCV camera
    sitting at +radius·Z in world, looking toward the origin (+Z forward)."""
    fy = (height / 2) / math.tan(math.radians(fov_y) / 2)
    K = torch.tensor([[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1.0]], device="cuda")[None]
    w2c = torch.eye(4, device="cuda")
    w2c[2, 3] = radius  # world origin sits at depth +radius in camera frame
    return w2c[None], K, width, height


def test_centered_gaussian_renders_bright_blob_at_center():
    w2c, K, w, h = _single_camera()
    params = {
        "mean": torch.zeros(1, 3, device="cuda"),
        "quat": torch.tensor([[1.0, 0, 0, 0]], device="cuda"),
        "scale": torch.full((1, 3), 0.1, device="cuda"),
        "opacity": torch.ones(1, 1, device="cuda"),
        "rgb": torch.ones(1, 3, device="cuda"),
    }
    img, alpha = render_views(params, w2c, K, w, h)  # (1,H,W,3), (1,H,W,1)
    assert img.shape == (1, h, w, 3)
    assert alpha.shape == (1, h, w, 1)
    cy, cx = h // 2, w // 2
    center = img[0, cy - 5:cy + 5, cx - 5:cx + 5].mean()
    corner = img[0, :10, :10].mean()
    assert center > corner, f"center {center} not brighter than corner {corner}"
    assert center > 0.6, f"white gaussian should brighten center, got {center}"
    assert alpha[0, cy, cx] > 0.5, f"alpha at center should be high, got {alpha[0, cy, cx]}"


def test_photometric_loss_zero_on_identical():
    img = torch.rand(2, 32, 32, 3, device="cuda")
    params = {"scale": torch.full((4, 3), 0.05, device="cuda"),
              "opacity": torch.full((4, 1), 0.5, device="cuda")}
    loss, comp = photometric_loss(img, img.clone(), params, ssim_weight=0.2, scale_reg_weight=0.0)
    assert comp["l1"] < 1e-6


def test_mask_loss_penalizes_vanished_alpha():
    """The core of the v2 fix: an all-zero alpha (Gaussians vanished) must
    incur HIGH mask loss where the target has foreground, so 'all invisible'
    is no longer a low-loss attractor. animals_v1 uses a BLACK background."""
    from decoder.render import foreground_mask
    # target on black bg: left half background (black), right half object (bright)
    target = torch.zeros(1, 16, 16, 3, device="cuda")
    target[:, :, 8:, :] = 0.9
    mask = foreground_mask(target, bg=0.0)
    assert mask[:, :, 8:, :].mean() > 0.99 and mask[:, :, :8, :].mean() < 0.01
    params = {"scale": torch.full((4, 3), 0.05, device="cuda"),
              "opacity": torch.zeros(4, 1, device="cuda")}
    vanished_alpha = torch.zeros(1, 16, 16, 1, device="cuda")
    black_render = torch.zeros(1, 16, 16, 3, device="cuda")
    _, comp = photometric_loss(black_render, target, params, alpha=vanished_alpha,
                               bg=0.0, mask_weight=1.0, scale_reg_weight=0.0)
    assert comp["mask"] > 1.0, f"vanished alpha should incur high mask loss, got {comp['mask']}"
