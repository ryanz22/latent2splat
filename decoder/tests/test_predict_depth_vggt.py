from pathlib import Path

import numpy as np
import torch
from PIL import Image

from decoder.clean.predict_depth_vggt import (
    _as_view_maps,
    _fit_global_affine_to_hull,
    _preprocess_pad,
    _restore_padded_map,
)


def _write_image(path: Path, size: tuple[int, int]) -> None:
    w, h = size
    yy, xx = np.mgrid[:h, :w]
    img = np.stack(
        [
            (xx % 256).astype(np.uint8),
            (yy % 256).astype(np.uint8),
            ((xx + yy) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(img, mode="RGB").save(path)


def test_preprocess_pad_preserves_aspect_for_tall_frame(tmp_path):
    path = tmp_path / "frame.png"
    _write_image(path, (512, 768))

    tensor, info = _preprocess_pad(path, target_size=518)

    assert tensor.shape == (3, 518, 518)
    assert info["height"] == 768
    assert info["width"] == 512
    assert info["new_height"] == 518
    assert 336 <= info["new_width"] <= 350
    assert info["pad_left"] > 0
    assert info["pad_top"] == 0


def test_restore_padded_map_returns_original_frame_size(tmp_path):
    path = tmp_path / "frame.png"
    _write_image(path, (512, 768))
    _, info = _preprocess_pad(path, target_size=518)

    square = torch.zeros(518, 518)
    top = info["pad_top"]
    left = info["pad_left"]
    new_h = info["new_height"]
    new_w = info["new_width"]
    square[top:top + new_h, left:left + new_w] = 3.0

    restored = _restore_padded_map(square, info)

    assert restored.shape == (768, 512)
    assert torch.allclose(restored.mean(), torch.tensor(3.0), atol=1e-5)


def test_as_view_maps_accepts_common_vggt_shapes():
    assert _as_view_maps(torch.zeros(1, 2, 518, 518, 1)).shape == (2, 518, 518)
    assert _as_view_maps(torch.zeros(1, 2, 518, 518)).shape == (2, 518, 518)
    assert _as_view_maps(torch.zeros(2, 518, 518, 1)).shape == (2, 518, 518)
    assert _as_view_maps(torch.zeros(2, 518, 518)).shape == (2, 518, 518)


def test_fit_global_affine_to_hull_uses_all_views():
    h, w = 8, 8
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    rel0 = (xx.float() + yy.float()) / 10.0
    rel1 = rel0 + 2.0
    scale = 0.4
    bias = 1.2
    hull_t = torch.stack([scale * rel0 + bias, scale * rel1 + bias])

    K = torch.eye(3).repeat(2, 1, 1)
    K[:, 0, 0] = 1e6
    K[:, 1, 1] = 1e6
    K[:, 0, 2] = (w - 1) / 2
    K[:, 1, 2] = (h - 1) / 2
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[:, 2, 3] = 2.0
    masks = torch.ones(2, h, w, 1)

    fit = _fit_global_affine_to_hull([rel0, rel1], hull_t, masks, K, c2w, radius=2.0)

    assert fit is not None
    assert torch.allclose(fit[0], torch.tensor(scale), atol=1e-4)
    assert torch.allclose(fit[1], torch.tensor(bias), atol=1e-4)
