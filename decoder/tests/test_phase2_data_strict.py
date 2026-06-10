from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from decoder.clean.phase2_data import Phase2Dataset


def _write_tiny_phase2_dataset(root):
    uid = "tiny"
    obj = root / uid
    (obj / "depth").mkdir(parents=True)
    (obj / "masks").mkdir(parents=True)
    (obj / "ltx_decoded").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"train": [uid]}))
    np.save(obj / "latent.npy", np.zeros((128, 2, 24, 16), dtype=np.float32))
    frame = np.full((4, 4, 3), 255, dtype=np.uint8)
    mask = np.full((4, 4), 255, dtype=np.uint8)
    Image.fromarray(frame).save(obj / "frame_000.png")
    Image.fromarray(frame).save(obj / "ltx_decoded" / "frame_000.png")
    Image.fromarray(mask).save(obj / "masks" / "mask_000.png")
    np.save(obj / "depth" / "depth_000.npy", np.ones((4, 4), dtype=np.float32))
    cams = {
        "width": 4,
        "height": 4,
        "radius": 2.0,
        "num_orbit_views": 1,
        "num_canonical_views": 0,
        "frames": [
            {
                "intrinsics": [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
                "c2w_opengl": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    (obj / "cameras.json").write_text(json.dumps(cams))


def test_missing_requested_condition_depth_raises(tmp_path):
    _write_tiny_phase2_dataset(tmp_path)
    ds = Phase2Dataset(
        tmp_path,
        "train",
        tmp_path / "manifest.json",
        k_views=1,
        cond_subdir="ltx_decoded",
        cond_depth_subdir="missing_depth",
        cond_view_spec="0",
    )
    with pytest.raises(FileNotFoundError, match="missing conditioning depth"):
        _ = ds[0]


def test_missing_requested_condition_depth_can_fallback_for_legacy_runs(tmp_path):
    _write_tiny_phase2_dataset(tmp_path)
    ds = Phase2Dataset(
        tmp_path,
        "train",
        tmp_path / "manifest.json",
        k_views=1,
        cond_subdir="ltx_decoded",
        cond_depth_subdir="missing_depth",
        cond_view_spec="0",
        strict_cond_depth=False,
    )
    sample = ds[0]
    assert sample["cond_depth_valid"].sum().item() == 0
    assert sample["cond_depths"].shape[-2:] == (4, 4)
