from __future__ import annotations

import json

import numpy as np
from PIL import Image

from decoder.clean.inspect_dataset import diagnose_depth_sidecars, inspect_condition_coverage


def _write_tiny_object(root, uid: str = "abc123"):
    obj = root / uid
    (obj / "depth").mkdir(parents=True)
    (obj / "masks").mkdir(parents=True)
    (obj / "da3_ltx").mkdir(parents=True)

    cams = {
        "width": 4,
        "height": 5,
        "radius": 2.0,
        "num_orbit_views": 1,
        "frames": [
            {
                "intrinsics": [[4.0, 0.0, 2.0], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]],
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
    Image.fromarray(np.full((5, 4), 255, dtype=np.uint8)).save(obj / "masks" / "mask_000.png")
    gt = np.full((5, 4), 2.0, dtype=np.float32)
    pred = gt + 0.25
    np.save(obj / "depth" / "depth_000.npy", gt)
    np.save(obj / "da3_ltx" / "depth_000.npy", pred)
    return obj


def test_diagnose_depth_sidecars_reports_raw_and_fraction_errors(tmp_path):
    _write_tiny_object(tmp_path)
    manifest = {"train": ["abc123"]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    out = diagnose_depth_sidecars(
        tmp_path,
        manifest_path,
        ["da3_ltx"],
        splits=["train"],
        limit_per_split=1,
        view_spec="0",
        max_px_per_view=0,
    )

    stats = out["splits"]["train"]["sidecars"]["da3_ltx"]
    assert stats["objects_with_valid_depth"] == 1
    assert stats["views_with_valid_depth"] == 1
    assert stats["sampled_pixels"] == 20
    assert abs(stats["raw_z_mae"] - 0.25) < 1e-6
    assert stats["frac_mae"] > 0
    assert stats["raw_z_affine"]["mae"] is None


def test_inspect_condition_coverage_skips_latents(tmp_path):
    _write_tiny_object(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"train": ["abc123"]}))
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(
        tmp_path / "abc123" / "frame_000.png"
    )
    out = inspect_condition_coverage(
        tmp_path,
        tmp_path / "manifest.json",
        cond_subdir=None,
        cond_depth_subdir="da3_ltx",
        splits=["train"],
        min_views=1,
    )
    stats = out["splits"]["train"]
    assert stats["objects"] == 1
    assert stats["objects_with_min_views"] == 1
