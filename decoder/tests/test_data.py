"""Dataset + real-camera tests. Skipped if the dataset isn't present.

The key invariant: every object is centered at the world origin (orbit target
= [0,0,0]), so the origin must project very near the image principal point in
all 25 real cameras. This validates the OpenGL→OpenCV conversion against the
actual data, independent of how the orbit was parameterized.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from decoder.clean.phase2_data import (
    available_frame_indices,
    depth_path_at,
    frame_path_at,
    load_conf_view_at,
    load_depth_view_at,
)
from decoder.data import (
    ObjaverseLatentDataset,
    entry_relpath,
    latent_path_for_entry,
    load_cameras,
    object_dir_for_entry,
)

DATASET_ROOT = Path(
    os.environ.get("DATASET_ROOT", Path(__file__).resolve().parents[3] / "data" / "objaverse120_v2")
)
_have_data = (DATASET_ROOT / "manifest.json").exists()
skip_no_data = pytest.mark.skipif(not _have_data, reason=f"dataset not at {DATASET_ROOT}")


@skip_no_data
def test_dataset_sample_shapes():
    """Structural checks that hold for both v1 (25-frame/T=4) and v2
    (9-frame-input/T=2, 49-view supervision) datasets — view count and T are
    read from the data, not hardcoded."""
    ds = ObjaverseLatentDataset(DATASET_ROOT, split="train")
    assert len(ds) > 0
    s = ds[0]
    C, T, H, W = s["latent"].shape
    assert C == 128 and H == 24 and W == 16          # spatial/channel fixed by VAE
    # T = (input_frames-1)//8 + 1: animals_v1=7 (49 frames), 9-frame=2, v1=4 (25)
    assert T in (2, 4, 7)
    V = s["frames"].shape[0]                          # supervision view count
    assert s["frames"].shape == (V, 768, 512, 3)
    assert s["w2c"].shape == (V, 4, 4)
    assert s["c2w_opengl"].shape == (V, 4, 4)
    assert s["K"].shape == (V, 3, 3)
    assert s["frames"].min() >= 0.0 and s["frames"].max() <= 1.0


@skip_no_data
def test_real_cameras_center_origin():
    """Origin projects near image center for all 25 real cameras of an object."""
    uid_dir = next(d for d in DATASET_ROOT.iterdir() if (d / "cameras.json").exists())
    cams = load_cameras(uid_dir / "cameras.json")
    w2c, K = cams["w2c"], cams["K"]
    w, h = cams["width"], cams["height"]
    origin = torch.tensor([0.0, 0.0, 0.0, 1.0])
    for i in range(w2c.shape[0]):
        p = (w2c[i] @ origin)[:3]
        assert p[2] > 0, f"view {i}: origin behind camera (z={p[2]})"
        uv = (K[i] @ (p / p[2]))[:2]
        # object is centered; origin should be within a few px of (cx,cy)
        assert abs(uv[0] - w / 2) < 5, f"view {i}: u={uv[0]} far from cx={w/2}"
        assert abs(uv[1] - h / 2) < 5, f"view {i}: v={uv[1]} far from cy={h/2}"


def test_combined_manifest_entry_paths(tmp_path, monkeypatch):
    entry = {"uid": "abc123", "tier": "objaverse_v8"}
    assert object_dir_for_entry(tmp_path / "renders", entry) == (
        tmp_path / "renders" / "objaverse_v8" / "abc123"
    )
    assert entry_relpath(entry) == Path("objaverse_v8") / "abc123"

    latent_dir = tmp_path / "latents_v8" / "objaverse_v8" / "abc123"
    latent_dir.mkdir(parents=True)
    latent = latent_dir / "latent.npy"
    latent.write_bytes(b"stub")
    monkeypatch.setenv("L2S_LATENTS_V8_ROOT", str(tmp_path / "latents_v8"))
    assert latent_path_for_entry(tmp_path / "renders", entry) == latent


def test_legacy_manifest_entry_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("L2S_LATENTS_V8_ROOT", raising=False)
    assert object_dir_for_entry(tmp_path / "renders", "legacy") == (
        tmp_path / "renders" / "legacy"
    )
    assert latent_path_for_entry(tmp_path / "renders", "legacy") == (
        tmp_path / "renders" / "legacy" / "latent.npy"
    )
    assert entry_relpath("legacy") == Path("legacy")


def test_depth_override_prefers_tier_uid_path(tmp_path, monkeypatch):
    obj_dir = tmp_path / "renders" / "objaverse_v8" / "abc123"
    out_dir = tmp_path / "override" / "objaverse_v8" / "abc123" / "refined"
    out_dir.mkdir(parents=True)
    arr = np.full((2, 3), 7.0, dtype=np.float32)
    np.save(out_dir / "depth_004.npy", arr)

    monkeypatch.setenv("L2S_DEPTH_OVERRIDE_ROOT", str(tmp_path / "override"))
    assert depth_path_at(obj_dir, 4, subdir="@refined") == out_dir / "depth_004.npy"
    got = load_depth_view_at(obj_dir, 4, subdir="@refined")
    assert torch.equal(got, torch.from_numpy(arr))


def test_dataset_eval_view_cap_loads_selected_views(tmp_path):
    uid = "tiny"
    obj = tmp_path / uid
    (obj / "masks").mkdir(parents=True)
    (obj / "depth").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"eval": [uid]}))
    np.save(obj / "latent.npy", np.zeros((128, 2, 4, 4), dtype=np.float32))

    frames = []
    for i in range(5):
        Image.fromarray(np.full((2, 3, 3), i, dtype=np.uint8)).save(obj / f"frame_{i:03d}.png")
        Image.fromarray(np.full((2, 3), 255, dtype=np.uint8)).save(obj / "masks" / f"mask_{i:03d}.png")
        np.save(obj / "depth" / f"depth_{i:03d}.npy", np.full((2, 3), i, dtype=np.float32))
        frames.append({
            "c2w_opengl": np.eye(4, dtype=np.float32).tolist(),
            "intrinsics": np.eye(3, dtype=np.float32).tolist(),
        })
    (obj / "cameras.json").write_text(json.dumps({
        "width": 3,
        "height": 2,
        "radius": 1.0,
        "num_orbit_views": 4,
        "num_canonical_views": 1,
        "frames": frames,
    }))

    ds = ObjaverseLatentDataset(tmp_path, split="eval", load_depths=True, max_views=3)
    sample = ds[0]
    assert sample["view_indices"].tolist() == [0, 2, 3]
    assert sample["frames"].shape == (3, 2, 3, 3)
    assert sample["masks"].shape == (3, 2, 3, 1)
    assert sample["depth"].shape == (3, 2, 3)
    assert sample["w2c"].shape == (3, 4, 4)
    assert sample["all_w2c"].shape == (5, 4, 4)
    assert sample["all_c2w_opengl"].shape == (5, 4, 4)
    assert sample["all_K"].shape == (5, 3, 3)
    assert torch.allclose(
        sample["frames"][:, 0, 0, 0],
        torch.tensor([0.0, 2 / 255.0, 3 / 255.0], dtype=torch.float32),
    )


def test_condition_frame_override_prefers_tier_uid_path(tmp_path, monkeypatch):
    obj_dir = tmp_path / "renders" / "objaverse_v8" / "abc123"
    out_dir = tmp_path / "condition" / "objaverse_v8" / "abc123" / "ltx_decoded"
    out_dir.mkdir(parents=True)
    frame = out_dir / "frame_006.png"
    frame.write_bytes(b"stub")

    monkeypatch.setenv("L2S_COND_OVERRIDE_ROOT", str(tmp_path / "condition"))
    assert frame_path_at(obj_dir, 6, subdir="@ltx_decoded") == frame
    assert available_frame_indices(obj_dir, subdir="@ltx_decoded") == [6]


def test_condition_conf_override_prefers_tier_uid_path(tmp_path, monkeypatch):
    obj_dir = tmp_path / "renders" / "objaverse_v8" / "abc123"
    out_dir = tmp_path / "depth" / "objaverse_v8" / "abc123" / "da3_ltx"
    out_dir.mkdir(parents=True)
    conf = np.full((2, 3), 0.7, dtype=np.float32)
    np.save(out_dir / "conf_006.npy", conf)

    monkeypatch.setenv("L2S_DEPTH_OVERRIDE_ROOT", str(tmp_path / "depth"))
    loaded = load_conf_view_at(obj_dir, 6, subdir="@da3_ltx")
    assert torch.allclose(loaded, torch.from_numpy(conf))
