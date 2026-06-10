"""Phase2Dataset: one object/sample, K-view subsample (always incl. ref view 0),
latent + per-object cameras present. CPU; uses the real animals_v4 manifest."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from decoder.clean.phase2_data import Phase2Dataset

V4 = "/home/rrzhang/projects/data/Animals v4 Final"
MANIFEST = V4 + "/animals_v4_approved_encoded.json"

pytestmark = pytest.mark.skipif(not Path(V4).is_dir(), reason="animals_v4 dataset not present")


def test_split_sizes():
    assert len(Phase2Dataset(V4, "train", MANIFEST)) == 76
    assert len(Phase2Dataset(V4, "eval", MANIFEST)) == 1
    assert len(Phase2Dataset(V4, "test", MANIFEST)) == 4


def test_sample_shapes_and_k_views():
    s = Phase2Dataset(V4, "train", MANIFEST, k_views=8)[0]
    assert s["latent"].shape == (128, 2, 24, 16)
    assert s["ref_K"].shape == (3, 3) and s["ref_c2w"].shape == (4, 4)
    assert s["frames"].shape[0] == 8 and s["masks"].shape[0] == 8
    assert s["w2c"].shape == (8, 4, 4) and s["K"].shape == (8, 3, 3)
    assert s["frames"].shape[1:] == (s["height"], s["width"], 3)
    assert s["masks"].shape[1:] == (s["height"], s["width"], 1)
    assert isinstance(s["radius"], float) and s["radius"] > 0


def test_subsample_includes_ref_view_and_is_distinct():
    ds = Phase2Dataset(V4, "train", MANIFEST, k_views=8)
    for _ in range(5):
        vi = ds[0]["view_indices"].tolist()
        assert vi[0] == 0                       # reference view always first
        assert len(set(vi)) == 8                # distinct views


def test_k_views_clamped_to_available():
    s = Phase2Dataset(V4, "train", MANIFEST, k_views=1000)[0]
    assert s["frames"].shape[0] == s["w2c"].shape[0]   # clamped to the object's view count


def test_held_out_uids_disjoint_from_train():
    tr = {e["uid"] for e in Phase2Dataset(V4, "train", MANIFEST).entries}
    ho = [e["uid"] for e in Phase2Dataset(V4, "eval", MANIFEST).entries
          + Phase2Dataset(V4, "test", MANIFEST).entries]
    assert tr and ho and all(u not in tr for u in ho)


def test_sample_has_ref_depth():
    s = Phase2Dataset(V4, "train", MANIFEST, k_views=8)[0]
    H, W = s["height"], s["width"]
    assert s["ref_depth"].shape == (H, W)            # ray-distance, (H,W)
    assert s["ref_depth_valid"].shape == (H, W)
    assert s["ref_depth_valid"].dtype == torch.bool
    # valid pixels must be a subset of the reference-view foreground
    fg0 = s["masks"][0, ..., 0] > 0.5
    assert (s["ref_depth_valid"] & ~fg0).sum() == 0   # no valid depth outside the silhouette
    assert s["ref_depth_valid"].sum() > 0             # some valid foreground depth exists


def test_fixed_conditioning_includes_gt_rgb_targets():
    s = Phase2Dataset(
        V4,
        "train",
        MANIFEST,
        k_views=2,
        cond_subdir="ltx_decoded",
        cond_view_spec="available",
    )[0]
    assert "cond_frames" in s and "cond_target_frames" in s
    assert s["cond_target_frames"].shape == s["cond_frames"].shape
    assert s["cond_target_frames"].shape[-1] == 3
