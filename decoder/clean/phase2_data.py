"""Phase-2 multi-object data feed: one object per sample, with a random K-view
subsample loaded per step (NOT all 49 views — that I/O would dominate training).
Reuses load_cameras; adds index-specific frame/mask loaders."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image

from decoder.data import (
    entry_uid,
    latent_path_for_entry,
    load_cameras,
    load_depth_view,
    object_dir_for_entry,
)


def _sidecar_base(obj_dir, subdir: str | None, env_name: str) -> Path:
    obj = Path(obj_dir)
    if not subdir:
        return obj
    if subdir.startswith("@"):
        actual = subdir.lstrip("@")
        override = os.environ.get(env_name, "")
        if override:
            uid = obj.name
            parent = obj.parent.name
            candidates = []
            if parent.startswith("objaverse_"):
                candidates.append(Path(override) / parent / uid / actual)
            candidates.append(Path(override) / uid / actual)
            for p in candidates:
                if p.exists():
                    return p
            if parent.startswith("objaverse_"):
                return Path(override) / parent / uid / actual
            return Path(override) / uid / actual
        return obj / actual
    return obj / subdir


def load_depth_view_at(obj_dir, index: int, subdir: str | None = None) -> torch.Tensor:
    """Load depth_NNN.npy from `obj_dir/subdir` or the dataset's default depth layout.

    When ``L2S_DEPTH_OVERRIDE_ROOT`` is set in the environment AND the subdir
    starts with '@', the file is read from ``$L2S_DEPTH_OVERRIDE_ROOT/<uid>/<subdir>``
    (with the '@' stripped). This lets us bake refined depths to a separate
    writable volume without touching the dataset volume.
    """
    if subdir:
        p = depth_path_at(obj_dir, index, subdir=subdir)
        return torch.from_numpy(np.load(p).astype(np.float32))
    return load_depth_view(Path(obj_dir), index)


def load_conf_view_at(obj_dir, index: int, subdir: str | None = None) -> torch.Tensor:
    """Load conf_NNN.npy from `obj_dir/subdir`."""
    if not subdir:
        raise FileNotFoundError("confidence maps require a subdir")
    p = _sidecar_base(obj_dir, subdir, "L2S_DEPTH_OVERRIDE_ROOT") / f"conf_{index:03d}.npy"
    return torch.from_numpy(np.load(p).astype(np.float32))


def load_views_at(obj_dir, indices, subdir=None) -> torch.Tensor:
    """frame_{i}.png at the given view indices -> (K,H,W,3) float32 in [0,1]."""
    base = _sidecar_base(obj_dir, subdir, "L2S_COND_OVERRIDE_ROOT")
    imgs = [np.asarray(Image.open(base / f"frame_{i:03d}.png").convert("RGB"),
                       dtype=np.float32) / 255.0 for i in indices]
    return torch.from_numpy(np.stack(imgs))


def frame_path_at(obj_dir, index: int, subdir: str | None = None) -> Path:
    """Resolve a frame path, including @ sidecar conditioning roots."""
    return _sidecar_base(obj_dir, subdir, "L2S_COND_OVERRIDE_ROOT") / f"frame_{index:03d}.png"


def depth_path_at(obj_dir, index: int, subdir: str | None = None) -> Path:
    """Resolve a depth path, including @ sidecar depth roots."""
    if subdir:
        return _sidecar_base(obj_dir, subdir, "L2S_DEPTH_OVERRIDE_ROOT") / f"depth_{index:03d}.npy"
    obj = Path(obj_dir)
    p = obj / "depth" / f"depth_{index:03d}.npy"
    if p.exists():
        return p
    return obj / f"depth_{index:03d}.npy"


def load_masks_at(obj_dir, indices) -> torch.Tensor:
    """masks/mask_{i}.png at the given view indices -> (K,H,W,1) float32 in {0,1}."""
    ms = [(np.asarray(Image.open(Path(obj_dir) / "masks" / f"mask_{i:03d}.png").convert("L")) > 0)
          .astype(np.float32) for i in indices]
    return torch.from_numpy(np.stack(ms))[..., None]


def available_frame_indices(obj_dir, subdir=None) -> list[int]:
    """Return view ids available as frame_NNN.png under `obj_dir/subdir`."""
    base = _sidecar_base(obj_dir, subdir, "L2S_COND_OVERRIDE_ROOT")
    out = []
    for p in sorted(base.glob("frame_*.png")):
        m = re.fullmatch(r"frame_(\d+)\.png", p.name)
        if m:
            out.append(int(m.group(1)))
    return out


def spread_view_indices(n_views: int, n_select: int, n_orbit_views: int | None = None) -> list[int]:
    """Evenly spaced view ids, avoiding canonical appended views when known."""
    limit = min(n_views, n_orbit_views or n_views)
    n_select = min(max(n_select, 1), limit)
    if n_select == 1:
        return [0]
    idx = torch.arange(n_select, dtype=torch.float32) * (float(limit) / n_select)
    return idx.round().long().clamp_max(limit - 1).unique().tolist()


def resolve_view_spec(spec: str | None, n_views: int, obj_dir=None, subdir=None,
                      n_orbit_views: int | None = None,
                      default_n: int | None = None) -> list[int] | None:
    """Resolve a conditioning view spec.

    Forms: "available", "spread:N", or an explicit comma list like "0,6,12".
    Empty specs return None unless `default_n` is supplied.
    """
    if spec is None or spec == "":
        if default_n is None:
            return None
        return spread_view_indices(n_views, default_n, n_orbit_views)
    if spec == "available":
        if obj_dir is None:
            raise ValueError("'available' view spec requires obj_dir")
        idxs = available_frame_indices(obj_dir, subdir=subdir)
        if not idxs:
            raise FileNotFoundError(f"no frame_*.png files found in {Path(obj_dir) / subdir}")
        return [i for i in idxs if 0 <= i < n_views]
    if spec.startswith("spread:"):
        return spread_view_indices(n_views, int(spec.split(":", 1)[1]), n_orbit_views)
    return [i for i in (int(x) for x in spec.split(",") if x.strip()) if 0 <= i < n_views]


class Phase2Dataset(torch.utils.data.Dataset):
    """One sample per object: latent + reference camera (view 0) + a random K-view
    subsample (always incl. view 0) of {w2c, K, frames, masks}. View images are loaded
    ONLY for the subsample. For the batch=1 multi-object Phase-2 loop."""

    def __init__(self, dataset_root, split="train", manifest_path=None,
                 k_views=8, bg_variant=None, cond_subdir=None,
                 cond_depth_subdir: str | None = None,
                 cond_visibility_depth_subdir: str | None = None,
                 cond_conf_subdir: str | None = None,
                 cond_view_spec: str | None = None,
                 cond_default_views: int | None = None,
                 strict_cond_depth: bool = True):
        self.root = Path(dataset_root)
        self.k_views = k_views
        self.bg_variant = bg_variant
        self.cond_subdir = cond_subdir
        self.cond_depth_subdir = cond_depth_subdir
        self.cond_visibility_depth_subdir = cond_visibility_depth_subdir
        self.cond_conf_subdir = cond_conf_subdir
        self.cond_view_spec = cond_view_spec
        self.cond_default_views = cond_default_views
        self.strict_cond_depth = strict_cond_depth
        mpath = Path(manifest_path) if manifest_path else self.root / "manifest.json"
        self.entries = json.loads(Path(mpath).read_text())[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        uid = entry_uid(entry)
        obj_dir = object_dir_for_entry(self.root, entry)
        cams = load_cameras(obj_dir / "cameras.json")
        v = cams["w2c"].shape[0]
        k = min(self.k_views, v)
        # reference view 0 always first; the rest random without replacement (torch RNG
        # is per-worker seeded by the DataLoader, so views vary across workers/epochs)
        others = torch.randperm(v - 1)[: k - 1] + 1
        sel = torch.cat([torch.zeros(1, dtype=torch.long), others])
        idxs = sel.tolist()
        # reference-view GT depth. gsplat render_mode="ED" returns expected
        # camera-z depth, so this target stays in raw Blender Z-depth units.
        H, W = cams["height"], cams["width"]
        masks = load_masks_at(obj_dir, idxs)
        depths = []
        depth_valid = []
        for view_i, mask_i in zip(idxs, masks):
            try:
                z = load_depth_view(obj_dir, view_i)                # (H,W) raw Blender Z
                valid = (z < 1e5) & (mask_i[..., 0] > 0.5)          # finite + foreground
            except (FileNotFoundError, IndexError, ValueError):
                z = torch.zeros(H, W)
                valid = torch.zeros(H, W, dtype=torch.bool)
            depths.append(z)
            depth_valid.append(valid)
        depths = torch.stack(depths)
        depth_valid = torch.stack(depth_valid)
        ref_depth = depths[0]
        ref_valid = depth_valid[0]
        out = {
            "uid": uid,
            "obj_dir": str(obj_dir),
            "latent": torch.from_numpy(
                np.load(latent_path_for_entry(self.root, entry, obj_dir)).astype(np.float32)
            ),
            "radius": float(cams["radius"]),
            "ref_K": cams["K"][0],
            "ref_c2w": cams["c2w_opengl"][0],
            "c2w_opengl": cams["c2w_opengl"][sel],
            "w2c": cams["w2c"][sel],
            "K": cams["K"][sel],
            "frames": load_views_at(obj_dir, idxs, subdir=self.bg_variant),
            "masks": masks,
            "width": cams["width"],
            "height": cams["height"],
            "view_indices": sel,
            "ref_depth": ref_depth,
            "ref_depth_valid": ref_valid,
            "depths": depths,
            "depth_valid": depth_valid,
            "num_orbit_views": cams["num_orbit_views"],
            "num_canonical_views": cams["num_canonical_views"],
        }
        cond_idxs = resolve_view_spec(
            self.cond_view_spec,
            v,
            obj_dir=obj_dir,
            subdir=self.cond_subdir,
            n_orbit_views=cams["num_orbit_views"],
            default_n=self.cond_default_views,
        )
        if cond_idxs is not None:
            cond_sel = torch.as_tensor(cond_idxs, dtype=torch.long)
            cond_masks = load_masks_at(obj_dir, cond_idxs)
            cond_depths = []
            cond_visibility_depths = []
            cond_target_depths = []
            cond_depth_valid = []
            cond_confs = []
            for view_i, mask_i in zip(cond_idxs, cond_masks):
                try:
                    z = load_depth_view_at(obj_dir, view_i, subdir=self.cond_depth_subdir)
                    valid = (z < 1e5) & (mask_i[..., 0] > 0.5)
                except (FileNotFoundError, IndexError, ValueError) as ex:
                    if self.strict_cond_depth and self.cond_depth_subdir:
                        raise FileNotFoundError(
                            f"missing conditioning depth uid={uid} view={view_i:03d} "
                            f"subdir={self.cond_depth_subdir!r}"
                        ) from ex
                    z = torch.zeros(H, W)
                    valid = torch.zeros(H, W, dtype=torch.bool)
                if self.cond_visibility_depth_subdir:
                    try:
                        z_vis = load_depth_view_at(
                            obj_dir, view_i, subdir=self.cond_visibility_depth_subdir
                        )
                    except (FileNotFoundError, IndexError, ValueError) as ex:
                        if self.strict_cond_depth:
                            raise FileNotFoundError(
                                f"missing conditioning visibility depth uid={uid} "
                                f"view={view_i:03d} "
                                f"subdir={self.cond_visibility_depth_subdir!r}"
                            ) from ex
                        z_vis = z
                else:
                    z_vis = z
                try:
                    z_target = load_depth_view(obj_dir, view_i)
                except (FileNotFoundError, IndexError, ValueError):
                    z_target = z
                try:
                    c = load_conf_view_at(obj_dir, view_i, subdir=self.cond_conf_subdir)
                except (FileNotFoundError, IndexError, ValueError):
                    c = torch.ones(H, W)
                cond_depths.append(z)
                cond_visibility_depths.append(z_vis)
                cond_target_depths.append(z_target)
                cond_depth_valid.append(valid)
                cond_confs.append(c)
            out.update({
                "cond_view_indices": cond_sel,
                "cond_frames": load_views_at(obj_dir, cond_idxs, subdir=self.cond_subdir),
                "cond_target_frames": load_views_at(obj_dir, cond_idxs, subdir=None),
                "cond_masks": cond_masks,
                "cond_w2c": cams["w2c"][cond_sel],
                "cond_c2w_opengl": cams["c2w_opengl"][cond_sel],
                "cond_K": cams["K"][cond_sel],
                "cond_depths": torch.stack(cond_depths),
                "cond_visibility_depths": torch.stack(cond_visibility_depths),
                "cond_target_depths": torch.stack(cond_target_depths),
                "cond_depth_valid": torch.stack(cond_depth_valid),
                "cond_confs": torch.stack(cond_confs),
            })
        return out
