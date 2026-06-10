"""Dataset + camera conversion for the VAE-latent → 3DGS decoder.

The dataset is LTX VAE encodes of Blender orbit renders, NOT denoised DiT
latents. View count and temporal dim are read from the data, not hardcoded:
  latent.npy   (C=128, T, H=24, W=16) float32   T=(input_frames-1)//8+1
               v2: 9 input frames → T=2;  v1: 25 → T=4
  frame_*.png  V supervision views, 512x768 RGB  (v2: 49, v1: 25)
  cameras.json OpenGL c2w + 3x3 intrinsics per frame

gsplat consumes OpenCV world-to-camera (looks +Z, +Y down). The renders are
stored OpenGL camera-to-world (looks -Z, +Y up), so we flip the Y/Z camera
axes and invert. That conversion is the single most correctness-critical
piece here and is unit-tested in tests/test_data.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# OpenGL camera axes -> OpenCV camera axes: keep X, flip Y and Z.
# Applied on the right of c2w (acts in camera space).
_GL_TO_CV = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def entry_uid(entry) -> str:
    """Return a UID from either a legacy manifest dict or a plain UID string."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        uid = entry.get("uid") or entry.get("object_id") or entry.get("id")
        if isinstance(uid, str) and uid:
            return uid
    raise KeyError(f"manifest entry does not contain a uid: {entry!r}")


def entry_tier(entry) -> str | None:
    """Return the dataset tier for combined manifests, if present."""
    if isinstance(entry, dict):
        tier = entry.get("tier")
        if isinstance(tier, str) and tier:
            return tier
    return None


def object_dir_for_entry(dataset_root: str | Path, entry) -> Path:
    """Resolve an object's frame/depth/camera directory.

    Legacy manifests store objects directly under ``dataset_root/<uid>``.
    Combined manifests include ``tier`` and store objects under
    ``dataset_root/<tier>/<uid>``.
    """
    root = Path(dataset_root)
    uid = entry_uid(entry)
    tier = entry_tier(entry)
    if tier:
        return root / tier / uid
    return root / uid


def entry_relpath(entry) -> Path:
    """Relative object path for sidecar outputs such as baked depth maps."""
    uid = entry_uid(entry)
    tier = entry_tier(entry)
    if tier:
        return Path(tier) / uid
    return Path(uid)


def latent_path_for_entry(dataset_root: str | Path, entry,
                          obj_dir: str | Path | None = None) -> Path:
    """Resolve latent.npy, including v8's separate Modal volume.

    Set ``L2S_LATENTS_V8_ROOT=/latents_v8`` for Jason's combined v7+v8 layout.
    The code accepts either ``/latents_v8/objaverse_v8/<uid>/latent.npy`` or
    ``/latents_v8/<uid>/latent.npy`` to keep local mirrors simple.
    """
    uid = entry_uid(entry)
    tier = entry_tier(entry)
    if tier == "objaverse_v8":
        override = os.environ.get("L2S_LATENTS_V8_ROOT", "")
        if override:
            base = Path(override)
            for p in (base / tier / uid / "latent.npy", base / uid / "latent.npy"):
                if p.exists():
                    return p
            return base / tier / uid / "latent.npy"
    odir = Path(obj_dir) if obj_dir is not None else object_dir_for_entry(dataset_root, entry)
    return odir / "latent.npy"


def opengl_c2w_to_opencv_w2c(c2w_gl: torch.Tensor) -> torch.Tensor:
    """(...,4,4) OpenGL camera-to-world -> OpenCV world-to-camera.

    c2w_cv = c2w_gl @ diag(1,-1,-1,1); w2c = inv(c2w_cv).
    """
    gl_to_cv = _GL_TO_CV.to(dtype=c2w_gl.dtype, device=c2w_gl.device)
    c2w_cv = c2w_gl @ gl_to_cv
    return torch.linalg.inv(c2w_cv)


def load_cameras(cameras_json: Path) -> dict:
    """Load cameras.json -> dict with stacked tensors.

    Returns:
      w2c:        (V,4,4) OpenCV world-to-camera
      K:          (V,3,3) intrinsics
      width,height: int
    """
    data = json.loads(Path(cameras_json).read_text())
    frames = data["frames"]
    c2w_gl = torch.tensor(
        np.stack([np.asarray(f["c2w_opengl"], dtype=np.float32) for f in frames])
    )
    K = torch.tensor(
        np.stack([np.asarray(f["intrinsics"], dtype=np.float32) for f in frames])
    )
    w2c = opengl_c2w_to_opencv_w2c(c2w_gl)
    return {
        "w2c": w2c,
        "c2w_opengl": c2w_gl,   # (V,4,4) raw OpenGL c2w — needed for unprojection
        "K": K,
        "width": int(data["width"]),
        "height": int(data["height"]),
        "radius": float(data["radius"]) if "radius" in data else None,  # per-object (Trial R4+)
        "num_orbit_views": int(data.get("num_orbit_views", len(frames))),
        "num_canonical_views": int(data.get("num_canonical_views", 0)),
    }


def _view_indices(num_views: int, indices: Sequence[int] | None = None) -> list[int]:
    if indices is None:
        return list(range(num_views))
    return [int(i) for i in indices if 0 <= int(i) < num_views]


def load_frames(obj_dir: Path, num_views: int = 25, subdir: str | None = None,
                indices: Sequence[int] | None = None) -> torch.Tensor:
    """Load frame_*.png -> (V,H,W,3) float32 in [0,1].

    `subdir` selects a background variant (e.g. "white"/"gray") from a same-named
    subdir; None loads the object dir's default (black-bg) frames.
    """
    base = Path(obj_dir) / subdir if subdir else Path(obj_dir)
    frames = []
    for i in _view_indices(num_views, indices):
        img = Image.open(base / f"frame_{i:03d}.png").convert("RGB")
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(frames))


def load_masks(obj_dir: Path, num_views: int,
               indices: Sequence[int] | None = None) -> torch.Tensor:
    """Load masks/mask_*.png -> (V,H,W,1) float32 foreground in {0,1} (255=object).

    Exact alpha-derived silhouette (clean even on a white background), used to
    mask the reconstruction/opacity loss to the object.
    """
    masks = []
    for i in _view_indices(num_views, indices):
        m = np.asarray(Image.open(Path(obj_dir) / "masks" / f"mask_{i:03d}.png").convert("L"))
        masks.append((m > 0).astype(np.float32))
    return torch.from_numpy(np.stack(masks))[..., None]  # (V,H,W,1)


def load_depth(obj_dir: Path) -> torch.Tensor:
    """Load the depth Z-pass -> (n,H,W) float32 raw Blender Z (metric, world units).

    Two on-disk layouts are supported, view-ordered by filename:
      depth/depth_NNN.npy    one per view         (Trial R4 / animals_v3)
      depth_NNN.npy at root  reference-view only  (animals_v4)
    Background ("no ray hit") pixels carry a large sentinel (~1e10); callers
    treat `depth < 1e5` (and/or the object mask) as valid foreground depth.
    """
    sub = Path(obj_dir) / "depth"
    files = sorted((sub if sub.is_dir() else Path(obj_dir)).glob("depth_*.npy"))
    depths = [np.load(f).astype(np.float32) for f in files]
    return torch.from_numpy(np.stack(depths))  # (n,H,W) raw Z


def load_depth_view(obj_dir: Path, index: int = 0) -> torch.Tensor:
    """Load one depth Z-pass view without materializing the whole depth stack."""
    obj_dir = Path(obj_dir)
    p = obj_dir / "depth" / f"depth_{index:03d}.npy"
    if not p.exists():
        p = obj_dir / f"depth_{index:03d}.npy"
    return torch.from_numpy(np.load(p).astype(np.float32))


def load_depth_views(obj_dir: Path, indices: Sequence[int]) -> torch.Tensor:
    """Load selected depth views without materializing the full depth stack."""
    return torch.stack([load_depth_view(obj_dir, int(i)) for i in indices], dim=0)


def zdepth_to_raydist(z: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Blender Z-pass (optical-axis depth, (H,W)) -> Euclidean ray distance (H,W).

    Means are parameterized by ray distance (mu = cam + t*ray_dir_unit), but
    Blender stores optical-axis Z; convert per pixel:
        t(u,v) = Z * sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)
    center factor = 1; ~1.14 at the 512x768 corner. (u=col, v=row.)
    """
    h, w = z.shape[-2:]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vv, uu = torch.meshgrid(torch.arange(h, dtype=z.dtype, device=z.device),
                            torch.arange(w, dtype=z.dtype, device=z.device), indexing="ij")
    factor = torch.sqrt(((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2 + 1.0)
    return z * factor


def depth_target_on_grid(z: torch.Tensor, mask: torch.Tensor, K: torch.Tensor,
                         grid_h: int, grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame Z-pass (H,W) + fg mask (H,W) -> (grid_h*grid_w,) target ray-distance +
    bool validity, to supervise a decoder's per-grid-cell predicted depth.

    Converts Z->ray-distance, then average-pools valid (foreground, finite) depth
    into each grid cell; a cell with no valid pixel is marked invalid.
    """
    t = zdepth_to_raydist(z, K)                       # (H,W)
    valid = (mask > 0) & (z < 1e5)                    # finite foreground (1e10 = no hit)
    vf = valid.to(t.dtype)
    num = F.adaptive_avg_pool2d((vf * t)[None, None], (grid_h, grid_w))[0, 0]
    den = F.adaptive_avg_pool2d(vf[None, None], (grid_h, grid_w))[0, 0]
    target = num / den.clamp_min(1e-6)
    return target.reshape(-1), (den > 0).reshape(-1)


class ObjaverseLatentDataset(torch.utils.data.Dataset):
    """One sample per object: latent + target views + cameras.

    Object dirs ({uid}/latent.npy, frame_*.png, cameras.json) live under
    `dataset_root`. The manifest (keyed by split) defaults to
    `dataset_root/manifest.json`, but animals_v1 keeps it in a sibling
    `manifests/` dir, so `manifest_path` can override the location.
    """

    def __init__(self, dataset_root: str | Path, split: str = "train",
                 manifest_path: str | Path | None = None, bg_variant: str | None = None,
                 load_depths: bool = False, max_views: int = 0,
                 view_indices: Sequence[int] | None = None):
        self.root = Path(dataset_root)
        self.bg_variant = bg_variant   # None (default black) | "black" | "white" | "gray"
        self.load_depths = load_depths
        self.max_views = int(max_views)
        self.view_indices = list(view_indices) if view_indices is not None else None
        mpath = Path(manifest_path) if manifest_path else self.root / "manifest.json"
        manifest = json.loads(Path(mpath).read_text())
        self.entries = manifest[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        uid = entry_uid(entry)
        obj_dir = object_dir_for_entry(self.root, entry)
        cams = load_cameras(obj_dir / "cameras.json")
        v = cams["w2c"].shape[0]
        if self.view_indices is not None:
            idxs = _view_indices(v, self.view_indices)
        elif self.max_views > 0 and v > self.max_views:
            orbit = max(1, min(v, int(cams.get("num_orbit_views", v))))
            idxs = [int(round(x)) for x in np.linspace(0, orbit - 1, self.max_views)]
            idxs = list(dict.fromkeys(i for i in idxs if 0 <= i < v))
        else:
            idxs = list(range(v))
        if not idxs:
            raise ValueError(f"no valid view indices for {uid}")
        sample = {
            "uid": uid,
            "obj_dir": str(obj_dir),
            "frames": load_frames(obj_dir, num_views=v, subdir=self.bg_variant,
                                  indices=idxs),  # (V,H,W,3)
            "w2c": cams["w2c"][idxs],          # (V,4,4)
            "c2w_opengl": cams["c2w_opengl"][idxs],  # (V,4,4)
            "K": cams["K"][idxs],              # (V,3,3)
            "all_w2c": cams["w2c"],
            "all_c2w_opengl": cams["c2w_opengl"],
            "all_K": cams["K"],
            "width": cams["width"],
            "height": cams["height"],
            "radius": cams["radius"],    # per-object (Trial R4+) or None
            "num_orbit_views": cams["num_orbit_views"],
            "num_canonical_views": cams["num_canonical_views"],
            "view_indices": torch.tensor(idxs, dtype=torch.long),
        }
        # latent is absent in format-preview trials (Trial R4); present in full runs.
        latent_path = latent_path_for_entry(self.root, entry, obj_dir)
        if latent_path.exists():
            sample["latent"] = torch.from_numpy(np.load(latent_path).astype(np.float32))
        if (obj_dir / "masks").is_dir():
            sample["masks"] = load_masks(obj_dir, v, indices=idxs)    # (V,H,W,1)
        if self.load_depths and ((obj_dir / "depth").is_dir() or (obj_dir / "depth_000.npy").exists()):
            if len(idxs) == v and idxs == list(range(v)):
                sample["depth"] = load_depth(obj_dir)   # (n,H,W) raw Z, bg sentinel ~1e10
            else:
                sample["depth"] = load_depth_views(obj_dir, idxs)
        return sample
