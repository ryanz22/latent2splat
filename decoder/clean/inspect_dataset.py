"""Validate a latent2splat dataset before launching expensive training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EXPECTED_LATENT_SHAPES = ((128, 2, 24, 16), (128, 2, 36, 24))


def _frame_ids(path: Path) -> set[int]:
    out = set()
    for p in path.glob("frame_*.png"):
        stem = p.stem.removeprefix("frame_")
        if stem.isdigit():
            out.add(int(stem))
    return out


def _sidecar_base(obj_dir: Path, subdir: str | None, env_name: str) -> Path:
    if not subdir:
        return obj_dir
    if subdir.startswith("@"):
        actual = subdir.lstrip("@")
        override = os.environ.get(env_name, "")
        if override:
            uid = obj_dir.name
            parent = obj_dir.parent.name
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
        return obj_dir / actual
    return obj_dir / subdir


def _mask_ids(path: Path) -> set[int]:
    out = set()
    for p in (path / "masks").glob("mask_*.png"):
        stem = p.stem.removeprefix("mask_")
        if stem.isdigit():
            out.add(int(stem))
    return out


def _depth_ids(path: Path, subdir: str | None = None) -> set[int]:
    if subdir:
        base = _sidecar_base(path, subdir, "L2S_DEPTH_OVERRIDE_ROOT")
    else:
        base = path / "depth" if (path / "depth").is_dir() else path
    out = set()
    for p in base.glob("depth_*.npy"):
        stem = p.stem.removeprefix("depth_")
        if stem.isdigit():
            out.add(int(stem))
    return out


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _entry_uid(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        uid = entry.get("uid") or entry.get("object_id") or entry.get("id")
        if isinstance(uid, str) and uid:
            return uid
    raise KeyError(f"manifest entry does not contain a uid: {entry!r}")


def _entry_tier(entry: Any) -> str | None:
    if isinstance(entry, dict):
        tier = entry.get("tier")
        if isinstance(tier, str) and tier:
            return tier
    return None


def _object_dir_for_entry(root: Path, entry: Any) -> Path:
    uid = _entry_uid(entry)
    tier = _entry_tier(entry)
    return root / tier / uid if tier else root / uid


def _latent_path_for_entry(root: Path, entry: Any, obj_dir: Path) -> Path:
    uid = _entry_uid(entry)
    tier = _entry_tier(entry)
    if tier == "objaverse_v8":
        override = os.environ.get("L2S_LATENTS_V8_ROOT", "")
        if override:
            base = Path(override)
            for p in (base / tier / uid / "latent.npy", base / uid / "latent.npy"):
                if p.exists():
                    return p
            return base / tier / uid / "latent.npy"
    return obj_dir / "latent.npy"


def _camera_summary(path: Path) -> tuple[int, int, int]:
    data = json.loads(path.read_text())
    return int(data["width"]), int(data["height"]), len(data.get("frames", []))


def _camera_data(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    frames = data.get("frames", [])
    return {
        "width": int(data["width"]),
        "height": int(data["height"]),
        "radius": float(data.get("radius", 1.0)),
        "num_views": len(frames),
        "num_orbit_views": int(data.get("num_orbit_views", len(frames))),
        "K": np.stack([np.asarray(f["intrinsics"], dtype=np.float64) for f in frames]),
        "c2w_opengl": np.stack([np.asarray(f["c2w_opengl"], dtype=np.float64) for f in frames]),
    }


def _resolve_view_ids(spec: str, cams: dict[str, Any], obj_dir: Path,
                      sidecar: str | None = None) -> list[int]:
    spec = (spec or "").strip()
    if spec == "available":
        ids = _depth_ids(obj_dir, sidecar)
        return [i for i in sorted(ids) if 0 <= i < cams["num_views"]]
    if spec.startswith("spread:"):
        n = max(1, int(spec.split(":", 1)[1]))
        limit = max(1, min(cams["num_views"], cams["num_orbit_views"]))
        if n == 1:
            return [0]
        idx = np.rint(np.arange(n, dtype=np.float64) * (float(limit) / n)).astype(np.int64)
        return sorted(set(int(i) for i in np.clip(idx, 0, limit - 1)))
    if spec:
        return [
            int(x) for x in spec.split(",")
            if x.strip() and 0 <= int(x) < cams["num_views"]
        ]
    return [0]


def _resize_nearest(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(x.shape[-2:]) == shape:
        return x.astype(np.float32, copy=False)
    h, w = shape
    yy = np.linspace(0, x.shape[-2] - 1, h).round().astype(np.int64)
    xx = np.linspace(0, x.shape[-1] - 1, w).round().astype(np.int64)
    return x[np.ix_(yy, xx)].astype(np.float32, copy=False)


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape:
        mask = _resize_nearest(mask.astype(np.float32), shape) > 0.5
    return mask


def _zdepth_factor(shape: tuple[int, int], K: np.ndarray) -> np.ndarray:
    h, w = shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.sqrt(((xx - cx) / fx) ** 2 + ((yy - cy) / fy) ** 2 + 1.0)


def _depth_bounds(c2w_opengl: np.ndarray, radius: float, half_frac: float) -> tuple[float, float]:
    d_cam = float(np.linalg.norm(c2w_opengl[:3, 3]))
    half = half_frac * float(radius)
    return max(d_cam - half, 0.05 * d_cam), d_cam + half


def _depth_frac(z: np.ndarray, K: np.ndarray, c2w_opengl: np.ndarray,
                radius: float, half_frac: float) -> np.ndarray:
    near, far = _depth_bounds(c2w_opengl, radius, half_frac)
    t = z.astype(np.float64) * _zdepth_factor(z.shape[-2:], K)
    return (t - near) / max(far - near, 1e-8)


def _finite_fg_depth(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask & np.isfinite(z) & (z > 0.0) & (z < 1e5)


def _sample_pair(x: np.ndarray, y: np.ndarray, max_px: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if max_px > 0 and x.shape[0] > max_px:
        stride = int(np.ceil(x.shape[0] / max_px))
        x = x[::stride]
        y = y[::stride]
    return x, y


def _safe_percentiles(x: np.ndarray) -> list[float] | None:
    if x.size == 0:
        return None
    return [float(v) for v in np.percentile(x, [1.0, 50.0, 99.0])]


def _mae_corr(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if x.size == 0:
        return None, None
    mae = float(np.mean(np.abs(x - y)))
    if x.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return mae, None
    corr = float(np.corrcoef(x, y)[0, 1])
    return mae, corr


def _affine_stats(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> dict[str, float | None]:
    if x.size < 16 or float(np.std(x)) < 1e-12:
        return {"a": None, "b": None, "mae": None}
    feat = np.stack([x, np.ones_like(x)], axis=1)
    ata = feat.T @ feat
    ata[0, 0] += ridge
    atb = feat.T @ y
    try:
        a, b = np.linalg.solve(ata, atb)
    except np.linalg.LinAlgError:
        return {"a": None, "b": None, "mae": None}
    pred = a * x + b
    return {"a": float(a), "b": float(b), "mae": float(np.mean(np.abs(pred - y)))}


def diagnose_depth_sidecars(dataset_root: str | Path,
                            manifest_path: str | Path,
                            sidecars: list[str],
                            splits: list[str] | None = None,
                            limit_per_split: int = 4,
                            view_spec: str = "0,6,12,18,24,30,36,42,48",
                            max_px_per_view: int = 5000,
                            half_frac: float = 0.5) -> dict[str, Any]:
    """Compare cached sidecar depths against GT depth using training's depth fraction.

    The goal is to catch convention bugs cheaply: missing files, raw metric-scale
    mismatch, inverse-depth storage, or depths that correlate only after the
    normalized near/far ray-depth conversion used by ``train_phase2``.
    """
    root = Path(dataset_root)
    manifest = json.loads(Path(manifest_path).read_text())
    split_names = splits or [s for s in ("train", "eval", "test") if s in manifest]
    out: dict[str, Any] = {
        "root": str(root),
        "manifest": str(manifest_path),
        "sidecars": sidecars,
        "view_spec": view_spec,
        "max_px_per_view": max_px_per_view,
        "splits": {},
    }
    totals: dict[str, dict[str, Any]] = {}

    for split in split_names:
        entries = manifest.get(split, [])
        if limit_per_split > 0:
            entries = entries[:limit_per_split]
        split_out: dict[str, Any] = {"objects": len(entries), "sidecars": {}}
        for sidecar in sidecars:
            raw_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            frac_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            inv_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            raw_ranges: list[np.ndarray] = []
            gt_ranges: list[np.ndarray] = []
            frac_ranges: list[np.ndarray] = []
            missing = 0
            shape_mismatch = 0
            views_seen = 0
            objects_seen = 0
            examples: list[dict[str, Any]] = []
            for entry in entries:
                uid = _entry_uid(entry)
                obj_dir = _object_dir_for_entry(root, entry)
                try:
                    cams = _camera_data(obj_dir / "cameras.json")
                except Exception as ex:
                    examples.append({"uid": uid, "error": f"camera:{type(ex).__name__}"})
                    continue
                view_ids = _resolve_view_ids(view_spec, cams, obj_dir, sidecar=sidecar)
                obj_has_view = False
                for view_i in view_ids:
                    gt_path = _sidecar_base(obj_dir, None, "L2S_DEPTH_OVERRIDE_ROOT") / "depth" / f"depth_{view_i:03d}.npy"
                    if not gt_path.exists():
                        gt_path = obj_dir / f"depth_{view_i:03d}.npy"
                    pred_path = _sidecar_base(obj_dir, sidecar, "L2S_DEPTH_OVERRIDE_ROOT") / f"depth_{view_i:03d}.npy"
                    if not gt_path.exists() or not pred_path.exists():
                        missing += 1
                        continue
                    try:
                        gt = np.load(gt_path).astype(np.float32)
                        pred_raw = np.load(pred_path).astype(np.float32)
                        if pred_raw.shape != gt.shape:
                            shape_mismatch += 1
                            pred_raw = _resize_nearest(pred_raw, gt.shape)
                        mask = _load_mask(obj_dir / "masks" / f"mask_{view_i:03d}.png", gt.shape)
                    except Exception as ex:
                        examples.append({
                            "uid": uid,
                            "view": view_i,
                            "sidecar": sidecar,
                            "error": type(ex).__name__,
                        })
                        continue
                    valid = _finite_fg_depth(gt, mask) & _finite_fg_depth(pred_raw, mask)
                    if int(valid.sum()) < 16:
                        continue
                    K = cams["K"][view_i]
                    c2w = cams["c2w_opengl"][view_i]
                    radius = float(cams["radius"])
                    pred_frac = _depth_frac(pred_raw, K, c2w, radius, half_frac)
                    gt_frac = _depth_frac(gt, K, c2w, radius, half_frac)
                    valid = valid & np.isfinite(pred_frac) & np.isfinite(gt_frac)
                    if int(valid.sum()) < 16:
                        continue
                    raw_x, raw_y = _sample_pair(pred_raw[valid], gt[valid], max_px_per_view)
                    frac_x, frac_y = _sample_pair(pred_frac[valid], gt_frac[valid], max_px_per_view)
                    inv_x, inv_y = _sample_pair(
                        1.0 / np.clip(pred_raw[valid], 1e-6, None),
                        gt[valid],
                        max_px_per_view,
                    )
                    raw_pairs.append((raw_x, raw_y))
                    frac_pairs.append((frac_x, frac_y))
                    inv_pairs.append((inv_x, inv_y))
                    raw_ranges.append(raw_x)
                    gt_ranges.append(raw_y)
                    frac_ranges.append(frac_x)
                    views_seen += 1
                    obj_has_view = True
                if obj_has_view:
                    objects_seen += 1
            if raw_pairs:
                raw_x = np.concatenate([p[0] for p in raw_pairs])
                raw_y = np.concatenate([p[1] for p in raw_pairs])
                frac_x = np.concatenate([p[0] for p in frac_pairs])
                frac_y = np.concatenate([p[1] for p in frac_pairs])
                inv_x = np.concatenate([p[0] for p in inv_pairs])
                inv_y = np.concatenate([p[1] for p in inv_pairs])
                raw_mae, raw_corr = _mae_corr(raw_x, raw_y)
                frac_mae, frac_corr = _mae_corr(frac_x, frac_y)
                inv_mae, inv_corr = _mae_corr(inv_x, inv_y)
                side_out = {
                    "objects_with_valid_depth": objects_seen,
                    "views_with_valid_depth": views_seen,
                    "missing_view_files": missing,
                    "shape_mismatch_views": shape_mismatch,
                    "sampled_pixels": int(raw_x.size),
                    "raw_z_p01_p50_p99": _safe_percentiles(raw_x),
                    "gt_z_p01_p50_p99": _safe_percentiles(raw_y),
                    "raw_frac_p01_p50_p99": _safe_percentiles(frac_x),
                    "gt_frac_p01_p50_p99": _safe_percentiles(frac_y),
                    "raw_z_mae": raw_mae,
                    "raw_z_corr": raw_corr,
                    "raw_z_affine": _affine_stats(raw_x, raw_y),
                    "frac_mae": frac_mae,
                    "frac_corr": frac_corr,
                    "frac_affine": _affine_stats(frac_x, frac_y),
                    "inv_z_corr_to_gt_z": inv_corr,
                    "inv_z_affine_to_gt_z": _affine_stats(inv_x, inv_y),
                    "examples": examples[:5],
                }
            else:
                side_out = {
                    "objects_with_valid_depth": 0,
                    "views_with_valid_depth": 0,
                    "missing_view_files": missing,
                    "shape_mismatch_views": shape_mismatch,
                    "sampled_pixels": 0,
                    "examples": examples[:5],
                }
            split_out["sidecars"][sidecar] = side_out
            total = totals.setdefault(sidecar, {
                "objects_with_valid_depth": 0,
                "views_with_valid_depth": 0,
                "missing_view_files": 0,
                "shape_mismatch_views": 0,
                "sampled_pixels": 0,
            })
            for key in total:
                total[key] += int(side_out.get(key, 0) or 0)
        out["splits"][split] = split_out
    out["totals"] = totals
    return out


def inspect_condition_coverage(dataset_root: str | Path,
                               manifest_path: str | Path,
                               cond_subdir: str | None,
                               cond_depth_subdir: str | None,
                               splits: list[str] | None = None,
                               limit_per_split: int = 0,
                               min_views: int = 1) -> dict[str, Any]:
    """Cheap coverage report for conditioning sidecars without reading latents."""
    root = Path(dataset_root)
    manifest = json.loads(Path(manifest_path).read_text())
    split_names = splits or [s for s in ("train", "eval", "test") if s in manifest]
    out: dict[str, Any] = {
        "root": str(root),
        "manifest": str(manifest_path),
        "cond_subdir": cond_subdir,
        "cond_depth_subdir": cond_depth_subdir,
        "min_views": min_views,
        "splits": {},
    }

    def empty_stats() -> dict[str, Any]:
        return {
            "objects": 0,
            "objects_with_frames": 0,
            "objects_with_depth": 0,
            "objects_with_both": 0,
            "objects_with_min_views": 0,
            "frame_counts_min": None,
            "frame_counts_max": 0,
            "depth_counts_min": None,
            "depth_counts_max": 0,
            "examples_missing": [],
        }

    def update_minmax(stats: dict[str, Any], prefix: str, count: int) -> None:
        key_min = f"{prefix}_counts_min"
        key_max = f"{prefix}_counts_max"
        stats[key_min] = count if stats[key_min] is None else min(stats[key_min], count)
        stats[key_max] = max(stats[key_max], count)

    totals = empty_stats()
    by_tier: dict[str, dict[str, Any]] = {}
    for split in split_names:
        entries = manifest.get(split, [])
        if limit_per_split > 0:
            entries = entries[:limit_per_split]
        split_stats = empty_stats()
        for entry in entries:
            uid = _entry_uid(entry)
            tier = _entry_tier(entry) or "legacy"
            obj_dir = _object_dir_for_entry(root, entry)
            tier_stats = by_tier.setdefault(tier, empty_stats())
            frame_ids = (
                _frame_ids(_sidecar_base(obj_dir, cond_subdir, "L2S_COND_OVERRIDE_ROOT"))
                if cond_subdir else _frame_ids(obj_dir)
            )
            depth_ids = (
                _depth_ids(obj_dir, cond_depth_subdir)
                if cond_depth_subdir else _depth_ids(obj_dir)
            )
            counts = {"frame": len(frame_ids), "depth": len(depth_ids)}
            for stats in (split_stats, tier_stats, totals):
                stats["objects"] += 1
                update_minmax(stats, "frame", counts["frame"])
                update_minmax(stats, "depth", counts["depth"])
                has_frames = counts["frame"] > 0
                has_depth = counts["depth"] > 0
                has_both = has_frames and has_depth
                has_min = counts["frame"] >= min_views and counts["depth"] >= min_views
                stats["objects_with_frames"] += int(has_frames)
                stats["objects_with_depth"] += int(has_depth)
                stats["objects_with_both"] += int(has_both)
                stats["objects_with_min_views"] += int(has_min)
                if (not has_min) and len(stats["examples_missing"]) < 10:
                    stats["examples_missing"].append({
                        "uid": uid,
                        "tier": tier,
                        "frames": counts["frame"],
                        "depth": counts["depth"],
                    })
        out["splits"][split] = split_stats
    out["tiers"] = by_tier
    out["totals"] = totals
    return out


def inspect_dataset(dataset_root: str | Path,
                    manifest_path: str | Path | None = None,
                    splits: list[str] | None = None,
                    limit_per_split: int = 0,
                    cond_subdir: str | None = None,
                    cond_depth_subdir: str | None = None,
                    expected_latent_shape: tuple[tuple[int, ...], ...] = EXPECTED_LATENT_SHAPES) -> dict[str, Any]:
    root = Path(dataset_root)
    manifest_file = Path(manifest_path) if manifest_path else root / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict keyed by split")
    split_names = splits or [s for s in ("train", "eval", "test") if s in manifest]

    summary: dict[str, Any] = {
        "root": str(root),
        "manifest": str(manifest_file),
        "splits": {k: len(v) for k, v in manifest.items() if isinstance(v, list)},
        "checked": {},
        "errors": [],
        "warnings": [],
    }
    widths: set[int] = set()
    heights: set[int] = set()
    latent_shapes: set[tuple[int, ...]] = set()
    total_checked = 0

    for split in split_names:
        entries = manifest.get(split)
        if not isinstance(entries, list):
            summary["errors"].append(f"split {split!r} missing or not a list")
            continue
        if limit_per_split > 0:
            entries = entries[:limit_per_split]
        split_stats = {
            "objects": len(entries),
            "missing": 0,
            "latents": 0,
            "frames_min": None,
            "frames_max": 0,
            "masks_min": None,
            "masks_max": 0,
            "depth_min": None,
            "depth_max": 0,
            "cond_frames_min": None,
            "cond_frames_max": 0,
            "cond_depth_min": None,
            "cond_depth_max": 0,
        }
        for entry in entries:
            total_checked += 1
            try:
                uid = _entry_uid(entry)
            except (KeyError, ValueError) as ex:
                summary["errors"].append(f"{split}: {ex}")
                split_stats["missing"] += 1
                continue
            obj_dir = _object_dir_for_entry(root, entry)
            if not obj_dir.is_dir():
                summary["errors"].append(f"{split}/{uid}: object directory missing")
                split_stats["missing"] += 1
                continue
            try:
                width, height, n_views = _camera_summary(obj_dir / "cameras.json")
            except Exception as ex:
                summary["errors"].append(f"{split}/{uid}: cameras.json failed: {type(ex).__name__}: {ex}")
                split_stats["missing"] += 1
                continue
            widths.add(width)
            heights.add(height)

            latent_file = _latent_path_for_entry(root, entry, obj_dir)
            if not latent_file.exists():
                summary["errors"].append(f"{split}/{uid}: missing latent.npy")
            else:
                try:
                    latent_shape = tuple(np.load(latent_file, mmap_mode="r").shape)
                    latent_shapes.add(latent_shape)
                    split_stats["latents"] += 1
                    if expected_latent_shape and latent_shape not in expected_latent_shape:
                        summary["warnings"].append(
                            f"{split}/{uid}: latent shape {latent_shape}, expected one of {expected_latent_shape}"
                        )
                except Exception as ex:
                    summary["errors"].append(f"{split}/{uid}: latent.npy failed: {type(ex).__name__}: {ex}")

            frames = _frame_ids(obj_dir)
            masks = _mask_ids(obj_dir)
            depths = _depth_ids(obj_dir)
            for key, count in (("frames", len(frames)), ("masks", len(masks)), ("depth", len(depths))):
                min_key, max_key = f"{key}_min", f"{key}_max"
                split_stats[min_key] = count if split_stats[min_key] is None else min(split_stats[min_key], count)
                split_stats[max_key] = max(split_stats[max_key], count)
            missing_frames = set(range(n_views)) - frames
            missing_masks = set(range(n_views)) - masks
            if missing_frames:
                summary["errors"].append(f"{split}/{uid}: missing {len(missing_frames)} frame PNGs")
            if missing_masks:
                summary["errors"].append(f"{split}/{uid}: missing {len(missing_masks)} mask PNGs")
            if frames:
                size = _image_size(obj_dir / f"frame_{min(frames):03d}.png")
                if size != (width, height):
                    summary["errors"].append(
                        f"{split}/{uid}: frame size {size}, cameras say {(width, height)}"
                    )
            if masks:
                size = _image_size(obj_dir / "masks" / f"mask_{min(masks):03d}.png")
                if size != (width, height):
                    summary["errors"].append(
                        f"{split}/{uid}: mask size {size}, cameras say {(width, height)}"
                    )

            if cond_subdir:
                cond_frames = _frame_ids(
                    _sidecar_base(obj_dir, cond_subdir, "L2S_COND_OVERRIDE_ROOT")
                )
                split_stats["cond_frames_min"] = (
                    len(cond_frames) if split_stats["cond_frames_min"] is None
                    else min(split_stats["cond_frames_min"], len(cond_frames))
                )
                split_stats["cond_frames_max"] = max(split_stats["cond_frames_max"], len(cond_frames))
                if not cond_frames:
                    summary["warnings"].append(f"{split}/{uid}: no {cond_subdir}/frame_*.png")
            if cond_depth_subdir:
                cond_depths = _depth_ids(obj_dir, cond_depth_subdir)
                split_stats["cond_depth_min"] = (
                    len(cond_depths) if split_stats["cond_depth_min"] is None
                    else min(split_stats["cond_depth_min"], len(cond_depths))
                )
                split_stats["cond_depth_max"] = max(split_stats["cond_depth_max"], len(cond_depths))
                if not cond_depths:
                    summary["warnings"].append(f"{split}/{uid}: no {cond_depth_subdir}/depth_*.npy")
        summary["checked"][split] = split_stats

    summary["total_checked"] = total_checked
    summary["widths"] = sorted(widths)
    summary["heights"] = sorted(heights)
    summary["latent_shapes"] = [list(s) for s in sorted(latent_shapes)]
    summary["ok"] = not summary["errors"]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--splits", default="train,eval,test")
    ap.add_argument("--limit_per_split", type=int, default=0)
    ap.add_argument("--cond_subdir", default="")
    ap.add_argument("--cond_depth_subdir", default="")
    ap.add_argument("--condition_coverage", type=int, default=0,
                    help="Only report conditioning frame/depth sidecar coverage; skip latent checks.")
    ap.add_argument("--condition_coverage_min_views", type=int, default=1)
    ap.add_argument("--depth_diag_sidecars", default="",
                    help="Comma-separated depth sidecars to compare against GT depth.")
    ap.add_argument("--depth_diag_views", default="0,6,12,18,24,30,36,42,48",
                    help="'available', 'spread:N', or comma-separated view ids.")
    ap.add_argument("--depth_diag_max_px_per_view", type=int, default=5000)
    ap.add_argument("--depth_diag_half_frac", type=float, default=0.5)
    ap.add_argument("--warn_only", type=int, default=0)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.condition_coverage:
        manifest = args.manifest or str(Path(args.dataset_root) / "manifest.json")
        summary = inspect_condition_coverage(
            args.dataset_root,
            manifest,
            args.cond_subdir or None,
            args.cond_depth_subdir or None,
            splits=splits,
            limit_per_split=args.limit_per_split,
            min_views=args.condition_coverage_min_views,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.depth_diag_sidecars:
        sidecars = [s.strip() for s in args.depth_diag_sidecars.split(",") if s.strip()]
        manifest = args.manifest or str(Path(args.dataset_root) / "manifest.json")
        summary = diagnose_depth_sidecars(
            args.dataset_root,
            manifest,
            sidecars,
            splits=splits,
            limit_per_split=args.limit_per_split,
            view_spec=args.depth_diag_views,
            max_px_per_view=args.depth_diag_max_px_per_view,
            half_frac=args.depth_diag_half_frac,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    summary = inspect_dataset(
        args.dataset_root,
        args.manifest,
        splits=splits,
        limit_per_split=args.limit_per_split,
        cond_subdir=args.cond_subdir or None,
        cond_depth_subdir=args.cond_depth_subdir or None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["errors"] and not args.warn_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
