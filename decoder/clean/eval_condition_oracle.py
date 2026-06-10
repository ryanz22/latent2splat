"""Evaluate conditioning-frame quality against same-view dataset targets.

This is a cheap first-principles diagnostic for the current decoder path:
if decoded/fixed conditioning frames are already blurry or color-shifted at
the same camera views, the 3DGS decoder cannot recover those details without
additional learned hallucination. If they are sharp and correctly colored, the
remaining loss is coming from 3D lifting, fusion, or rendering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from decoder.clean.metrics import fg_masked_psnr, sharpness_ratio
from decoder.clean.phase2_data import load_masks_at, load_views_at, resolve_view_spec
from decoder.data import entry_uid, load_cameras, object_dir_for_entry


def _splits(spec: str) -> list[str]:
    if spec == "all":
        return ["train", "eval", "test"]
    return [s.strip() for s in spec.split(",") if s.strip()]


def _fit_train_affine(root: Path, manifest: dict, cond_subdir: str,
                      view_spec: str, max_objects: int, max_px_per_obj: int,
                      ridge: float) -> torch.Tensor:
    entries = manifest.get("train", [])
    if max_objects > 0:
        entries = entries[:max_objects]
    ata = torch.zeros(4, 4, dtype=torch.float64)
    atb = torch.zeros(4, 3, dtype=torch.float64)
    n_px = 0
    n_views = 0
    for entry in entries:
        obj_dir = object_dir_for_entry(root, entry)
        cams = load_cameras(obj_dir / "cameras.json")
        idxs = resolve_view_spec(
            view_spec, cams["w2c"].shape[0], obj_dir=obj_dir,
            subdir=cond_subdir, n_orbit_views=cams["num_orbit_views"],
        )
        if not idxs:
            continue
        cond = load_views_at(obj_dir, idxs, subdir=cond_subdir).reshape(-1, 3)
        target = load_views_at(obj_dir, idxs, subdir=None).reshape(-1, 3)
        mask = load_masks_at(obj_dir, idxs).reshape(-1) > 0.5
        if not mask.any():
            continue
        x = cond[mask].to(torch.float64)
        y = target[mask].to(torch.float64)
        if max_px_per_obj > 0 and x.shape[0] > max_px_per_obj:
            stride = (x.shape[0] + max_px_per_obj - 1) // max_px_per_obj
            x = x[::stride]
            y = y[::stride]
        feat = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
        ata += feat.T @ feat
        atb += feat.T @ y
        n_px += int(x.shape[0])
        n_views += len(idxs)
    if n_px < 16:
        raise RuntimeError("not enough foreground pixels to fit affine color calibration")
    reg = torch.eye(4, dtype=torch.float64) * max(float(ridge), 0.0)
    reg[-1, -1] = 0.0
    sol = torch.linalg.solve(ata + reg, atb).to(torch.float32)
    w = sol[:3].flatten().tolist()
    b = sol[3].tolist()
    print(f"[oracle] train_affine views={n_views} px={n_px} W={w} b={b}", flush=True)
    return sol


def _apply_affine(frames: torch.Tensor, sol: torch.Tensor | None) -> torch.Tensor:
    if sol is None:
        return frames
    flat = frames.reshape(-1, 3)
    out = flat @ sol[:3] + sol[3]
    return out.reshape_as(frames).clamp(0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--cond_subdir", default="ltx_decoded")
    ap.add_argument("--cond_view_indices", default="available")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fit_train_affine", type=int, default=0)
    ap.add_argument("--color_calib_max_objects", type=int, default=0)
    ap.add_argument("--color_calib_sample_px", type=int, default=20000)
    ap.add_argument("--color_calib_ridge", type=float, default=1e-4)
    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    manifest = json.loads(Path(args.manifest).read_text())
    affine = None
    if args.fit_train_affine:
        affine = _fit_train_affine(
            root, manifest, args.cond_subdir, args.cond_view_indices,
            args.color_calib_max_objects, args.color_calib_sample_px,
            args.color_calib_ridge,
        )

    rows: list[dict] = []
    for split in _splits(args.split):
        entries = manifest.get(split, [])
        if args.limit > 0:
            entries = entries[:args.limit]
        split_rows = []
        for obj_i, entry in enumerate(entries, 1):
            uid = entry_uid(entry)
            obj_dir = object_dir_for_entry(root, entry)
            try:
                cams = load_cameras(obj_dir / "cameras.json")
                idxs = resolve_view_spec(
                    args.cond_view_indices,
                    cams["w2c"].shape[0],
                    obj_dir=obj_dir,
                    subdir=args.cond_subdir,
                    n_orbit_views=cams["num_orbit_views"],
                )
                if not idxs:
                    continue
                cond = _apply_affine(load_views_at(obj_dir, idxs, args.cond_subdir), affine)
                target = load_views_at(obj_dir, idxs, None)
                masks = load_masks_at(obj_dir, idxs) > 0.5
            except Exception as ex:
                print(f"[oracle] skip {split}/{entry_uid(entry)[:10]} {type(ex).__name__}",
                      flush=True)
                continue
            psnr = fg_masked_psnr(cond, target, masks)
            sharp = sharpness_ratio(cond, target, masks[..., 0])
            mae = float((cond - target).abs()[masks.expand_as(cond)].mean())
            row = {
                "split": split,
                "uid": uid,
                "views": len(idxs),
                "fg_psnr": psnr,
                "sharpness": sharp,
                "fg_mae": mae,
            }
            rows.append(row)
            split_rows.append(row)
            if args.progress_every > 0 and obj_i % args.progress_every == 0:
                print(f"[oracle] {split} progress {obj_i}/{len(entries)}", flush=True)
        if split_rows:
            n = len(split_rows)
            mean_psnr = sum(r["fg_psnr"] for r in split_rows) / n
            mean_sharp = sum(r["sharpness"] for r in split_rows) / n
            mean_mae = sum(r["fg_mae"] for r in split_rows) / n
            mean_views = sum(r["views"] for r in split_rows) / n
            print(
                f"[oracle] {split} objects={n} views/obj={mean_views:.1f} "
                f"fg_psnr={mean_psnr:.2f} sharp={mean_sharp:.3f} mae={mean_mae:.4f}",
                flush=True,
            )
    if rows:
        n = len(rows)
        print(
            f"[oracle] ALL objects={n} fg_psnr={sum(r['fg_psnr'] for r in rows) / n:.2f} "
            f"sharp={sum(r['sharpness'] for r in rows) / n:.3f} "
            f"mae={sum(r['fg_mae'] for r in rows) / n:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
