"""Cache VGGT multi-view depth predictions for decoder conditioning.

VGGT is a feed-forward multi-view geometry model. This script consumes the same
conditioning frames as the decoder, predicts per-view depth in one multi-view
forward pass, aligns that relative depth to the existing silhouette hull scale,
and writes ``depth_NNN.npy`` files compatible with ``train_phase2``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF

from decoder.clean.phase2_data import frame_path_at, load_masks_at, resolve_view_spec
from decoder.clean.geometry import depth_bounds
from decoder.clean.predict_depth_anything import (
    _align_relative_depth,
    _raydist_to_zdepth,
    _visual_hull_depths,
)
from decoder.data import entry_relpath, entry_uid, load_cameras, object_dir_for_entry, zdepth_to_raydist


def _entries(manifest_path: Path, split: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    if split == "all":
        out = []
        for key in ("train", "eval", "test"):
            out.extend(manifest.get(key, []))
        return out
    return manifest[split]


def _preprocess_pad(path: Path, target_size: int = 518) -> tuple[torch.Tensor, dict]:
    """Load RGB, resize with aspect preservation, and pad to square."""
    img = Image.open(path).convert("RGB")
    width, height = img.size
    if width >= height:
        new_w = target_size
        new_h = max(14, round(height * (new_w / width) / 14) * 14)
    else:
        new_h = target_size
        new_w = max(14, round(width * (new_h / height) / 14) * 14)
    new_w = min(new_w, target_size)
    new_h = min(new_h, target_size)
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    tensor = TF.ToTensor()(img)
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
    info = {
        "width": width,
        "height": height,
        "new_width": new_w,
        "new_height": new_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }
    return tensor, info


def _restore_padded_map(x: torch.Tensor, info: dict) -> torch.Tensor:
    """Undo ``_preprocess_pad`` for a single square map."""
    top = int(info["pad_top"])
    left = int(info["pad_left"])
    new_h = int(info["new_height"])
    new_w = int(info["new_width"])
    crop = x[top:top + new_h, left:left + new_w]
    out = F.interpolate(
        crop[None, None].float(),
        size=(int(info["height"]), int(info["width"])),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return out


def _as_view_maps(x: torch.Tensor) -> torch.Tensor:
    """Normalize VGGT map outputs to ``(S, H, W)``."""
    x = x.detach().float().cpu()
    if x.ndim == 5 and x.shape[0] == 1 and x.shape[-1] == 1:
        return x[0, ..., 0]
    if x.ndim == 5 and x.shape[0] == 1:
        return x[0].squeeze(-1)
    if x.ndim == 4 and x.shape[0] == 1:
        return x[0]
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[..., 0]
    if x.ndim == 3:
        return x
    raise ValueError(f"unsupported VGGT map shape: {tuple(x.shape)}")


def _fit_global_affine_to_hull(rel: list[torch.Tensor],
                               hull_z: torch.Tensor,
                               masks: torch.Tensor,
                               K: torch.Tensor,
                               c2w: torch.Tensor,
                               radius: float) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fit one relative-depth affine for all VGGT views.

    VGGT is multi-view, so its view-to-view depth scale carries useful geometry.
    Per-view alignment destroys that consistency. This global fit keeps the
    relative inter-view structure while still putting the prediction in the
    dataset camera scale.
    """
    xs, ys = [], []
    for i, r in enumerate(rel):
        near, far = depth_bounds(c2w[i], radius, 0.5)
        hull_t = zdepth_to_raydist(hull_z[i], K[i])
        valid = (
            (masks[i, ..., 0] > 0.5)
            & torch.isfinite(r)
            & torch.isfinite(hull_t)
            & (hull_z[i] < 1e5)
        )
        if valid.sum() < 32 or float(r[valid].std()) < 1e-6:
            continue
        x = r[valid].float()
        y = hull_t[valid].float().clamp(float(near), float(far))
        y_lo, y_hi = torch.quantile(y, y.new_tensor([0.02, 0.98]))
        keep = (y >= y_lo) & (y <= y_hi)
        if keep.sum() < 32:
            continue
        xs.append(x[keep])
        ys.append(y[keep])
    if not xs:
        return None
    x_all = torch.cat(xs)
    y_all = torch.cat(ys)
    if x_all.numel() < 32 or float(x_all.std()) < 1e-6:
        return None
    A = torch.stack([x_all, torch.ones_like(x_all)], dim=1)
    sol = torch.linalg.lstsq(A, y_all[:, None]).solution[:, 0]
    return sol[0], sol[1]


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--image_subdir", default="ltx_decoded")
    ap.add_argument("--out_subdir", default="vggt_ltx")
    ap.add_argument("--out_root", default="",
                    help="If set, write <out_root>/<tier>/<uid>/<out_subdir>/ sidecars.")
    ap.add_argument("--view_indices", default="available")
    ap.add_argument("--model", default="facebook/VGGT-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target_size", type=int, default=518)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--hull_scale", type=float, default=0.25)
    ap.add_argument("--hull_samples", type=int, default=64)
    ap.add_argument("--hull_min_view_frac", type=float, default=0.7)
    ap.add_argument("--hull_mask_margin", type=int, default=1)
    ap.add_argument("--conf_quantile", type=float, default=0.0)
    ap.add_argument("--align_scope", default="global", choices=["global", "per_view"])
    args = ap.parse_args()

    from vggt.models.vggt import VGGT

    root = Path(args.dataset_root)
    entries = _entries(Path(args.manifest), args.split)
    if args.limit > 0:
        entries = entries[:args.limit]
    out_root = Path(args.out_root) if args.out_root else None
    device = torch.device(args.device)
    dtype = torch.float16
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    model = VGGT.from_pretrained(args.model).to(device).eval()
    print(f"[predict_vggt_depth] objects={len(entries)} model={args.model} out={args.out_subdir}", flush=True)

    for n, entry in enumerate(entries, 1):
        uid = entry_uid(entry)
        obj_dir = object_dir_for_entry(root, entry)
        out_dir = (
            out_root / entry_relpath(entry) / args.out_subdir
            if out_root else obj_dir / args.out_subdir
        )
        meta_path = out_dir / "metadata.json"
        if meta_path.exists() and not args.overwrite:
            if any(out_dir.glob("depth_*.npy")):
                print(f"[predict_vggt_depth] {n}/{len(entries)} skip {uid[:8]}: exists", flush=True)
                continue
            print(
                f"[predict_vggt_depth] {n}/{len(entries)} refresh {uid[:8]}: "
                "metadata exists but no depth files",
                flush=True,
            )
        cams = load_cameras(obj_dir / "cameras.json")
        try:
            idxs = resolve_view_spec(
                args.view_indices,
                cams["K"].shape[0],
                obj_dir=obj_dir,
                subdir=args.image_subdir,
                n_orbit_views=cams["num_orbit_views"],
            )
        except FileNotFoundError as exc:
            print(f"[predict_vggt_depth] {n}/{len(entries)} skip {uid[:8]}: {exc}", flush=True)
            continue
        if not idxs:
            print(f"[predict_vggt_depth] skip {uid[:8]}: no views", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        images, infos = [], []
        for view_i in idxs:
            img, info = _preprocess_pad(
                frame_path_at(obj_dir, view_i, subdir=args.image_subdir),
                target_size=args.target_size,
            )
            images.append(img)
            infos.append(info)
        images_t = torch.stack(images).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            pred = model(images_t)
        depth_sq = _as_view_maps(pred["depth"])
        conf_sq = _as_view_maps(pred["depth_conf"])

        masks = load_masks_at(obj_dir, idxs).float()
        sel = torch.as_tensor(idxs, dtype=torch.long)
        K = cams["K"][sel]
        c2w = cams["c2w_opengl"][sel]
        w2c = cams["w2c"][sel]
        hull_z = _visual_hull_depths(
            masks, K, c2w, w2c, float(cams["radius"]), args.hull_scale,
            args.hull_samples, args.hull_min_view_frac, args.hull_mask_margin,
        )

        rel_maps = [_restore_padded_map(depth_sq[i], infos[i]) for i in range(len(idxs))]
        conf_maps = [_restore_padded_map(conf_sq[i], infos[i]) for i in range(len(idxs))]
        valid_masks = []
        for local_i, conf in enumerate(conf_maps):
            mask = masks[local_i, ..., 0].bool()
            valid_mask = mask
            if args.conf_quantile > 0 and mask.any():
                q = torch.quantile(conf[mask], min(max(args.conf_quantile, 0.0), 0.99))
                valid_mask = mask & (conf >= q)
            valid_masks.append(valid_mask)

        global_affine = None
        if args.align_scope == "global":
            global_affine = _fit_global_affine_to_hull(
                rel_maps, hull_z, masks, K, c2w, float(cams["radius"])
            )

        conf_ranges = []
        for local_i, view_i in enumerate(idxs):
            rel = rel_maps[local_i]
            conf = conf_maps[local_i]
            mask = masks[local_i, ..., 0].bool()
            valid_mask = valid_masks[local_i]
            if global_affine is not None:
                near, far = depth_bounds(c2w[local_i], float(cams["radius"]), 0.5)
                t = (global_affine[0].to(rel.dtype) * rel + global_affine[1].to(rel.dtype)).clamp(
                    float(near), float(far)
                )
            else:
                if args.align_scope == "global":
                    print(
                        f"[predict_vggt_depth] {uid[:8]} falling back to per-view alignment",
                        flush=True,
                    )
                t = _align_relative_depth(rel, hull_z[local_i], valid_mask, K[local_i],
                                          c2w[local_i], float(cams["radius"]))
            z = _raydist_to_zdepth(t, K[local_i])
            z = torch.where(mask, z, z.new_full(z.shape, 1e10))
            np.save(out_dir / f"depth_{view_i:03d}.npy", z.numpy().astype(np.float32))
            np.save(out_dir / f"conf_{view_i:03d}.npy", conf.numpy().astype(np.float32))
            if mask.any():
                vals = conf[mask]
                conf_ranges.append([float(vals.min()), float(vals.max())])

        meta = {
            "uid": uid,
            "model": args.model,
            "image_subdir": args.image_subdir,
            "view_indices": idxs,
            "target_size": args.target_size,
            "align": f"{args.align_scope}_affine_to_silhouette_visual_hull",
            "global_affine": (
                [float(global_affine[0]), float(global_affine[1])]
                if global_affine is not None else None
            ),
            "conf_quantile": args.conf_quantile,
            "confidence_ranges": conf_ranges,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[predict_vggt_depth] {n}/{len(entries)} wrote {uid[:8]} views={idxs}", flush=True)


if __name__ == "__main__":
    main()
