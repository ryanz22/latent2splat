"""Cache frozen monocular depth predictions for decoder conditioning.

This is a feed-forward inference-time depth prior: it consumes the same cached
RGB frames used by the decoder (typically ``ltx_decoded/frame_NNN.png``), runs a
frozen depth model, metric-aligns the relative depth to a silhouette visual hull,
and writes ``depth_NNN.npy`` files that ``train_phase2`` can load with
``--cond_depth_subdir``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from decoder.clean.geometry import depth_bounds
from decoder.clean.phase2_data import frame_path_at, load_masks_at, resolve_view_spec
from decoder.data import entry_relpath, entry_uid, load_cameras, object_dir_for_entry, zdepth_to_raydist
from decoder.clean.geometry import ray_dirs_world


def _entries(manifest_path: Path, split: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    if split == "all":
        out = []
        for key in ("train", "eval", "test"):
            out.extend(manifest.get(key, []))
        return out
    return manifest[split]


def _scaled_intrinsics(K_all: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    K_s = K_all.clone()
    K_s[:, 0, 0] *= sx
    K_s[:, 0, 2] *= sx
    K_s[:, 1, 1] *= sy
    K_s[:, 1, 2] *= sy
    return K_s


def _raydist_to_zdepth(t: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    h, w = t.shape[-2:]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vv, uu = torch.meshgrid(
        torch.arange(h, dtype=t.dtype, device=t.device),
        torch.arange(w, dtype=t.dtype, device=t.device),
        indexing="ij",
    )
    factor = torch.sqrt(((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2 + 1.0)
    return t / factor.clamp_min(1e-6)


def _visual_hull_depths(fg: torch.Tensor, K_all: torch.Tensor, c2w_all: torch.Tensor,
                        w2c_all: torch.Tensor, radius: float, scale: float,
                        samples: int, min_view_frac: float, mask_margin: int) -> torch.Tensor:
    """Silhouette-carved front depth for the selected conditioning views."""
    n_views, H, W = fg.shape[0], fg.shape[1], fg.shape[2]
    scale = min(max(scale, 0.02), 1.0)
    vh = max(8, int(round(H * scale)))
    vw = max(8, int(round(W * scale)))
    device, dtype = fg.device, fg.dtype
    masks = F.interpolate(fg.permute(0, 3, 1, 2), size=(vh, vw), mode="area").clamp(0.0, 1.0)
    if mask_margin > 0:
        masks = F.max_pool2d(masks, kernel_size=2 * mask_margin + 1, stride=1, padding=mask_margin)
    K_s = _scaled_intrinsics(K_all.to(device=device, dtype=dtype), vw / W, vh / H)
    c2w_s = c2w_all.to(device=device, dtype=dtype)
    w2c_s = w2c_all.to(device=device, dtype=dtype)
    sample_frac = torch.linspace(0.0, 1.0, max(samples, 2), device=device, dtype=dtype)
    min_views = max(1, min(n_views, int(np.ceil(min_view_frac * n_views))))
    out_z = []
    for src_i in range(n_views):
        dirs = ray_dirs_world(K_s[src_i], c2w_s[src_i], vh, vw).to(device=device, dtype=dtype)
        origin = c2w_s[src_i, :3, 3]
        near, far = depth_bounds(c2w_s[src_i], radius, 0.5)
        t = near + sample_frac * (far - near)
        pts = origin.view(1, 1, 3) + dirs[:, None, :] * t.view(1, -1, 1)
        pts_flat = pts.reshape(-1, 3)
        inside_count = torch.zeros(pts_flat.shape[0], device=device, dtype=dtype)
        for ref_i in range(n_views):
            cam = pts_flat @ w2c_s[ref_i, :3, :3].T + w2c_s[ref_i, :3, 3]
            z = cam[:, 2]
            fx, fy = K_s[ref_i, 0, 0], K_s[ref_i, 1, 1]
            cx, cy = K_s[ref_i, 0, 2], K_s[ref_i, 1, 2]
            u = fx * (cam[:, 0] / z.clamp_min(1e-6)) + cx
            v = fy * (cam[:, 1] / z.clamp_min(1e-6)) + cy
            inb = (z > 1e-6) & (u >= 0) & (u <= vw - 1) & (v >= 0) & (v <= vh - 1)
            grid_x = (u / max(vw - 1, 1)) * 2.0 - 1.0
            grid_y = (v / max(vh - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], -1).view(1, -1, 1, 2)
            m = F.grid_sample(masks[ref_i:ref_i + 1], grid, mode="bilinear",
                              padding_mode="zeros", align_corners=True)
            inside_count += ((m.view(-1) > 0.25) & inb).to(dtype)
        inside = inside_count.reshape(vh * vw, -1) >= min_views
        src_fg = masks[src_i, 0].reshape(-1) > 0.25
        any_hit = inside.any(dim=1) & src_fg
        first = inside.float().argmax(dim=1)
        t_hit = t[first]
        pts_hit = origin.view(1, 3) + dirs * t_hit[:, None]
        cam_src = pts_hit @ w2c_s[src_i, :3, :3].T + w2c_s[src_i, :3, 3]
        z_hit = torch.where(any_hit, cam_src[:, 2], cam_src.new_full((vh * vw,), 1e10))
        z_full = F.interpolate(z_hit.view(1, 1, vh, vw), size=(H, W),
                               mode="bilinear", align_corners=False)[0, 0]
        out_z.append(z_full)
    return torch.stack(out_z)


def _align_relative_depth(rel: torch.Tensor, hull_z: torch.Tensor, mask: torch.Tensor,
                          K: torch.Tensor, c2w: torch.Tensor, radius: float) -> torch.Tensor:
    near, far = depth_bounds(c2w, radius, 0.5)
    hull_t = zdepth_to_raydist(hull_z, K)
    valid = (mask > 0.5) & torch.isfinite(rel) & torch.isfinite(hull_t) & (hull_z < 1e5)
    if valid.sum() < 32 or float(rel[valid].std()) < 1e-6:
        r = rel
        lo, hi = torch.quantile(r[mask > 0.5], r.new_tensor([0.02, 0.98]))
        t = near + ((r - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0) * (far - near)
        return t
    x = rel[valid].float()
    y = hull_t[valid].float().clamp(float(near), float(far))
    y_lo, y_hi = torch.quantile(y, y.new_tensor([0.02, 0.98]))
    keep = (y >= y_lo) & (y <= y_hi)
    x, y = x[keep], y[keep]
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    sol = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
    t = sol[0].to(rel.dtype) * rel + sol[1].to(rel.dtype)
    return t.clamp(float(near), float(far))


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--image_subdir", default="ltx_decoded")
    ap.add_argument("--out_subdir", default="depth_anything_ltx")
    ap.add_argument("--out_root", default="",
                    help="If set, write <out_root>/<tier>/<uid>/<out_subdir>/ sidecars.")
    ap.add_argument("--view_indices", default="available")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--hull_scale", type=float, default=0.25)
    ap.add_argument("--hull_samples", type=int, default=64)
    ap.add_argument("--hull_min_view_frac", type=float, default=0.7)
    ap.add_argument("--hull_mask_margin", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    root = Path(args.dataset_root)
    entries = _entries(Path(args.manifest), args.split)
    if args.limit > 0:
        entries = entries[:args.limit]
    out_root = Path(args.out_root) if args.out_root else None
    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()
    print(f"[predict_depth] objects={len(entries)} model={args.model} out={args.out_subdir}", flush=True)

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
                print(f"[predict_depth] {n}/{len(entries)} skip {uid[:8]}: exists", flush=True)
                continue
            print(
                f"[predict_depth] {n}/{len(entries)} refresh {uid[:8]}: "
                "metadata exists but no depth files",
                flush=True,
            )
        cams = load_cameras(obj_dir / "cameras.json")
        try:
            idxs = resolve_view_spec(
                args.view_indices, cams["K"].shape[0], obj_dir=obj_dir,
                subdir=args.image_subdir, n_orbit_views=cams["num_orbit_views"],
            )
        except FileNotFoundError as exc:
            print(f"[predict_depth] {n}/{len(entries)} skip {uid[:8]}: {exc}", flush=True)
            continue
        if not idxs:
            print(f"[predict_depth] skip {uid[:8]}: no views", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        masks = load_masks_at(obj_dir, idxs).float()
        sel = torch.as_tensor(idxs, dtype=torch.long)
        K = cams["K"][sel]
        c2w = cams["c2w_opengl"][sel]
        w2c = cams["w2c"][sel]
        hull_z = _visual_hull_depths(
            masks, K, c2w, w2c, float(cams["radius"]), args.hull_scale,
            args.hull_samples, args.hull_min_view_frac, args.hull_mask_margin,
        )
        rel_ranges = []
        for local_i, view_i in enumerate(idxs):
            img = Image.open(frame_path_at(obj_dir, view_i, subdir=args.image_subdir)).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pred = model(**inputs).predicted_depth
            pred = F.interpolate(pred[:, None], size=(cams["height"], cams["width"]),
                                 mode="bicubic", align_corners=False)[0, 0].float().cpu()
            mask = masks[local_i, ..., 0].bool()
            t = _align_relative_depth(pred, hull_z[local_i], mask, K[local_i],
                                      c2w[local_i], float(cams["radius"]))
            z = _raydist_to_zdepth(t, K[local_i])
            z = torch.where(mask, z, z.new_full(z.shape, 1e10))
            np.save(out_dir / f"depth_{view_i:03d}.npy", z.numpy().astype(np.float32))
            if mask.any():
                vals = pred[mask]
                rel_ranges.append([float(vals.min()), float(vals.max())])
        meta = {
            "uid": uid,
            "model": args.model,
            "image_subdir": args.image_subdir,
            "view_indices": idxs,
            "align": "affine_to_silhouette_visual_hull",
            "relative_depth_ranges": rel_ranges,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[predict_depth] {n}/{len(entries)} wrote {uid[:8]} views={idxs}", flush=True)


if __name__ == "__main__":
    main()
