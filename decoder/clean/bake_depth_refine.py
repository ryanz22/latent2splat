"""Bake DepthRefineUNet predictions to a new subdir matching DA3's layout.

For each train+eval+test object's available conditioning views, run the
DepthRefineUNet (frac-space residual on top of prior depth) and write the
refined depth as `<out_subdir>/depth_{i:03d}.npy` per-object. Fusion code
reads it via `--cond_depth_subdir <out_subdir>` or `@<out_subdir>` when
written to an override volume.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from decoder.clean.geometry import depth_bounds
from decoder.clean.phase2_data import (
    available_frame_indices, load_views_at, load_masks_at, load_depth_view_at,
)
from decoder.clean.train_phase2 import DepthRefineUNet, _depth_multiview_support_maps
from decoder.data import entry_relpath, entry_uid, load_cameras, object_dir_for_entry, zdepth_to_raydist


DEPTH_BG_SENTINEL = 1e10


def _frac_from_zdepth(z: torch.Tensor, K: torch.Tensor, mask: torch.Tensor,
                      c2w: torch.Tensor, radius: float, half_frac: float
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """(H,W) Blender Z + mask -> normalized depth-frac in [1e-4, 1-1e-4] + validity."""
    valid = torch.isfinite(z) & (z > 1e-6) & (z < 1e5) & (mask > 0.5)
    t = zdepth_to_raydist(z, K)
    d_near, d_far = depth_bounds(c2w, radius, half_frac)
    frac = ((t - d_near) / max(d_far - d_near, 1e-6)).clamp(1e-4, 1.0 - 1e-4)
    frac = torch.where(valid, frac, frac.new_full(frac.shape, 1e-4))
    return frac, valid.to(z.dtype)


def _zdepth_from_frac(frac: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor,
                      radius: float, half_frac: float) -> torch.Tensor:
    """Inverse of _frac_from_zdepth: frac -> ray-distance -> Blender Z."""
    d_near, d_far = depth_bounds(c2w, radius, half_frac)
    t = frac * (d_far - d_near) + d_near
    h, w = frac.shape[-2:]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vv, uu = torch.meshgrid(
        torch.arange(h, dtype=frac.dtype, device=frac.device),
        torch.arange(w, dtype=frac.dtype, device=frac.device),
        indexing="ij",
    )
    factor = torch.sqrt(((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2 + 1.0)
    return t / factor.clamp_min(1e-6)


@torch.no_grad()
def bake_object(head, obj_dir: Path, view_idxs: list[int], out_dir: Path,
                device, args: argparse.Namespace) -> int:
    if not view_idxs:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = load_cameras(obj_dir / "cameras.json")
    sel = torch.as_tensor(view_idxs, dtype=torch.long)
    K_all = cams["K"][sel].to(device)
    c2w_all = cams["c2w_opengl"][sel].to(device)
    radius = float(cams["radius"])

    frames = load_views_at(obj_dir, view_idxs, subdir=args.cond_subdir).to(device)
    masks_hw1 = load_masks_at(obj_dir, view_idxs).to(device)
    prior_depths = torch.stack([
        load_depth_view_at(obj_dir, i, subdir=args.cond_depth_subdir) for i in view_idxs
    ]).to(device)

    fracs, valids = [], []
    for k in range(len(view_idxs)):
        f, v = _frac_from_zdepth(
            prior_depths[k], K_all[k], masks_hw1[k, ..., 0], c2w_all[k], radius,
            args.half_frac,
        )
        fracs.append(f)
        valids.append(v)
    prior_frac = torch.stack(fracs)
    prior_valid = torch.stack(valids)

    mv_features = None
    if args.depth_refine_multiview_features:
        mv_features = _depth_multiview_support_maps(
            prior_depths,
            masks_hw1,
            K_all,
            c2w_all,
            radius,
            args.depth_refine_multiview_tol_frac,
            args.depth_refine_multiview_refs,
            args.depth_refine_multiview_radius_px,
        )

    rgb = frames.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    mask_chw = masks_hw1.permute(0, 3, 1, 2).clamp(0.0, 1.0)

    if args.infer_scale < 0.999:
        h, w = rgb.shape[-2:]
        sz = (max(8, int(round(h * args.infer_scale))),
              max(8, int(round(w * args.infer_scale))))
        rgb_in = F.interpolate(rgb, size=sz, mode="bilinear", align_corners=False)
        mask_in = F.interpolate(mask_chw, size=sz, mode="area").clamp(0.0, 1.0)
        prior_frac_in = F.interpolate(prior_frac[:, None], size=sz, mode="bilinear",
                                      align_corners=False)[:, 0]
        prior_valid_in = (F.interpolate(prior_valid[:, None], size=sz, mode="area")[:, 0] > 0.5).to(rgb.dtype)
        if mv_features is not None:
            mv_features = F.interpolate(mv_features, size=sz, mode="bilinear",
                                        align_corners=False).clamp(0.0, 1.0)
    else:
        rgb_in, mask_in, prior_frac_in, prior_valid_in = rgb, mask_chw, prior_frac, prior_valid

    feat = torch.cat([
        rgb_in * mask_in,
        mask_in,
        prior_frac_in[:, None],
        prior_valid_in[:, None],
    ], dim=1)
    if mv_features is not None:
        feat = torch.cat([feat, mv_features.to(device=feat.device, dtype=feat.dtype)], dim=1)
    delta = head(feat)
    if args.delta_scale > 0:
        delta = args.delta_scale * torch.tanh(delta)
    refined_frac_in = torch.sigmoid(
        torch.logit(prior_frac_in[:, None].clamp(1e-4, 1.0 - 1e-4)) + delta
    )[:, 0]
    refined_frac_in = torch.where(prior_valid_in > 0.5, refined_frac_in, prior_frac_in)

    if refined_frac_in.shape[-2:] != prior_frac.shape[-2:]:
        refined_frac = F.interpolate(refined_frac_in[:, None], size=prior_frac.shape[-2:],
                                     mode="bilinear", align_corners=False)[:, 0]
    else:
        refined_frac = refined_frac_in

    z_out = torch.zeros_like(prior_depths)
    for k in range(len(view_idxs)):
        z_out[k] = _zdepth_from_frac(refined_frac[k], K_all[k], c2w_all[k], radius,
                                     args.half_frac)
    # Mask BG to sentinel
    z_out = torch.where(masks_hw1[..., 0] > 0.5, z_out, torch.full_like(z_out, DEPTH_BG_SENTINEL))

    n = 0
    for k, i in enumerate(view_idxs):
        depth = z_out[k].cpu().numpy().astype(np.float32)
        np.save(out_dir / f"depth_{i:03d}.npy", depth)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=os.environ.get("PHASE2_DATA_ROOT", ""))
    ap.add_argument("--manifest", default=os.environ.get("PHASE2_MANIFEST", "manifest.json"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_subdir", default="da3_mv_refined")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--cond_subdir", default="ltx_decoded")
    ap.add_argument("--cond_depth_subdir", default="da3_ltx")
    ap.add_argument("--half_frac", type=float, default=0.5)
    ap.add_argument("--infer_scale", type=float, default=0.5)
    ap.add_argument("--delta_scale", type=float, default=0.7)
    ap.add_argument("--depth_refine_multiview_features", type=int, default=0)
    ap.add_argument("--depth_refine_multiview_refs", type=int, default=4)
    ap.add_argument("--depth_refine_multiview_tol_frac", type=float, default=0.02)
    ap.add_argument("--depth_refine_multiview_radius_px", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--splits", default="train,eval,test")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text())
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    entries: list = []
    for s in splits:
        entries.extend(manifest.get(s, []))
    print(f"[bake_refine] dataset={args.dataset_root} entries={len(entries)} splits={splits}",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    in_channels = 6 + (4 if args.depth_refine_multiview_features else 0)
    head = DepthRefineUNet(hidden=args.hidden, in_channels=in_channels).to(device)
    state = ckpt.get("depth_refine_head") or ckpt.get("model") or ckpt
    head.load_state_dict(state)
    head.eval()
    print(f"[bake_refine] loaded ckpt step={ckpt.get('step', '?')} mv={args.depth_refine_multiview_features}",
          flush=True)

    t0 = time.time()
    total = 0
    out_root = Path(args.out_root) if args.out_root else None
    for j, entry in enumerate(entries):
        uid = entry_uid(entry)
        obj_dir = object_dir_for_entry(root, entry)
        if not obj_dir.is_dir():
            continue
        try:
            view_idxs = available_frame_indices(obj_dir, subdir=args.cond_subdir)
        except Exception:
            continue
        out_dir = (out_root / entry_relpath(entry) / args.out_subdir) if out_root else (obj_dir / args.out_subdir)
        n = bake_object(head, obj_dir, view_idxs, out_dir, device, args)
        total += n
        if (j + 1) % 50 == 0 or j == len(entries) - 1:
            print(f"[bake_refine] {j + 1}/{len(entries)} obj, {total} files, "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
    print(f"[bake_refine] DONE. {total} depth files across {len(entries)} objects.",
          flush=True)


if __name__ == "__main__":
    main()
