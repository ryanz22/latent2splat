"""Bake Stage B depth predictions to a new subdir matching DA3's layout.

For each train+eval+test object's available conditioning views, run the
DepthAnchorNet and write the resulting depth as `<out_subdir>/depth_{i:03d}.npy`
into the per-object directory. Existing fusion code reads it via
`--cond_depth_subdir <out_subdir>`.
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

from decoder.clean.phase2_data import (
    available_frame_indices, depth_path_at, load_views_at, load_depth_view_at,
)
from decoder.clean.train_depth_anchor import DepthAnchorNet, _depth_from_log, LOG_EPS
from decoder.data import entry_relpath, entry_uid, object_dir_for_entry


DEPTH_BG_SENTINEL = 1e10


def _load_manifest(p: Path) -> dict:
    return json.loads(p.read_text())


@torch.no_grad()
def bake_object(model, obj_dir: Path, view_idxs: list[int], out_dir: Path,
                device, infer_scale: float, use_da3_input: bool,
                cond_subdir: str = "ltx_decoded",
                da3_subdir: str = "da3_ltx") -> int:
    if not view_idxs:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    rgbs = load_views_at(obj_dir, view_idxs, subdir=cond_subdir).to(device)
    rgbs = rgbs.permute(0, 3, 1, 2)
    H, W = rgbs.shape[-2:]
    if infer_scale < 0.999:
        size = (max(8, int(round(H * infer_scale))),
                max(8, int(round(W * infer_scale))))
        rgbs_in = F.interpolate(rgbs, size=size, mode="bilinear", align_corners=False)
    else:
        rgbs_in = rgbs
    if use_da3_input:
        da3s = []
        for i in view_idxs:
            d = load_depth_view_at(obj_dir, i, subdir=da3_subdir).to(device)
            da3s.append(d)
        da3 = torch.stack(da3s).unsqueeze(1)    # (K, 1, H, W)
        if da3.shape[-2:] != rgbs_in.shape[-2:]:
            da3 = F.interpolate(da3, size=rgbs_in.shape[-2:], mode="bilinear", align_corners=False)
        da3_valid = (torch.isfinite(da3) & (da3 > 1e-3) & (da3 < 1e6)).float()
        da3_log = torch.where(da3_valid > 0.5, torch.log(da3.clamp_min(LOG_EPS)),
                              torch.zeros_like(da3))
        pred_log = model(rgbs_in, da3_log, da3_valid)
    else:
        pred_log = model(rgbs_in)
    if pred_log.shape[-2:] != (H, W):
        pred_log = F.interpolate(pred_log, size=(H, W), mode="bilinear", align_corners=False)
    pred = _depth_from_log(pred_log)[:, 0]    # (K, H, W)
    # Load masks (uint8) — write background as sentinel to match DA3 convention
    from PIL import Image
    n = 0
    for k, i in enumerate(view_idxs):
        depth = pred[k].cpu().numpy().astype(np.float32)
        # Mask BG to sentinel
        mask_path = obj_dir / "masks" / f"mask_{i:03d}.png"
        if mask_path.exists():
            m = np.asarray(Image.open(mask_path).convert("L")) > 127
            if m.shape != depth.shape:
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (depth.shape[1], depth.shape[0]), Image.NEAREST)) > 127
            depth = np.where(m, depth, DEPTH_BG_SENTINEL)
        out_p = out_dir / f"depth_{i:03d}.npy"
        np.save(out_p, depth)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=os.environ.get("PHASE2_DATA_ROOT", ""))
    ap.add_argument("--manifest", default=os.environ.get("PHASE2_MANIFEST", "manifest.json"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cond_subdir", default="ltx_decoded")
    ap.add_argument("--da3_subdir", default="da3_ltx")
    ap.add_argument("--out_subdir", default="da3_anchor")
    ap.add_argument("--out_root", default="",
                    help="If set, write to <out_root>/<uid>/<out_subdir>/ instead of <obj_dir>/<out_subdir>/. Used to bake to a different volume.")
    ap.add_argument("--infer_scale", type=float, default=0.5)
    ap.add_argument("--splits", default="train,eval,test")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_objects", type=int, default=0,
                    help="Optional total object limit after split concatenation.")
    ap.add_argument("--max_written_objects", type=int, default=0,
                    help="Optional stop after this many objects produce at least one depth file.")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = _load_manifest(manifest_path)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    entries = []
    for s in splits:
        entries.extend(manifest.get(s, []))
    if args.offset > 0:
        entries = entries[args.offset:]
    if args.max_objects > 0:
        entries = entries[:args.max_objects]
    print(f"[bake] dataset={args.dataset_root} entries={len(entries)} splits={splits}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cargs = ckpt.get("args", {})
    decoder_ch = int(cargs.get("decoder_ch", 128))
    use_da3_input = bool(cargs.get("use_da3_input", 1))
    model = DepthAnchorNet(decoder_ch=decoder_ch, pretrained=False,
                           use_da3_input=use_da3_input).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[bake] loaded ckpt step={ckpt.get('step', '?')} from {args.ckpt}", flush=True)

    t0 = time.time()
    total = 0
    written_objects = 0
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
        if use_da3_input:
            view_idxs = [
                i for i in view_idxs
                if depth_path_at(obj_dir, i, subdir=args.da3_subdir).exists()
            ]
        out_dir = (out_root / entry_relpath(entry) / args.out_subdir) if out_root else (obj_dir / args.out_subdir)
        n = bake_object(model, obj_dir, view_idxs, out_dir, device, args.infer_scale,
                        use_da3_input=use_da3_input,
                        cond_subdir=args.cond_subdir,
                        da3_subdir=args.da3_subdir)
        if n > 0:
            written_objects += 1
        total += n
        if (j + 1) % 50 == 0 or j == len(entries) - 1:
            print(f"[bake] {j + 1}/{len(entries)} objects, {written_objects} written objects, "
                  f"{total} files written, "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
        if args.max_written_objects > 0 and written_objects >= args.max_written_objects:
            print(f"[bake] reached max_written_objects={args.max_written_objects}", flush=True)
            break
    print(f"[bake] DONE. {total} depth files written across {len(entries)} objects.", flush=True)


if __name__ == "__main__":
    main()
