"""Cache Depth Anything 3 pose-conditioned depth for decoder conditioning.

DA3 consumes the same LTX-decoded/source RGB frames as the decoder plus the
known OpenCV w2c/K cameras. It writes ``depth_NNN.npy`` and ``conf_NNN.npy``
files in the layout already understood by ``train_phase2``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from decoder.clean.phase2_data import frame_path_at, resolve_view_spec
from decoder.data import entry_relpath, entry_uid, load_cameras, object_dir_for_entry


def _entries(manifest_path: Path, split: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    if split == "all":
        out = []
        for key in ("train", "eval", "test"):
            out.extend(manifest.get(key, []))
        return out
    return manifest[split]


def _resize_map(x: np.ndarray, height: int, width: int) -> np.ndarray:
    if tuple(x.shape[-2:]) == (height, width):
        return x.astype(np.float32)
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    out = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    return out.numpy().astype(np.float32)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--image_subdir", default="ltx_decoded")
    ap.add_argument("--out_subdir", default="da3_ltx")
    ap.add_argument("--out_root", default="",
                    help="If set, write <out_root>/<tier>/<uid>/<out_subdir>/ sidecars.")
    ap.add_argument("--view_indices", default="available")
    ap.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--process_res_method", default="upper_bound_resize")
    ap.add_argument("--ref_view_strategy", default="saddle_balanced")
    ap.add_argument("--align_to_input_ext_scale", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=0,
                    help="Stop after writing this many new objects; skipped existing/missing inputs do not count.")
    ap.add_argument("--max_consecutive_missing_inputs", type=int, default=0,
                    help="If >0, stop after this many consecutive objects without input frames. "
                    "Useful for chunked LTX-decode sidecars on sparse manifests.")
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    from depth_anything_3.api import DepthAnything3

    root = Path(args.dataset_root)
    entries = _entries(Path(args.manifest), args.split)
    if args.offset > 0:
        entries = entries[args.offset:]
    if args.limit > 0:
        entries = entries[:args.limit]
    out_root = Path(args.out_root) if args.out_root else None
    model = DepthAnything3.from_pretrained(args.model).to(args.device)
    print(
        f"[predict_da3_depth] objects={len(entries)} model={args.model} "
        f"out={args.out_subdir} out_root={args.out_root or '<dataset>'} "
        f"offset={args.offset} max_new={args.max_new}",
        flush=True,
    )

    written = 0
    skipped_existing = 0
    skipped_missing_inputs = 0
    skipped_bad_inputs = 0
    missing_input_streak = 0
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
                skipped_existing += 1
                missing_input_streak = 0
                print(f"[predict_da3_depth] {n}/{len(entries)} skip {uid[:8]}: exists", flush=True)
                continue
            print(
                f"[predict_da3_depth] {n}/{len(entries)} refresh {uid[:8]}: "
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
            skipped_missing_inputs += 1
            missing_input_streak += 1
            print(f"[predict_da3_depth] {n}/{len(entries)} skip {uid[:8]}: {exc}", flush=True)
            if (args.max_consecutive_missing_inputs > 0
                    and missing_input_streak >= args.max_consecutive_missing_inputs):
                print(
                    "[predict_da3_depth] reached "
                    f"max_consecutive_missing_inputs={args.max_consecutive_missing_inputs}",
                    flush=True,
                )
                break
            continue
        if not idxs:
            skipped_missing_inputs += 1
            missing_input_streak += 1
            print(f"[predict_da3_depth] skip {uid[:8]}: no views", flush=True)
            if (args.max_consecutive_missing_inputs > 0
                    and missing_input_streak >= args.max_consecutive_missing_inputs):
                print(
                    "[predict_da3_depth] reached "
                    f"max_consecutive_missing_inputs={args.max_consecutive_missing_inputs}",
                    flush=True,
                )
                break
            continue
        missing_input_streak = 0
        image_paths = [str(frame_path_at(obj_dir, i, subdir=args.image_subdir)) for i in idxs]
        sel = torch.as_tensor(idxs, dtype=torch.long)
        extrinsics = cams["w2c"][sel].numpy().astype(np.float32)
        intrinsics = cams["K"][sel].numpy().astype(np.float32)
        try:
            pred = model.inference(
                image=image_paths,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                align_to_input_ext_scale=bool(args.align_to_input_ext_scale),
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                ref_view_strategy=args.ref_view_strategy,
            )
        except (OSError, ValueError) as exc:
            skipped_bad_inputs += 1
            print(
                f"[predict_da3_depth] {n}/{len(entries)} skip {uid[:8]}: "
                f"bad image input ({type(exc).__name__}: {exc})",
                flush=True,
            )
            continue
        depth = np.asarray(pred.depth, dtype=np.float32)
        conf = np.asarray(getattr(pred, "conf", np.ones_like(depth)), dtype=np.float32)
        height, width = int(cams["height"]), int(cams["width"])
        maps: list[tuple[int, np.ndarray, np.ndarray]] = []
        try:
            for local_i, view_i in enumerate(idxs):
                z = _resize_map(depth[local_i], height, width)
                c = _resize_map(conf[local_i], height, width)
                mask_path = obj_dir / "masks" / f"mask_{view_i:03d}.png"
                if mask_path.exists():
                    from PIL import Image

                    with Image.open(mask_path) as mask_im:
                        mask = np.asarray(mask_im.convert("L")) > 0
                    z = np.where(mask, z, np.full_like(z, 1e10))
                maps.append((view_i, z.astype(np.float32), c.astype(np.float32)))
        except (OSError, ValueError) as exc:
            skipped_bad_inputs += 1
            print(
                f"[predict_da3_depth] {n}/{len(entries)} skip {uid[:8]}: "
                f"bad mask/depth input ({type(exc).__name__}: {exc})",
                flush=True,
            )
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for view_i, z, c in maps:
            np.save(out_dir / f"depth_{view_i:03d}.npy", z)
            np.save(out_dir / f"conf_{view_i:03d}.npy", c)
        meta = {
            "uid": uid,
            "model": args.model,
            "image_subdir": args.image_subdir,
            "view_indices": idxs,
            "process_res": args.process_res,
            "process_res_method": args.process_res_method,
            "ref_view_strategy": args.ref_view_strategy,
            "align_to_input_ext_scale": bool(args.align_to_input_ext_scale),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        written += 1
        print(f"[predict_da3_depth] {n}/{len(entries)} wrote {uid[:8]} views={idxs}", flush=True)
        if args.max_new > 0 and written >= args.max_new:
            print(f"[predict_da3_depth] reached max_new={args.max_new}", flush=True)
            break
    print(
        f"[predict_da3_depth] done written={written} skipped_existing={skipped_existing} "
        f"skipped_missing_inputs={skipped_missing_inputs} skipped_bad_inputs={skipped_bad_inputs}",
        flush=True,
    )


if __name__ == "__main__":
    main()
