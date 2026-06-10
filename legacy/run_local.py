"""Local runner for the 3DGS decoder — calls legacy/train.py directly, no Modal.

Use this on a local CUDA box (e.g. an RTX 5090) instead of the Modal harness.
The `decoder/` package is pure PyTorch + gsplat and runs as-is; this script
just replaces the Modal wrappers (overfit / freegs / probe / visualize) with a
plain argparse CLI.

Examples:
  # overfit one object with Method E (current arch)
  python legacy/run_local.py overfit --arch e --steps 400 --log-every 20 \
      --dataset ../data/Animals\\ v1/animals_v1 \
      --manifest ../data/Animals\\ v1/manifests/animals_v1_encoded.json \
      --uid af3281f986cc40b9b3cbca1f72e77f46 --bg 0.0 --out runs/

  # decoder-free baseline
  python legacy/run_local.py freegs --steps 1000 --dataset ... --manifest ... --out runs/

  # render the saved checkpoint to a target|render|alpha grid
  python legacy/run_local.py viz --ckpt runs/overfit_e.pt --out runs/

Dataset path notes: pass the directory that holds the per-object {uid}/ folders
as --dataset, and the manifest json (keys train/eval/test) as --manifest. For
animals_v1 the manifest lives in a sibling manifests/ dir, hence two args.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _viz(ckpt_path: str, out_dir: str, views=(0, 12, 24, 36)) -> None:
    ck = torch.load(ckpt_path, map_location="cpu")
    render, target = ck["render"], ck["target"]
    alpha = ck.get("alpha")
    res = ck["result"]
    print(f"[viz] {res.get('uid')} last_l1={res['last_l1']:.4f} "
          f"final_psnr={res['final_psnr']:.2f} V={render.shape[0]}")
    views = [v for v in views if v < render.shape[0]]

    def im(t):
        return (t.clamp(0, 1).numpy() * 255).astype(np.uint8)

    rows = []
    for v in views:
        cols = [im(target[v]), im(render[v])]
        if alpha is not None:
            a = alpha[v, ..., 0].clamp(0, 1).numpy()
            cols.append((np.stack([a, a, a], -1) * 255).astype(np.uint8))
        rows.append(np.concatenate(cols, 1))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = Path(ckpt_path).stem
    p = Path(out_dir) / f"viz_{name}.png"
    Image.fromarray(np.concatenate(rows, 0)).save(p)
    print(f"[viz] wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", required=True, help="dir holding {uid}/ folders")
    common.add_argument("--manifest", default=None, help="manifest json (default: dataset/manifest.json)")
    common.add_argument("--out", default="runs", help="output dir for checkpoints")
    common.add_argument("--steps", type=int, default=400)
    common.add_argument("--log-every", type=int, default=20)
    common.add_argument("--uid", default="", help="object uid (default: first train entry)")
    common.add_argument("--bg", type=float, default=0.0, help="dataset bg gray level (animals=0 black)")
    common.add_argument("--device", default="cuda")

    p_of = sub.add_parser("overfit", parents=[common])
    p_of.add_argument("--arch", default="e", choices=["a", "b", "c", "d", "e"])
    p_of.add_argument("--lr", type=float, default=1e-4)
    p_of.add_argument("--fg-weight", type=float, default=10.0)
    p_of.add_argument("--mask-weight", type=float, default=0.0)
    p_of.add_argument("--scale-reg", type=float, default=0.0)
    p_of.add_argument("--opacity-mode", default="pdf", choices=["pdf", "sigmoid"])
    p_of.add_argument("--opacity-reg", type=float, default=0.0)
    p_of.add_argument("--random-bg", action="store_true", default=True)
    p_of.add_argument("--no-random-bg", dest="random_bg", action="store_false")

    p_fg = sub.add_parser("freegs", parents=[common])
    p_fg.add_argument("--n-gaussians", type=int, default=30000)
    p_fg.add_argument("--fg-weight", type=float, default=10.0)

    p_vz = sub.add_parser("viz")
    p_vz.add_argument("--ckpt", required=True)
    p_vz.add_argument("--out", default="runs")

    args = ap.parse_args()

    if args.cmd == "viz":
        _viz(args.ckpt, args.out)
        return

    from legacy.train import overfit_one_object, overfit_freegs

    if args.cmd == "overfit":
        r = overfit_one_object(
            dataset_root=args.dataset, arch=args.arch, steps=args.steps, lr=args.lr,
            device=args.device, log_every=args.log_every, out_dir=args.out,
            manifest_path=args.manifest, bg=args.bg, mask_weight=args.mask_weight,
            fg_weight=args.fg_weight, scale_reg_weight=args.scale_reg,
            opacity_mode=args.opacity_mode, opacity_reg_weight=args.opacity_reg,
            random_bg=args.random_bg, uid=args.uid)
        print(f"[overfit-{args.arch}] DONE first_l1={r['first_l1']:.4f} "
              f"last_l1={r['last_l1']:.4f} drop={r['l1_drop_ratio']:.1f}x "
              f"final_psnr={r['final_psnr']:.2f}")
        _viz(str(Path(args.out) / f"overfit_{args.arch}.pt"), args.out)
    elif args.cmd == "freegs":
        r = overfit_freegs(
            dataset_root=args.dataset, n_gaussians=args.n_gaussians, steps=args.steps,
            device=args.device, log_every=args.log_every, out_dir=args.out,
            manifest_path=args.manifest, bg=args.bg, fg_weight=args.fg_weight,
            uid=args.uid)
        print(f"[freegs] DONE final_psnr={r['final_psnr']:.2f}")
        _viz(str(Path(args.out) / "overfit_freegs.pt"), args.out)


if __name__ == "__main__":
    main()
