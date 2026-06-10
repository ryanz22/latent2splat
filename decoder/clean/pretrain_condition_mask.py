"""Direct supervised pretraining for latent-decoded conditioning masks.

This trains a shared feed-forward refiner:

    LTX decoded RGB + RGB-derived prior mask -> foreground mask

The saved auxiliary head can be loaded by train_phase2.py with
``--condition_mask_refine_unet 1 --condition_mask_source rgb_border
--resume_aux_from <ckpt>``. This is not per-object optimization: the same small
U-Net is applied to every decoded conditioning frame at inference time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from decoder.clean.condition_refine import (
    ConditionMaskRefineUNet,
    apply_mask_refiner,
    rgb_border_mask,
)
from decoder.clean.phase2_data import load_masks_at, load_views_at, resolve_view_spec
from decoder.data import entry_uid, load_cameras, object_dir_for_entry


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _resize(
    cond: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale >= 0.999:
        return cond, mask
    h, w = cond.shape[1:3]
    out_size = (max(8, int(round(h * scale))), max(8, int(round(w * scale))))
    cond_chw = F.interpolate(
        cond.permute(0, 3, 1, 2), size=out_size, mode="bilinear", align_corners=False
    )
    mask_chw = F.interpolate(
        mask.permute(0, 3, 1, 2), size=out_size, mode="area"
    ).clamp(0.0, 1.0)
    return cond_chw.permute(0, 2, 3, 1), mask_chw.permute(0, 2, 3, 1)


def _load_view_batch(
    root: Path,
    entry,
    cond_subdir: str,
    view_spec: str,
    views_per_step: int,
    rng: random.Random,
    device: torch.device,
    train_scale: float,
) -> dict:
    uid = entry_uid(entry)
    obj_dir = object_dir_for_entry(root, entry)
    cams = load_cameras(obj_dir / "cameras.json")
    idxs_all = resolve_view_spec(
        view_spec,
        cams["w2c"].shape[0],
        obj_dir=obj_dir,
        subdir=cond_subdir,
        n_orbit_views=cams["num_orbit_views"],
        default_n=None,
    )
    if not idxs_all:
        raise RuntimeError(f"no conditioning views for {uid}")
    n = min(max(int(views_per_step), 1), len(idxs_all))
    idxs = rng.sample(idxs_all, n) if len(idxs_all) > n else list(idxs_all)
    idxs.sort()
    cond = load_views_at(obj_dir, idxs, subdir=cond_subdir).to(device=device)
    mask = load_masks_at(obj_dir, idxs).to(device=device)
    cond, mask = _resize(cond, mask, train_scale)
    return {"uid": uid, "idxs": idxs, "cond": cond, "mask": mask}


def _mask_loss(pred: torch.Tensor, target: torch.Tensor, fg_weight: float) -> torch.Tensor:
    pred = pred.clamp(1e-4, 1.0 - 1e-4)
    target = target.clamp(0.0, 1.0)
    w = 1.0 + max(float(fg_weight), 0.0) * target
    bce = F.binary_cross_entropy(pred, target, weight=w, reduction="sum") / w.sum().clamp_min(1.0)
    inter = (pred * target).sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (pred.sum() + target.sum() + 1.0)
    return bce + dice


def _mask_metrics(pred: torch.Tensor, target: torch.Tensor, prior: torch.Tensor) -> dict:
    p = pred > 0.5
    t = target > 0.5
    pr = prior > 0.5

    def stats(x: torch.Tensor) -> tuple[float, float, float]:
        inter = (x & t).float().sum()
        union = (x | t).float().sum().clamp_min(1.0)
        iou = inter / union
        fp = (x & ~t).float().sum() / (~t).float().sum().clamp_min(1.0)
        fn = (~x & t).float().sum() / t.float().sum().clamp_min(1.0)
        return float(iou.item()), float(fp.item()), float(fn.item())

    piou, pfp, pfn = stats(pr)
    riou, rfp, rfn = stats(p)
    return {
        "prior_iou": piou,
        "refined_iou": riou,
        "prior_fp": pfp,
        "refined_fp": rfp,
        "prior_fn": pfn,
        "refined_fn": rfn,
        "prior_l1": float((prior - target).abs().mean().item()),
        "refined_l1": float((pred - target).abs().mean().item()),
    }


@torch.no_grad()
def _eval_head(
    head: ConditionMaskRefineUNet,
    entries: list,
    args: argparse.Namespace,
    rng: random.Random,
    device: torch.device,
) -> dict:
    head.eval()
    rows = []
    for entry in entries[: max(args.eval_objects, 0)]:
        try:
            batch = _load_view_batch(
                Path(args.dataset_root), entry, args.cond_subdir,
                args.cond_view_indices, args.eval_views_per_object, rng,
                device, args.train_scale,
            )
        except Exception as ex:
            print(f"[mask_pretrain] eval skip {entry!r}: {type(ex).__name__}: {ex}", flush=True)
            continue
        prior = rgb_border_mask(
            batch["cond"], threshold=args.prior_threshold, softness=args.prior_softness
        )
        refined = apply_mask_refiner(head, batch["cond"], prior, args.residual_scale)
        rows.append(_mask_metrics(refined, batch["mask"], prior))
    head.train()
    if not rows:
        return {
            "prior_iou": math.nan,
            "refined_iou": math.nan,
            "prior_fp": math.nan,
            "refined_fp": math.nan,
            "prior_fn": math.nan,
            "refined_fn": math.nan,
            "prior_l1": math.nan,
            "refined_l1": math.nan,
            "n": 0,
        }
    out = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    out["n"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=os.environ.get("PHASE2_DATA_ROOT", ""))
    ap.add_argument("--manifest", default=os.environ.get("PHASE2_MANIFEST", "manifest.json"))
    ap.add_argument("--cond_subdir", default="ltx_decoded")
    ap.add_argument("--cond_view_indices", default="available")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--views_per_step", type=int, default=4)
    ap.add_argument("--eval_views_per_object", type=int, default=8)
    ap.add_argument("--eval_objects", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_at_start", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--residual_scale", type=float, default=8.0)
    ap.add_argument("--prior_threshold", type=float, default=0.12)
    ap.add_argument("--prior_softness", type=float, default=0.02)
    ap.add_argument("--train_scale", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--fg_weight", type=float, default=4.0)
    ap.add_argument("--prior_weight", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/condition_mask_pretrain")
    args = ap.parse_args()

    if not args.dataset_root:
        raise ValueError("--dataset_root or PHASE2_DATA_ROOT is required")
    root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = _load_manifest(manifest_path)
    train_entries = list(manifest["train"])
    heldout_entries = list(manifest.get("eval", [])) + list(manifest.get("test", []))
    if not heldout_entries:
        heldout_entries = train_entries[: args.eval_objects]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = ConditionMaskRefineUNet(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = out / "condition_mask_pretrain_metrics.jsonl"
    t0 = time.time()
    best_iou = -1.0
    best_step = -1
    best_metrics = None

    def save_ckpt(path: Path, step: int, metrics: dict | None = None) -> None:
        torch.save(
            {
                "step": step,
                "args": vars(args),
                "metrics": metrics,
                "condition_mask_refine_head": head.state_dict(),
            },
            path,
        )

    def run_eval(step: int) -> dict:
        nonlocal best_iou, best_step, best_metrics
        ev = _eval_head(head, heldout_entries, args, random.Random(args.seed + 999), device)
        row = {"step": step, **ev}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[mask_pretrain] EVAL step {step} n={ev['n']} "
            f"prior_iou={ev['prior_iou']:.3f} refined_iou={ev['refined_iou']:.3f} "
            f"prior_fp={ev['prior_fp']:.4f} refined_fp={ev['refined_fp']:.4f} "
            f"prior_fn={ev['prior_fn']:.4f} refined_fn={ev['refined_fn']:.4f}",
            flush=True,
        )
        if ev["n"] > 0 and math.isfinite(ev["refined_iou"]) and ev["refined_iou"] > best_iou:
            best_iou = ev["refined_iou"]
            best_step = step
            best_metrics = ev
            save_ckpt(out / "condition_mask_refine_aux_best.pt", step, ev)
            print(f"[mask_pretrain] best step {best_step} refined_iou={best_iou:.3f}", flush=True)
        return ev

    print(
        f"[mask_pretrain] train={len(train_entries)} heldout={len(heldout_entries)} "
        f"views/step={args.views_per_step} scale={args.train_scale} device={device}",
        flush=True,
    )
    if args.eval_at_start:
        run_eval(-1)
    for step in range(max(args.steps, 1)):
        entry = train_entries[rng.randrange(len(train_entries))]
        try:
            batch = _load_view_batch(
                root, entry, args.cond_subdir, args.cond_view_indices,
                args.views_per_step, rng, device, args.train_scale,
            )
        except Exception as ex:
            print(f"[mask_pretrain] train skip {entry!r}: {type(ex).__name__}: {ex}", flush=True)
            continue
        prior = rgb_border_mask(
            batch["cond"], threshold=args.prior_threshold, softness=args.prior_softness
        )
        refined = apply_mask_refiner(head, batch["cond"], prior, args.residual_scale)
        loss = _mask_loss(refined, batch["mask"], args.fg_weight)
        if args.prior_weight > 0:
            loss = loss + args.prior_weight * (refined - prior).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(args.log_every, 1) == 0:
            m = _mask_metrics(refined.detach(), batch["mask"], prior)
            print(
                f"[mask_pretrain] step {step} loss={loss.item():.5f} "
                f"prior_iou={m['prior_iou']:.3f} refined_iou={m['refined_iou']:.3f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        if step % max(args.eval_every, 1) == 0 or step == args.steps - 1:
            run_eval(step)

    save_ckpt(out / "condition_mask_refine_aux.pt", max(args.steps - 1, 0), best_metrics)
    print(f"[mask_pretrain] saved {out / 'condition_mask_refine_aux.pt'}", flush=True)
    if best_step >= -1 and best_metrics is not None:
        print(
            f"[mask_pretrain] best saved {out / 'condition_mask_refine_aux_best.pt'} "
            f"step={best_step} refined_iou={best_iou:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
