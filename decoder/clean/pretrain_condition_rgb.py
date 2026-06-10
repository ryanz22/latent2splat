"""Direct supervised pretraining for latent-decoded RGB conditioning frames.

This trains a shared feed-forward refiner:

    LTX decoded RGB + mask -> Blender source RGB

The saved auxiliary head can be loaded by train_phase2.py with
``--condition_rgb_refine_unet 1 --resume_aux_from <ckpt>``. This is not
per-object optimization: the same small U-Net is applied to every decoded
conditioning frame at inference time.
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

from decoder.clean.condition_refine import ConditionRGBRefineUNet, apply_rgb_refiner
from decoder.clean.phase2_data import load_masks_at, load_views_at, resolve_view_spec
from decoder.data import entry_uid, load_cameras, object_dir_for_entry


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _resize(
    cond: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale >= 0.999:
        return cond, target, mask
    h, w = cond.shape[1:3]
    out_size = (max(8, int(round(h * scale))), max(8, int(round(w * scale))))
    cond_chw = F.interpolate(cond.permute(0, 3, 1, 2), size=out_size,
                             mode="bilinear", align_corners=False)
    tgt_chw = F.interpolate(target.permute(0, 3, 1, 2), size=out_size,
                            mode="bilinear", align_corners=False)
    mask_chw = F.interpolate(mask.permute(0, 3, 1, 2), size=out_size,
                             mode="area").clamp(0.0, 1.0)
    return cond_chw.permute(0, 2, 3, 1), tgt_chw.permute(0, 2, 3, 1), mask_chw.permute(0, 2, 3, 1)


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
    target = load_views_at(obj_dir, idxs, subdir=None).to(device=device)
    mask = load_masks_at(obj_dir, idxs).to(device=device)
    cond, target, mask = _resize(cond, target, mask, train_scale)
    return {"uid": uid, "idxs": idxs, "cond": cond, "target": target, "mask": mask}


def _has_condition_views(root: Path, entry, cond_subdir: str,
                         view_spec: str) -> bool:
    """Return whether an entry has the requested decoded conditioning frames."""
    try:
        obj_dir = object_dir_for_entry(root, entry)
        cams = load_cameras(obj_dir / "cameras.json")
        idxs = resolve_view_spec(
            view_spec,
            cams["w2c"].shape[0],
            obj_dir=obj_dir,
            subdir=cond_subdir,
            n_orbit_views=cams["num_orbit_views"],
            default_n=None,
        )
        return bool(idxs)
    except Exception:
        return False


def _filter_entries_with_condition(root: Path, entries: list, cond_subdir: str,
                                   view_spec: str) -> list:
    return [
        entry for entry in entries
        if _has_condition_views(root, entry, cond_subdir, view_spec)
    ]


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.expand_as(pred)
    return ((pred - target).abs() * w).sum() / w.sum().clamp_min(1.0)


def _masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    w = mask.expand_as(pred)
    mse = (((pred - target) ** 2) * w).sum() / w.sum().clamp_min(1.0)
    return float((-10.0 * torch.log10(mse.clamp_min(1e-8))).item())


def _grad_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = pred.permute(0, 3, 1, 2)
    t = target.permute(0, 3, 1, 2)
    m = mask.permute(0, 3, 1, 2)
    gx_p = p[..., :, 1:] - p[..., :, :-1]
    gx_t = t[..., :, 1:] - t[..., :, :-1]
    gy_p = p[..., 1:, :] - p[..., :-1, :]
    gy_t = t[..., 1:, :] - t[..., :-1, :]
    mx = (m[..., :, 1:] * m[..., :, :-1]).expand_as(gx_p)
    my = (m[..., 1:, :] * m[..., :-1, :]).expand_as(gy_p)
    lx = ((gx_p - gx_t).abs() * mx).sum() / mx.sum().clamp_min(1.0)
    ly = ((gy_p - gy_t).abs() * my).sum() / my.sum().clamp_min(1.0)
    return 0.5 * (lx + ly)


@torch.no_grad()
def _eval_head(
    head: ConditionRGBRefineUNet,
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
            print(f"[rgb_pretrain] eval skip {entry!r}: {type(ex).__name__}: {ex}", flush=True)
            continue
        refined = apply_rgb_refiner(head, batch["cond"], batch["mask"], args.residual_scale)
        rows.append({
            "prior_l1": float(_masked_l1(batch["cond"], batch["target"], batch["mask"]).item()),
            "refined_l1": float(_masked_l1(refined, batch["target"], batch["mask"]).item()),
            "prior_psnr": _masked_psnr(batch["cond"], batch["target"], batch["mask"]),
            "refined_psnr": _masked_psnr(refined, batch["target"], batch["mask"]),
        })
    head.train()
    if not rows:
        return {
            "prior_l1": math.nan,
            "refined_l1": math.nan,
            "prior_psnr": math.nan,
            "refined_psnr": math.nan,
            "n": 0,
        }
    return {
        "prior_l1": sum(r["prior_l1"] for r in rows) / len(rows),
        "refined_l1": sum(r["refined_l1"] for r in rows) / len(rows),
        "prior_psnr": sum(r["prior_psnr"] for r in rows) / len(rows),
        "refined_psnr": sum(r["refined_psnr"] for r in rows) / len(rows),
        "n": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=os.environ.get("PHASE2_DATA_ROOT", ""))
    ap.add_argument("--manifest", default=os.environ.get("PHASE2_MANIFEST", "manifest.json"))
    ap.add_argument("--cond_subdir", default="ltx_decoded")
    ap.add_argument("--cond_view_indices", default="available")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--views_per_step", type=int, default=2)
    ap.add_argument("--eval_views_per_object", type=int, default=4)
    ap.add_argument("--eval_objects", type=int, default=4)
    ap.add_argument("--max_train_objects", type=int, default=0)
    ap.add_argument("--max_heldout_objects", type=int, default=0)
    ap.add_argument("--filter_missing_condition", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_at_start", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--residual_scale", type=float, default=0.15)
    ap.add_argument("--train_scale", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_weight", type=float, default=0.1)
    ap.add_argument("--delta_weight", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/condition_rgb_pretrain")
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
    if args.max_train_objects > 0:
        train_entries = train_entries[: args.max_train_objects]
    if args.max_heldout_objects > 0:
        heldout_entries = heldout_entries[: args.max_heldout_objects]
    if args.filter_missing_condition:
        before_train = len(train_entries)
        before_heldout = len(heldout_entries)
        train_entries = _filter_entries_with_condition(
            root, train_entries, args.cond_subdir, args.cond_view_indices
        )
        heldout_entries = _filter_entries_with_condition(
            root, heldout_entries, args.cond_subdir, args.cond_view_indices
        )
        print(
            f"[rgb_pretrain] filtered missing condition "
            f"train {before_train}->{len(train_entries)} "
            f"heldout {before_heldout}->{len(heldout_entries)}",
            flush=True,
        )
    if not train_entries:
        raise ValueError("no train entries remain after filtering")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = ConditionRGBRefineUNet(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = out / "condition_rgb_pretrain_metrics.jsonl"
    t0 = time.time()
    best_score = math.inf
    best_step = -1
    best_metrics = None

    def save_ckpt(path: Path, step: int, metrics: dict | None = None) -> None:
        ckpt = {
            "step": step,
            "args": vars(args),
            "metrics": metrics,
            "condition_rgb_refine_head": head.state_dict(),
        }
        torch.save(ckpt, path)

    def run_eval(step: int) -> dict:
        nonlocal best_score, best_step, best_metrics
        ev = _eval_head(head, heldout_entries, args, random.Random(args.seed + 999), device)
        row = {"step": step, **ev}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[rgb_pretrain] EVAL step {step} n={ev['n']} "
            f"prior_l1={ev['prior_l1']:.5f} refined_l1={ev['refined_l1']:.5f} "
            f"prior_psnr={ev['prior_psnr']:.2f} refined_psnr={ev['refined_psnr']:.2f}",
            flush=True,
        )
        if ev["n"] > 0 and math.isfinite(ev["refined_l1"]) and ev["refined_l1"] < best_score:
            best_score = ev["refined_l1"]
            best_step = step
            best_metrics = ev
            save_ckpt(out / "condition_rgb_refine_aux_best.pt", step, ev)
            print(
                f"[rgb_pretrain] best step {best_step} "
                f"refined_l1={best_score:.5f} refined_psnr={ev['refined_psnr']:.2f}",
                flush=True,
            )
        return ev

    print(
        f"[rgb_pretrain] train={len(train_entries)} heldout={len(heldout_entries)} "
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
            print(f"[rgb_pretrain] train skip {entry!r}: {type(ex).__name__}: {ex}", flush=True)
            continue
        refined = apply_rgb_refiner(head, batch["cond"], batch["mask"], args.residual_scale)
        loss = _masked_l1(refined, batch["target"], batch["mask"])
        if args.grad_weight > 0:
            loss = loss + args.grad_weight * _grad_l1(refined, batch["target"], batch["mask"])
        if args.delta_weight > 0:
            delta = (refined - batch["cond"]) * batch["mask"]
            loss = loss + args.delta_weight * (
                delta.square().sum() / batch["mask"].sum().clamp_min(1.0)
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(args.log_every, 1) == 0:
            prior_l1 = _masked_l1(batch["cond"], batch["target"], batch["mask"]).item()
            refined_l1 = _masked_l1(refined.detach(), batch["target"], batch["mask"]).item()
            print(
                f"[rgb_pretrain] step {step} loss={loss.item():.5f} "
                f"prior_l1={prior_l1:.5f} refined_l1={refined_l1:.5f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        if step % max(args.eval_every, 1) == 0 or step == args.steps - 1:
            run_eval(step)

    save_ckpt(out / "condition_rgb_refine_aux.pt", max(args.steps - 1, 0), best_metrics)
    print(f"[rgb_pretrain] saved {out / 'condition_rgb_refine_aux.pt'}", flush=True)
    if best_step >= -1 and best_metrics is not None:
        print(
            f"[rgb_pretrain] best saved {out / 'condition_rgb_refine_aux_best.pt'} "
            f"step={best_step} refined_l1={best_metrics['refined_l1']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
