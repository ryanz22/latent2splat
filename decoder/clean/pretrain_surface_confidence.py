"""Direct supervised pretraining for the source-splat confidence head.

This trains the same ``SurfaceConfidenceUNet`` used by ``train_phase2.py`` on
per-conditioning-view agreement between predicted conditioning depth and
Blender GT depth. It is a cheap pre-raster stage: the saved auxiliary checkpoint
can be evaluated in the 3DGS renderer with ``--surface_confidence_unet 1`` and
``--resume_aux_from``.
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

from decoder.clean.pretrain_depth_refine import _load_manifest, _load_view_batch
from decoder.clean.train_phase2 import SurfaceConfidenceUNet


def _features(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["mask"].clamp(0.0, 1.0)
    feat = torch.cat([
        batch["rgb"].clamp(0.0, 1.0) * mask,
        mask,
        batch["prior_frac"][:, None],
        batch["prior_valid"][:, None],
        batch["mv_features"].to(device=mask.device, dtype=mask.dtype),
    ], dim=1)
    valid = (
        (batch["prior_valid"] > 0.5)
        & (batch["target_valid"] > 0.5)
        & (mask[:, 0] > 0.5)
    )
    return feat, valid


def _target(batch: dict, valid: torch.Tensor, frac_tol: float) -> torch.Tensor:
    err = (batch["prior_frac"] - batch["target_frac"]).abs()
    tol = max(float(frac_tol), 1e-6)
    target = torch.exp(-0.5 * (err / tol).clamp(0.0, 12.0).square())
    return torch.where(valid, target, target.new_zeros(()))


def _forward(head: SurfaceConfidenceUNet, batch: dict,
             init: float, delta_scale: float,
             frac_tol: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feat, valid = _features(batch)
    raw = head(feat)
    delta = max(float(delta_scale), 0.0) * torch.tanh(raw)
    init_c = min(max(float(init), 1e-4), 1.0 - 1e-4)
    prior = math.log(init_c / (1.0 - init_c))
    pred = torch.sigmoid(raw.new_tensor(prior) + delta)[:, 0]
    target = _target(batch, valid, frac_tol)
    return pred, target, delta[:, 0], valid


@torch.no_grad()
def _eval_head(head: SurfaceConfidenceUNet, entries: list,
               args: argparse.Namespace, rng: random.Random,
               device: torch.device) -> dict:
    head.eval()
    rows = []
    for entry in entries[: max(args.eval_objects, 0)]:
        try:
            batch = _load_view_batch(
                Path(args.dataset_root),
                entry,
                args.cond_subdir,
                args.cond_depth_subdir,
                args.cond_view_indices,
                args.eval_views_per_object,
                rng,
                device,
                args.train_scale,
                args.half_frac,
                True,
                args.multiview_refs,
                args.multiview_tol_frac,
                args.multiview_radius_px,
            )
        except Exception as ex:
            print(f"[surface_pretrain] eval skip {entry!r}: {type(ex).__name__}: {ex}", flush=True)
            continue
        pred, target, delta, valid = _forward(
            head, batch, args.init, args.delta_scale, args.target_frac_tol
        )
        if not valid.any():
            continue
        pv = pred[valid]
        tv = target[valid]
        rows.append({
            "bce": float(F.binary_cross_entropy(pv.clamp(1e-4, 1.0 - 1e-4), tv).item()),
            "mae": float((pv - tv).abs().mean().item()),
            "pred_mean": float(pv.mean().item()),
            "target_mean": float(tv.mean().item()),
            "delta_abs": float((delta.abs() * batch["prior_valid"]).sum().item()
                               / batch["prior_valid"].sum().clamp_min(1.0).item()),
        })
    head.train()
    if not rows:
        return {"n": 0, "bce": math.nan, "mae": math.nan,
                "pred_mean": math.nan, "target_mean": math.nan, "delta_abs": math.nan}
    return {
        "n": len(rows),
        "bce": sum(r["bce"] for r in rows) / len(rows),
        "mae": sum(r["mae"] for r in rows) / len(rows),
        "pred_mean": sum(r["pred_mean"] for r in rows) / len(rows),
        "target_mean": sum(r["target_mean"] for r in rows) / len(rows),
        "delta_abs": sum(r["delta_abs"] for r in rows) / len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=os.environ.get("PHASE2_DATA_ROOT", ""))
    ap.add_argument("--manifest", default=os.environ.get("PHASE2_MANIFEST", "manifest.json"))
    ap.add_argument("--cond_subdir", default="ltx_decoded")
    ap.add_argument("--cond_depth_subdir", default="da3_ltx")
    ap.add_argument("--cond_view_indices", default="available")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--views_per_step", type=int, default=3)
    ap.add_argument("--eval_views_per_object", type=int, default=4)
    ap.add_argument("--eval_objects", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_at_start", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--init", type=float, default=0.995)
    ap.add_argument("--delta_scale", type=float, default=6.0)
    ap.add_argument("--target_frac_tol", type=float, default=0.015)
    ap.add_argument("--train_scale", type=float, default=0.5)
    ap.add_argument("--half_frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--outlier_weight", type=float, default=4.0)
    ap.add_argument("--outlier_power", type=float, default=1.0)
    ap.add_argument("--positive_weight", type=float, default=1.0)
    ap.add_argument("--negative_weight", type=float, default=2.0)
    ap.add_argument("--prior_weight", type=float, default=1e-4)
    ap.add_argument("--tv_weight", type=float, default=1e-4)
    ap.add_argument("--multiview_refs", type=int, default=4)
    ap.add_argument("--multiview_tol_frac", type=float, default=0.02)
    ap.add_argument("--multiview_radius_px", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/surface_confidence_pretrain")
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
    head = SurfaceConfidenceUNet(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = out / "surface_confidence_pretrain_metrics.jsonl"
    best = math.inf
    best_metrics = None
    best_step = -1
    t0 = time.time()

    def save_ckpt(path: Path, step: int, metrics: dict | None = None) -> None:
        torch.save({
            "step": step,
            "args": vars(args),
            "metrics": metrics,
            "surface_confidence_head": head.state_dict(),
        }, path)

    def run_eval(step: int) -> dict:
        nonlocal best, best_metrics, best_step
        ev = _eval_head(head, heldout_entries, args, random.Random(args.seed + 999), device)
        with metrics_path.open("a") as f:
            f.write(json.dumps({"step": step, **ev}, sort_keys=True) + "\n")
        print(
            f"[surface_pretrain] EVAL step {step} n={ev['n']} "
            f"bce={ev['bce']:.5f} mae={ev['mae']:.5f} "
            f"pred={ev['pred_mean']:.3f} tgt={ev['target_mean']:.3f} "
            f"delta={ev['delta_abs']:.5f}",
            flush=True,
        )
        if ev["n"] > 0 and math.isfinite(ev["mae"]) and ev["mae"] < best:
            best = ev["mae"]
            best_metrics = ev
            best_step = step
            save_ckpt(out / "surface_confidence_aux_best.pt", step, ev)
        return ev

    print(
        f"[surface_pretrain] train={len(train_entries)} heldout={len(heldout_entries)} "
        f"views/step={args.views_per_step} scale={args.train_scale} device={device}",
        flush=True,
    )
    if args.eval_at_start:
        run_eval(-1)
    for step in range(max(args.steps, 1)):
        entry = train_entries[rng.randrange(len(train_entries))]
        try:
            batch = _load_view_batch(
                root,
                entry,
                args.cond_subdir,
                args.cond_depth_subdir,
                args.cond_view_indices,
                args.views_per_step,
                rng,
                device,
                args.train_scale,
                args.half_frac,
                True,
                args.multiview_refs,
                args.multiview_tol_frac,
                args.multiview_radius_px,
            )
        except Exception as ex:
            print(f"[surface_pretrain] train skip {entry!r}: {type(ex).__name__}: {ex}", flush=True)
            continue
        pred, target, delta, valid = _forward(
            head, batch, args.init, args.delta_scale, args.target_frac_tol
        )
        if not valid.any():
            continue
        pv = pred[valid].clamp(1e-4, 1.0 - 1e-4)
        tv = target[valid]
        loss_px = F.binary_cross_entropy(pv, tv, reduction="none")
        prior_err = (batch["prior_frac"][valid] - batch["target_frac"][valid]).abs().detach()
        hard_w = 1.0 + args.outlier_weight * (
            prior_err / max(args.target_frac_tol, 1e-6)
        ).clamp(0.0, 20.0).pow(max(args.outlier_power, 1e-6))
        cls_w = tv * max(args.positive_weight, 0.0) + (1.0 - tv) * max(args.negative_weight, 0.0)
        weight = (hard_w * cls_w).clamp_min(1e-6)
        loss = (loss_px * weight).sum() / weight.sum().clamp_min(1e-6)
        if args.prior_weight > 0:
            loss = loss + args.prior_weight * (
                (delta.square() * batch["prior_valid"]).sum()
                / batch["prior_valid"].sum().clamp_min(1.0)
            )
        if args.tv_weight > 0 and delta.shape[-1] > 1 and delta.shape[-2] > 1:
            valid_x = batch["prior_valid"][:, :, 1:] * batch["prior_valid"][:, :, :-1]
            valid_y = batch["prior_valid"][:, 1:, :] * batch["prior_valid"][:, :-1, :]
            tv_x = ((delta[:, :, 1:] - delta[:, :, :-1]).abs() * valid_x).sum() / valid_x.sum().clamp_min(1.0)
            tv_y = ((delta[:, 1:, :] - delta[:, :-1, :]).abs() * valid_y).sum() / valid_y.sum().clamp_min(1.0)
            loss = loss + args.tv_weight * 0.5 * (tv_x + tv_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(args.log_every, 1) == 0:
            print(
                f"[surface_pretrain] step {step} loss={loss.item():.5f} "
                f"mae={(pv.detach() - tv).abs().mean().item():.5f} "
                f"pred={pv.detach().mean().item():.3f} tgt={tv.mean().item():.3f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        if step % max(args.eval_every, 1) == 0 or step == args.steps - 1:
            run_eval(step)

    save_ckpt(out / "surface_confidence_aux.pt", max(args.steps - 1, 0), best_metrics)
    print(f"[surface_pretrain] saved {out / 'surface_confidence_aux.pt'}", flush=True)
    if best_metrics is not None:
        print(
            f"[surface_pretrain] best saved {out / 'surface_confidence_aux_best.pt'} "
            f"step={best_step} mae={best_metrics['mae']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
