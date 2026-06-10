"""Direct supervised pretraining for the joint RGBD conditioning refiner.

This trains a shared feed-forward head:

    LTX decoded RGB + estimated depth -> Blender RGB + GT depth

The saved auxiliary head can be loaded by train_phase2.py with
``--condition_rgbd_refine_unet 1 --resume_aux_from <ckpt>``. It is still a
single set of shared weights, not per-object optimization.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from decoder.clean.condition_refine import (
    ConditionRGBDRefineUNet,
    ConditionRGBDViewRefineUNet,
    apply_rgbd_refiner,
    rgb_border_mask,
)
from decoder.clean.geometry import depth_bounds
from decoder.clean.phase2_data import (
    depth_path_at,
    frame_path_at,
    load_depth_view_at,
    load_masks_at,
    load_views_at,
    resolve_view_spec,
)
from decoder.clean.train_phase2 import _depth_multiview_support_maps, _erode_mask_2d
from decoder.data import (
    entry_uid,
    load_cameras,
    load_depth_view,
    object_dir_for_entry,
    zdepth_to_raydist,
)


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _ids_with_prefix(base: Path, prefix: str, suffix: str) -> set[int]:
    found = set()
    if not base.exists():
        return found
    pat = re.compile(rf"{re.escape(prefix)}_(\d+){re.escape(suffix)}$")
    for p in base.iterdir():
        m = pat.fullmatch(p.name)
        if m:
            found.add(int(m.group(1)))
    return found


def _filter_entries_with_sidecars(
    root: Path,
    entries: list,
    cond_subdir: str,
    cond_depth_subdir: str,
    view_spec: str,
    min_views: int,
) -> list:
    out = []
    min_views = max(int(min_views), 1)
    for entry in entries:
        try:
            obj_dir = object_dir_for_entry(root, entry)
            if not (obj_dir / "cameras.json").exists():
                continue
            frame_base = frame_path_at(obj_dir, 0, subdir=cond_subdir).parent
            depth_base = depth_path_at(obj_dir, 0, subdir=cond_depth_subdir).parent
            available = _ids_with_prefix(frame_base, "frame", ".png") & _ids_with_prefix(
                depth_base, "depth", ".npy"
            )
            if len(available) < min_views:
                continue
            if view_spec == "available":
                idxs = sorted(available)
            else:
                cams = load_cameras(obj_dir / "cameras.json")
                idxs = resolve_view_spec(
                    view_spec,
                    cams["w2c"].shape[0],
                    obj_dir=obj_dir,
                    subdir=cond_subdir,
                    n_orbit_views=cams["num_orbit_views"],
                    default_n=None,
                )
            if idxs and sum(1 for i in idxs if i in available) >= min_views:
                out.append(entry)
        except Exception:
            continue
    return out


def _apply_mask_source(frames: torch.Tensor, gt_masks: torch.Tensor,
                       args: argparse.Namespace) -> torch.Tensor:
    if args.mask_source == "gt":
        out = gt_masks
    elif args.mask_source == "rgb_border":
        out = rgb_border_mask(
            frames,
            threshold=args.mask_rgb_threshold,
            softness=args.mask_rgb_softness,
        )
    elif args.mask_source == "rgb_white":
        score = torch.linalg.vector_norm(frames.clamp(0.0, 1.0) - 1.0, dim=-1, keepdim=True)
        score = score / math.sqrt(3.0)
        if args.mask_rgb_softness > 0:
            out = torch.sigmoid(
                (score - args.mask_rgb_threshold) / max(args.mask_rgb_softness, 1e-6)
            )
        else:
            out = (score > args.mask_rgb_threshold).to(frames.dtype)
    else:
        raise ValueError(f"unknown mask_source={args.mask_source}")
    x = out.permute(0, 3, 1, 2)
    if args.mask_rgb_dilate_px > 0:
        r = int(args.mask_rgb_dilate_px)
        x = F.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r)
    if args.mask_rgb_erode_px > 0:
        r = int(args.mask_rgb_erode_px)
        x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=2 * r + 1, stride=1, padding=r)
    return x.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _depth_frac_valid(
    depths: torch.Tensor,
    masks: torch.Tensor,
    k_all: torch.Tensor,
    c2w_all: torch.Tensor,
    radius: float,
    half_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fracs = []
    valids = []
    for i in range(depths.shape[0]):
        z = depths[i]
        valid = (
            torch.isfinite(z)
            & (z > 1e-6)
            & (z < 1e5)
            & (masks[i, ..., 0] > 0.5)
        )
        t = zdepth_to_raydist(z, k_all[i].to(device=z.device, dtype=z.dtype))
        d_near, d_far = depth_bounds(c2w_all[i], radius, half_frac)
        frac = ((t - d_near) / max(d_far - d_near, 1e-6)).clamp(1e-4, 1.0 - 1e-4)
        fracs.append(torch.where(valid, frac, frac.new_full(frac.shape, 1e-4)))
        valids.append(valid.to(dtype=z.dtype))
    return torch.stack(fracs), torch.stack(valids)


def _resize_batch(batch: dict, scale: float) -> dict:
    if scale >= 0.999:
        return batch
    h, w = batch["cond"].shape[1:3]
    out_size = (max(8, int(round(h * scale))), max(8, int(round(w * scale))))

    def resize_img(x: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(x.permute(0, 3, 1, 2), size=out_size,
                          mode="bilinear", align_corners=False)
        return y.permute(0, 2, 3, 1)

    def resize_mask(x: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(x.permute(0, 3, 1, 2), size=out_size, mode="area")
        return y.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    out = dict(batch)
    out["cond"] = resize_img(batch["cond"])
    out["target"] = resize_img(batch["target"])
    out["mask"] = resize_mask(batch["mask"])
    for key in ["prior_frac", "target_frac"]:
        out[key] = F.interpolate(
            batch[key][:, None], size=out_size, mode="bilinear", align_corners=False
        )[:, 0]
    for key in ["prior_valid", "target_valid"]:
        out[key] = (
            F.interpolate(batch[key][:, None], size=out_size, mode="area")[:, 0] > 0.5
        ).to(batch[key].dtype)
    if batch.get("mv_features") is not None:
        out["mv_features"] = F.interpolate(
            batch["mv_features"], size=out_size, mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
    return out


def _load_view_batch(
    root: Path,
    entry,
    args: argparse.Namespace,
    views_per_step: int,
    rng: random.Random,
    device: torch.device,
) -> dict:
    uid = entry_uid(entry)
    obj_dir = object_dir_for_entry(root, entry)
    cams = load_cameras(obj_dir / "cameras.json")
    idxs_all = resolve_view_spec(
        args.cond_view_indices,
        cams["w2c"].shape[0],
        obj_dir=obj_dir,
        subdir=args.cond_subdir,
        n_orbit_views=cams["num_orbit_views"],
        default_n=None,
    )
    if not idxs_all:
        raise RuntimeError(f"no conditioning views for {uid}")
    n = min(max(int(views_per_step), 1), len(idxs_all))
    idxs = rng.sample(idxs_all, n) if len(idxs_all) > n else list(idxs_all)
    idxs.sort()

    cond = load_views_at(obj_dir, idxs, subdir=args.cond_subdir).to(device=device)
    target = load_views_at(obj_dir, idxs, subdir=None).to(device=device)
    gt_mask = load_masks_at(obj_dir, idxs).to(device=device)
    mask = _apply_mask_source(cond, gt_mask, args)
    prior_depths = torch.stack([
        load_depth_view_at(obj_dir, i, subdir=args.cond_depth_subdir) for i in idxs
    ]).to(device=device)
    target_depths = torch.stack([load_depth_view(obj_dir, i) for i in idxs]).to(device=device)
    sel = torch.as_tensor(idxs, dtype=torch.long)
    k_all = cams["K"][sel].to(device=device)
    c2w_all = cams["c2w_opengl"][sel].to(device=device)
    radius = float(cams["radius"])
    prior_frac, prior_valid = _depth_frac_valid(
        prior_depths, mask, k_all, c2w_all, radius, args.half_frac
    )
    target_frac, target_valid = _depth_frac_valid(
        target_depths, gt_mask, k_all, c2w_all, radius, args.half_frac
    )
    mv_features = None
    if args.multiview_features:
        mv_features = _depth_multiview_support_maps(
            prior_depths,
            mask,
            k_all,
            c2w_all,
            radius,
            args.multiview_tol_frac,
            args.multiview_refs,
            args.multiview_radius_px,
        )
    batch = {
        "uid": uid,
        "idxs": idxs,
        "cond": cond,
        "target": target,
        "mask": mask,
        "prior_frac": prior_frac,
        "prior_valid": prior_valid,
        "target_frac": target_frac,
        "target_valid": target_valid,
        "mv_features": mv_features,
    }
    return _resize_batch(batch, args.train_scale)


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


def _forward_refine(head: ConditionRGBDRefineUNet, batch: dict,
                    args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor,
                                                       torch.Tensor, torch.Tensor,
                                                       torch.Tensor]:
    apply_valid = batch["prior_valid"]
    if args.apply_erode_px > 0:
        apply_valid = apply_valid * _erode_mask_2d(
            batch["mask"][..., 0].clamp(0.0, 1.0), args.apply_erode_px
        )
    refined_rgb, refined_depth, rgb_delta, depth_delta = apply_rgbd_refiner(
        head,
        batch["cond"],
        batch["mask"],
        batch["prior_frac"],
        batch["prior_valid"],
        args.rgb_residual_scale,
        args.depth_delta_scale,
        extra_features=batch.get("mv_features"),
        apply_valid=apply_valid,
    )
    valid_depth = (
        (apply_valid > 0.5)
        & (batch["target_valid"] > 0.5)
        & (batch["mask"][..., 0] > args.rgb_alpha_min)
    )
    return refined_rgb, refined_depth, rgb_delta, depth_delta[:, 0], valid_depth


@torch.no_grad()
def _eval_head(head: ConditionRGBDRefineUNet, entries: list,
               args: argparse.Namespace, rng: random.Random,
               device: torch.device) -> dict:
    head.eval()
    rows = []
    for entry in entries[: max(args.eval_objects, 0)]:
        try:
            batch = _load_view_batch(
                Path(args.dataset_root), entry, args, args.eval_views_per_object, rng, device
            )
        except Exception as ex:
            print(f"[rgbd_pretrain] eval skip {entry!r}: {type(ex).__name__}: {ex}",
                  flush=True)
            continue
        refined_rgb, refined_depth, rgb_delta, depth_delta, valid_depth = _forward_refine(
            head, batch, args
        )
        valid_rgb = batch["mask"] > args.rgb_alpha_min
        if not valid_rgb.any():
            continue
        row = {
            "prior_rgb_l1": float(_masked_l1(batch["cond"], batch["target"], valid_rgb).item()),
            "refined_rgb_l1": float(_masked_l1(refined_rgb, batch["target"], valid_rgb).item()),
            "prior_rgb_psnr": _masked_psnr(batch["cond"], batch["target"], valid_rgb),
            "refined_rgb_psnr": _masked_psnr(refined_rgb, batch["target"], valid_rgb),
            "rgb_delta_abs": float((rgb_delta.abs() * valid_rgb.permute(0, 3, 1, 2)).sum().item()
                                   / valid_rgb.sum().clamp_min(1.0).item()),
            "depth_delta_abs": float((depth_delta.abs() * batch["prior_valid"]).sum().item()
                                     / batch["prior_valid"].sum().clamp_min(1.0).item()),
        }
        if valid_depth.any():
            prior_err = (batch["prior_frac"][valid_depth] - batch["target_frac"][valid_depth]).abs()
            refined_err = (refined_depth[valid_depth] - batch["target_frac"][valid_depth]).abs()
            row["prior_depth_mae"] = float(prior_err.mean().item())
            row["refined_depth_mae"] = float(refined_err.mean().item())
        else:
            row["prior_depth_mae"] = math.nan
            row["refined_depth_mae"] = math.nan
        rows.append(row)
    head.train()
    if not rows:
        return {
            "prior_rgb_l1": math.nan,
            "refined_rgb_l1": math.nan,
            "prior_rgb_psnr": math.nan,
            "refined_rgb_psnr": math.nan,
            "prior_depth_mae": math.nan,
            "refined_depth_mae": math.nan,
            "rgb_delta_abs": math.nan,
            "depth_delta_abs": math.nan,
            "n": 0,
        }
    keys = rows[0].keys()
    out = {"n": len(rows)}
    for key in keys:
        vals = [r[key] for r in rows if math.isfinite(float(r[key]))]
        out[key] = sum(vals) / len(vals) if vals else math.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=os.environ.get("PHASE2_DATA_ROOT", ""))
    ap.add_argument("--manifest", default=os.environ.get("PHASE2_MANIFEST", "manifest.json"))
    ap.add_argument("--cond_subdir", default="ltx_decoded")
    ap.add_argument("--cond_depth_subdir", default="da3_ltx")
    ap.add_argument("--cond_view_indices", default="available")
    ap.add_argument("--mask_source", default="gt", choices=["gt", "rgb_border", "rgb_white"])
    ap.add_argument("--mask_rgb_threshold", type=float, default=0.08)
    ap.add_argument("--mask_rgb_softness", type=float, default=0.02)
    ap.add_argument("--mask_rgb_erode_px", type=int, default=0)
    ap.add_argument("--mask_rgb_dilate_px", type=int, default=0)
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
    ap.add_argument("--arch", default="unet", choices=["unet", "view"])
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--context_layers", type=int, default=2)
    ap.add_argument("--context_heads", type=int, default=4)
    ap.add_argument("--max_views", type=int, default=64)
    ap.add_argument("--rgb_residual_scale", type=float, default=0.25)
    ap.add_argument("--depth_delta_scale", type=float, default=0.8)
    ap.add_argument("--train_scale", type=float, default=0.5)
    ap.add_argument("--half_frac", type=float, default=0.5)
    ap.add_argument("--rgb_weight", type=float, default=1.0)
    ap.add_argument("--depth_weight", type=float, default=1.0)
    ap.add_argument("--grad_weight", type=float, default=0.15)
    ap.add_argument("--delta_weight", type=float, default=1e-4)
    ap.add_argument("--tv_weight", type=float, default=1e-4)
    ap.add_argument("--huber_delta", type=float, default=0.02)
    ap.add_argument("--outlier_weight", type=float, default=4.0)
    ap.add_argument("--outlier_power", type=float, default=1.0)
    ap.add_argument("--rgb_alpha_min", type=float, default=0.5)
    ap.add_argument("--apply_erode_px", type=int, default=0)
    ap.add_argument("--multiview_features", type=int, default=1)
    ap.add_argument("--multiview_refs", type=int, default=4)
    ap.add_argument("--multiview_tol_frac", type=float, default=0.02)
    ap.add_argument("--multiview_radius_px", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/condition_rgbd_pretrain")
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
        train_entries = _filter_entries_with_sidecars(
            root, train_entries, args.cond_subdir, args.cond_depth_subdir,
            args.cond_view_indices, args.views_per_step,
        )
        heldout_entries = _filter_entries_with_sidecars(
            root, heldout_entries, args.cond_subdir, args.cond_depth_subdir,
            args.cond_view_indices, args.eval_views_per_object,
        )
        print(
            f"[rgbd_pretrain] filtered sidecars train={len(train_entries)}/{before_train} "
            f"heldout={len(heldout_entries)}/{before_heldout}",
            flush=True,
        )
    if not train_entries:
        raise RuntimeError("no training entries remain after filtering/limits")
    if not heldout_entries:
        heldout_entries = train_entries[: args.eval_objects]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 6 + (4 if args.multiview_features else 0)
    if args.arch == "view":
        head = ConditionRGBDViewRefineUNet(
            hidden=args.hidden,
            in_channels=in_channels,
            max_views=args.max_views,
            context_layers=args.context_layers,
            context_heads=args.context_heads,
        ).to(device)
    else:
        head = ConditionRGBDRefineUNet(hidden=args.hidden, in_channels=in_channels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = out / "condition_rgbd_pretrain_metrics.jsonl"
    t0 = time.time()
    best_score = math.inf
    best_step = -1
    best_metrics = None

    def save_ckpt(path: Path, step: int, metrics: dict | None = None) -> None:
        ckpt = {
            "step": step,
            "args": vars(args),
            "metrics": metrics,
            "condition_rgbd_refine_head": head.state_dict(),
        }
        torch.save(ckpt, path)

    def run_eval(step: int) -> dict:
        nonlocal best_score, best_step, best_metrics
        ev = _eval_head(head, heldout_entries, args, random.Random(args.seed + 999), device)
        row = {"step": step, **ev}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[rgbd_pretrain] EVAL step {step} n={ev['n']} "
            f"rgb {ev['prior_rgb_l1']:.5f}->{ev['refined_rgb_l1']:.5f} "
            f"psnr {ev['prior_rgb_psnr']:.2f}->{ev['refined_rgb_psnr']:.2f} "
            f"depth {ev['prior_depth_mae']:.5f}->{ev['refined_depth_mae']:.5f}",
            flush=True,
        )
        score = ev["refined_rgb_l1"] + ev["refined_depth_mae"]
        if ev["n"] > 0 and math.isfinite(score) and score < best_score:
            best_score = score
            best_step = step
            best_metrics = ev
            save_ckpt(out / "condition_rgbd_refine_aux_best.pt", step, ev)
            print(
                f"[rgbd_pretrain] best step {best_step} score={best_score:.5f}",
                flush=True,
            )
        return ev

    print(
        f"[rgbd_pretrain] train={len(train_entries)} heldout={len(heldout_entries)} "
        f"views/step={args.views_per_step} scale={args.train_scale} "
        f"mask={args.mask_source} arch={args.arch} mview={args.multiview_features} "
        f"params={sum(p.numel() for p in head.parameters()):,} device={device}",
        flush=True,
    )
    if args.eval_at_start:
        run_eval(-1)
    for step in range(max(args.steps, 1)):
        entry = train_entries[rng.randrange(len(train_entries))]
        try:
            batch = _load_view_batch(root, entry, args, args.views_per_step, rng, device)
        except Exception as ex:
            print(f"[rgbd_pretrain] train skip {entry!r}: {type(ex).__name__}: {ex}",
                  flush=True)
            continue
        refined_rgb, refined_depth, rgb_delta, depth_delta, valid_depth = _forward_refine(
            head, batch, args
        )
        valid_rgb = batch["mask"] > args.rgb_alpha_min
        if not valid_rgb.any() and not valid_depth.any():
            continue
        loss = refined_rgb.new_zeros(())
        if valid_rgb.any() and args.rgb_weight > 0:
            rgb_loss = _masked_l1(refined_rgb, batch["target"], valid_rgb)
            if args.grad_weight > 0:
                rgb_loss = rgb_loss + args.grad_weight * _grad_l1(
                    refined_rgb, batch["target"], valid_rgb
                )
            loss = loss + args.rgb_weight * rgb_loss
        if valid_depth.any() and args.depth_weight > 0:
            depth_px = F.huber_loss(
                refined_depth[valid_depth],
                batch["target_frac"][valid_depth],
                delta=args.huber_delta,
                reduction="none",
            )
            if args.outlier_weight > 0:
                prior_err = (
                    batch["prior_frac"][valid_depth] - batch["target_frac"][valid_depth]
                ).abs().detach()
                weight = 1.0 + args.outlier_weight * (
                    prior_err / max(args.huber_delta, 1e-6)
                ).clamp(0.0, 20.0).pow(max(args.outlier_power, 1e-6))
                depth_px = depth_px * weight / weight.mean().clamp_min(1e-6)
            loss = loss + args.depth_weight * depth_px.mean()
        if args.delta_weight > 0:
            valid = batch["prior_valid"][:, None]
            loss = loss + args.delta_weight * (
                (rgb_delta.square() * valid).sum() / (valid.sum() * 3.0).clamp_min(1.0)
                + (depth_delta.square() * batch["prior_valid"]).sum()
                / batch["prior_valid"].sum().clamp_min(1.0)
            )
        if args.tv_weight > 0 and depth_delta.shape[-1] > 1 and depth_delta.shape[-2] > 1:
            valid = batch["prior_valid"]
            valid_x = valid[:, :, 1:] * valid[:, :, :-1]
            valid_y = valid[:, 1:, :] * valid[:, :-1, :]
            delta_all = torch.cat([rgb_delta, depth_delta[:, None]], dim=1)
            tv_x = ((delta_all[:, :, :, 1:] - delta_all[:, :, :, :-1]).abs()
                    * valid_x[:, None]).sum() / valid_x.sum().clamp_min(1.0)
            tv_y = ((delta_all[:, :, 1:, :] - delta_all[:, :, :-1, :]).abs()
                    * valid_y[:, None]).sum() / valid_y.sum().clamp_min(1.0)
            loss = loss + args.tv_weight * 0.5 * (tv_x + tv_y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(args.log_every, 1) == 0:
            prior_rgb_l1 = _masked_l1(batch["cond"], batch["target"], valid_rgb).item()
            refined_rgb_l1 = _masked_l1(refined_rgb.detach(), batch["target"], valid_rgb).item()
            prior_depth = (
                (batch["prior_frac"][valid_depth] - batch["target_frac"][valid_depth])
                .abs().mean().item()
                if valid_depth.any() else math.nan
            )
            refined_depth = (
                (refined_depth.detach()[valid_depth] - batch["target_frac"][valid_depth])
                .abs().mean().item()
                if valid_depth.any() else math.nan
            )
            print(
                f"[rgbd_pretrain] step {step} loss={loss.item():.5f} "
                f"rgb {prior_rgb_l1:.5f}->{refined_rgb_l1:.5f} "
                f"depth {prior_depth:.5f}->{refined_depth:.5f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        if step % max(args.eval_every, 1) == 0 or step == args.steps - 1:
            run_eval(step)

    save_ckpt(out / "condition_rgbd_refine_aux.pt", max(args.steps - 1, 0), best_metrics)
    print(f"[rgbd_pretrain] saved {out / 'condition_rgbd_refine_aux.pt'}", flush=True)
    if best_step >= -1 and best_metrics is not None:
        print(
            f"[rgbd_pretrain] best saved {out / 'condition_rgbd_refine_aux_best.pt'} "
            f"step={best_step} score={best_score:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
