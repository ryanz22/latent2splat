"""Direct supervised pretraining for the feed-forward depth-refine head.

The render-loop depth-refine experiments are expensive because every update
rasterizes a 3DGS scene. This script trains the same DepthRefineUNet directly
on per-view conditioning pairs:

    LTX decoded RGB + DA3 depth -> Blender GT depth

It saves only auxiliary head weights. Use train_phase2.py with
``--depth_refine_unet 1 --resume_aux_from <ckpt>`` to evaluate the head inside
the existing 3DGS renderer.
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

from decoder.clean.geometry import depth_bounds
from decoder.clean.phase2_data import (
    depth_path_at,
    frame_path_at,
    load_depth_view_at,
    load_masks_at,
    load_views_at,
    resolve_view_spec,
)
from decoder.clean.train_phase2 import DepthRefineUNet, _depth_multiview_support_maps
from decoder.data import entry_uid, load_cameras, load_depth_view, object_dir_for_entry, zdepth_to_raydist


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _filter_entries_with_sidecars(
    root: Path,
    entries: list,
    cond_subdir: str,
    cond_depth_subdir: str,
    view_spec: str,
    min_views: int,
) -> list:
    """Keep entries whose cached RGB/depth conditioning is present."""
    out = []
    min_views = max(int(min_views), 1)

    def ids_with_prefix(base: Path, prefix: str, suffix: str) -> set[int]:
        found = set()
        if not base.exists():
            return found
        pat = re.compile(rf"{re.escape(prefix)}_(\d+){re.escape(suffix)}$")
        for p in base.iterdir():
            m = pat.fullmatch(p.name)
            if m:
                found.add(int(m.group(1)))
        return found

    for entry in entries:
        try:
            obj_dir = object_dir_for_entry(root, entry)
            frame_base = frame_path_at(obj_dir, 0, subdir=cond_subdir).parent
            depth_base = depth_path_at(obj_dir, 0, subdir=cond_depth_subdir).parent
            available = ids_with_prefix(frame_base, "frame", ".png") & ids_with_prefix(
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
            if not idxs:
                continue
            if sum(1 for i in idxs if i in available) >= min_views:
                out.append(entry)
        except Exception:
            continue
    return out


def _erode_mask_2d(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Binary erosion for masks shaped (N,H,W), returned as float mask."""
    radius_px = max(int(radius_px), 0)
    if radius_px <= 0:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"expected (N,H,W) mask, got {tuple(mask.shape)}")
    x = mask[:, None].to(dtype=torch.float32)
    eroded = 1.0 - F.max_pool2d(
        1.0 - x,
        kernel_size=2 * radius_px + 1,
        stride=1,
        padding=radius_px,
    )
    return eroded[:, 0].to(dtype=mask.dtype)


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


def _resize_training_tensors(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    prior_frac: torch.Tensor,
    prior_valid: torch.Tensor,
    target_frac: torch.Tensor,
    target_valid: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale >= 0.999:
        return rgb, mask, prior_frac, prior_valid, target_frac, target_valid
    h, w = rgb.shape[-2:]
    out_size = (max(8, int(round(h * scale))), max(8, int(round(w * scale))))
    rgb = F.interpolate(rgb, size=out_size, mode="bilinear", align_corners=False)
    mask = F.interpolate(mask, size=out_size, mode="area").clamp(0.0, 1.0)
    prior_frac = F.interpolate(prior_frac[:, None], size=out_size, mode="bilinear",
                               align_corners=False)[:, 0]
    target_frac = F.interpolate(target_frac[:, None], size=out_size, mode="bilinear",
                                align_corners=False)[:, 0]
    prior_valid = (F.interpolate(prior_valid[:, None], size=out_size, mode="area")[:, 0] > 0.5).to(rgb.dtype)
    target_valid = (F.interpolate(target_valid[:, None], size=out_size, mode="area")[:, 0] > 0.5).to(rgb.dtype)
    return rgb, mask, prior_frac, prior_valid, target_frac, target_valid


def _load_view_batch(
    root: Path,
    entry,
    cond_subdir: str,
    cond_depth_subdir: str,
    view_spec: str,
    views_per_step: int,
    rng: random.Random,
    device: torch.device,
    train_scale: float,
    half_frac: float,
    multiview_features: bool = False,
    multiview_refs: int = 4,
    multiview_tol_frac: float = 0.02,
    multiview_radius_px: int = 0,
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

    frames = load_views_at(obj_dir, idxs, subdir=cond_subdir).to(device=device)
    masks = load_masks_at(obj_dir, idxs).to(device=device)
    prior_depths = torch.stack([
        load_depth_view_at(obj_dir, i, subdir=cond_depth_subdir) for i in idxs
    ]).to(device=device)
    target_depths = torch.stack([
        load_depth_view(obj_dir, i) for i in idxs
    ]).to(device=device)
    sel = torch.as_tensor(idxs, dtype=torch.long)
    k_all = cams["K"][sel].to(device=device)
    c2w_all = cams["c2w_opengl"][sel].to(device=device)
    radius = float(cams["radius"])

    prior_frac, prior_valid = _depth_frac_valid(
        prior_depths, masks, k_all, c2w_all, radius, half_frac
    )
    target_frac, target_valid = _depth_frac_valid(
        target_depths, masks, k_all, c2w_all, radius, half_frac
    )
    mv_features = None
    if multiview_features:
        mv_features = _depth_multiview_support_maps(
            prior_depths,
            masks,
            k_all,
            c2w_all,
            radius,
            multiview_tol_frac,
            multiview_refs,
            multiview_radius_px,
        )

    rgb = frames.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    mask = masks.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    rgb, mask, prior_frac, prior_valid, target_frac, target_valid = _resize_training_tensors(
        rgb, mask, prior_frac, prior_valid, target_frac, target_valid, train_scale
    )
    if mv_features is not None and mv_features.shape[-2:] != rgb.shape[-2:]:
        mv_features = F.interpolate(
            mv_features,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
    return {
        "uid": uid,
        "idxs": idxs,
        "rgb": rgb,
        "mask": mask,
        "prior_frac": prior_frac,
        "prior_valid": prior_valid,
        "target_frac": target_frac,
        "target_valid": target_valid,
        "mv_features": mv_features,
    }


def _forward_refine(
    head: DepthRefineUNet,
    batch: dict,
    delta_scale: float,
    apply_erode_px: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prior_frac = batch["prior_frac"]
    prior_valid = batch["prior_valid"]
    mask = batch["mask"]
    apply_valid = prior_valid
    if apply_erode_px > 0:
        apply_valid = apply_valid * _erode_mask_2d(mask[:, 0].clamp(0.0, 1.0), apply_erode_px)
    feat = torch.cat([
        batch["rgb"] * mask,
        mask,
        prior_frac[:, None],
        prior_valid[:, None],
    ], dim=1)
    if batch.get("mv_features") is not None:
        feat = torch.cat([
            feat,
            batch["mv_features"].to(device=feat.device, dtype=feat.dtype),
        ], dim=1)
    delta = head(feat)
    if delta_scale > 0:
        delta = delta_scale * torch.tanh(delta)
    else:
        delta = delta * 0.0
    refined = torch.sigmoid(torch.logit(prior_frac[:, None].clamp(1e-4, 1.0 - 1e-4)) + delta)[:, 0]
    refined = torch.where(apply_valid > 0.5, refined, prior_frac)
    valid = (apply_valid > 0.5) & (batch["target_valid"] > 0.5) & (mask[:, 0] > 0.5)
    return refined, delta[:, 0], valid


@torch.no_grad()
def _eval_head(
    head: DepthRefineUNet,
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
                Path(args.dataset_root), entry, args.cond_subdir, args.cond_depth_subdir,
                args.cond_view_indices, args.eval_views_per_object, rng, device,
                args.train_scale, args.half_frac,
                bool(args.depth_refine_multiview_features),
                args.depth_refine_multiview_refs,
                args.depth_refine_multiview_tol_frac,
                args.depth_refine_multiview_radius_px,
            )
        except Exception as ex:
            print(f"[depth_pretrain] eval skip {entry!r}: {type(ex).__name__}: {ex}", flush=True)
            continue
        refined, delta, valid = _forward_refine(
            head, batch, args.delta_scale, args.depth_refine_apply_erode_px
        )
        if not valid.any():
            continue
        prior_err = (batch["prior_frac"][valid] - batch["target_frac"][valid]).abs()
        refined_err = (refined[valid] - batch["target_frac"][valid]).abs()
        rows.append({
            "prior_mae": float(prior_err.mean().item()),
            "refined_mae": float(refined_err.mean().item()),
            "delta_abs": float((delta.abs() * batch["prior_valid"]).sum().item()
                               / batch["prior_valid"].sum().clamp_min(1.0).item()),
        })
    head.train()
    if not rows:
        return {"prior_mae": math.nan, "refined_mae": math.nan, "delta_abs": math.nan, "n": 0}
    return {
        "prior_mae": sum(r["prior_mae"] for r in rows) / len(rows),
        "refined_mae": sum(r["refined_mae"] for r in rows) / len(rows),
        "delta_abs": sum(r["delta_abs"] for r in rows) / len(rows),
        "n": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=os.environ.get("PHASE2_DATA_ROOT", ""))
    ap.add_argument("--manifest", default=os.environ.get("PHASE2_MANIFEST", "manifest.json"))
    ap.add_argument("--cond_subdir", default="ltx_decoded")
    ap.add_argument("--cond_depth_subdir", default="da3_ltx")
    ap.add_argument("--cond_view_indices", default="available")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--views_per_step", type=int, default=2)
    ap.add_argument("--eval_views_per_object", type=int, default=4)
    ap.add_argument("--eval_objects", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_at_start", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--delta_scale", type=float, default=0.7)
    ap.add_argument("--train_scale", type=float, default=0.5)
    ap.add_argument("--half_frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--huber_delta", type=float, default=0.02)
    ap.add_argument("--outlier_weight", type=float, default=4.0)
    ap.add_argument("--outlier_power", type=float, default=1.0)
    ap.add_argument("--prior_weight", type=float, default=1e-4)
    ap.add_argument("--tv_weight", type=float, default=1e-4)
    ap.add_argument("--depth_refine_multiview_features", type=int, default=0)
    ap.add_argument("--depth_refine_multiview_refs", type=int, default=4)
    ap.add_argument("--depth_refine_multiview_tol_frac", type=float, default=0.02)
    ap.add_argument("--depth_refine_multiview_radius_px", type=int, default=0)
    ap.add_argument("--depth_refine_apply_erode_px", type=int, default=0)
    ap.add_argument("--filter_missing_condition", type=int, default=0)
    ap.add_argument("--max_train_objects", type=int, default=0)
    ap.add_argument("--max_heldout_objects", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/depth_refine_pretrain")
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
            f"[depth_pretrain] filtered sidecars train={len(train_entries)}/{before_train} "
            f"heldout={len(heldout_entries)}/{before_heldout}",
            flush=True,
        )
    if args.max_train_objects > 0:
        train_entries = train_entries[: args.max_train_objects]
    if args.max_heldout_objects > 0:
        heldout_entries = heldout_entries[: args.max_heldout_objects]
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
    in_channels = 6 + (4 if args.depth_refine_multiview_features else 0)
    head = DepthRefineUNet(hidden=args.hidden, in_channels=in_channels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = out / "depth_pretrain_metrics.jsonl"
    t0 = time.time()
    best_score = math.inf
    best_step = -1
    best_metrics = None

    def save_ckpt(path: Path, step: int, metrics: dict | None = None) -> None:
        ckpt = {
            "step": step,
            "args": vars(args),
            "metrics": metrics,
            "depth_refine_head": head.state_dict(),
        }
        torch.save(ckpt, path)

    def run_eval(step: int) -> dict:
        nonlocal best_score, best_step, best_metrics
        ev = _eval_head(head, heldout_entries, args, random.Random(args.seed + 999),
                        device)
        row = {"step": step, **ev}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[depth_pretrain] EVAL step {step} n={ev['n']} "
            f"prior={ev['prior_mae']:.5f} refined={ev['refined_mae']:.5f} "
            f"delta={ev['delta_abs']:.5f}",
            flush=True,
        )
        if ev["n"] > 0 and math.isfinite(ev["refined_mae"]) and ev["refined_mae"] < best_score:
            best_score = ev["refined_mae"]
            best_step = step
            best_metrics = ev
            save_ckpt(out / "depth_refine_aux_best.pt", step, ev)
            print(
                f"[depth_pretrain] best step {best_step} refined={best_score:.5f} "
                f"prior={ev['prior_mae']:.5f}",
                flush=True,
            )
        return ev

    print(
        f"[depth_pretrain] train={len(train_entries)} heldout={len(heldout_entries)} "
        f"views/step={args.views_per_step} scale={args.train_scale} device={device}",
        flush=True,
    )
    if args.eval_at_start:
        run_eval(-1)
    for step in range(max(args.steps, 1)):
        entry = train_entries[rng.randrange(len(train_entries))]
        try:
            batch = _load_view_batch(
                root, entry, args.cond_subdir, args.cond_depth_subdir,
                args.cond_view_indices, args.views_per_step, rng, device,
                args.train_scale, args.half_frac,
                bool(args.depth_refine_multiview_features),
                args.depth_refine_multiview_refs,
                args.depth_refine_multiview_tol_frac,
                args.depth_refine_multiview_radius_px,
            )
        except Exception as ex:
            print(f"[depth_pretrain] train skip {entry!r}: {type(ex).__name__}: {ex}", flush=True)
            continue
        refined, delta, valid = _forward_refine(
            head, batch, args.delta_scale, args.depth_refine_apply_erode_px
        )
        if not valid.any():
            continue
        target = batch["target_frac"]
        prior = batch["prior_frac"]
        loss_px = F.huber_loss(refined[valid], target[valid],
                               delta=args.huber_delta, reduction="none")
        if args.outlier_weight > 0:
            prior_err = (prior[valid] - target[valid]).abs().detach()
            w = 1.0 + args.outlier_weight * (
                prior_err / max(args.huber_delta, 1e-6)
            ).clamp(0.0, 20.0).pow(max(args.outlier_power, 1e-6))
            loss_px = loss_px * w / w.mean().clamp_min(1e-6)
        loss = loss_px.mean()
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
            prior_mae = (prior[valid] - target[valid]).abs().mean().item()
            refined_mae = (refined.detach()[valid] - target[valid]).abs().mean().item()
            print(
                f"[depth_pretrain] step {step} loss={loss.item():.5f} "
                f"prior={prior_mae:.5f} refined={refined_mae:.5f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        if step % max(args.eval_every, 1) == 0 or step == args.steps - 1:
            run_eval(step)

    save_ckpt(out / "depth_refine_aux.pt", max(args.steps - 1, 0), best_metrics)
    print(f"[depth_pretrain] saved {out / 'depth_refine_aux.pt'}", flush=True)
    if best_step >= -1 and best_metrics is not None:
        print(
            f"[depth_pretrain] best saved {out / 'depth_refine_aux_best.pt'} "
            f"step={best_step} refined={best_metrics['refined_mae']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
