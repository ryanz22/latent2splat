"""Stage B: train a custom depth network from scratch on (LTX-RGB, GT-depth) pairs.

Replaces DA3 entirely at inference. Output is a per-view depth map written into a
new subdir (default `da3_anchor/`) matching the layout DA3 uses, so existing
fusion code reads it via --cond_depth_subdir.

Loss combines:
 - scale-invariant log-L1 inside the foreground mask
 - edge-aware gradient matching (depth derivative aligned with image derivative)
 - depth gradient hinge near silhouette boundary

Model: torchvision ResNet-18 (ImageNet pretrained) encoder + light UNet decoder.
~14M params; trains comfortably in a few hours on a single A100.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from decoder.clean.phase2_data import (
    available_frame_indices,
    depth_path_at,
    load_depth_view_at,
    load_views_at,
    load_masks_at,
)
from decoder.data import entry_uid, load_depth_view, object_dir_for_entry


DEPTH_BG_SENTINEL = 1e10
LOG_EPS = 1e-3


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)
    return (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


class DepthAnchorNet(nn.Module):
    """ResNet-18 encoder + UNet-style decoder predicting log-depth in mask.

    Input channels:
      - 3 (RGB only)
      - 5 (RGB + DA3 log-depth + DA3 valid-mask) when `use_da3_input=True`
    """

    def __init__(self, decoder_ch: int = 128, pretrained: bool = True,
                 use_da3_input: bool = False, residual_da3: bool = True):
        super().__init__()
        self.use_da3_input = use_da3_input
        self.residual_da3 = residual_da3 and use_da3_input
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        if use_da3_input:
            # 3 RGB + 1 log-depth + 1 valid-mask = 5 channels
            new_conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv1.weight[:, :3] = resnet.conv1.weight
                # Initialize new channels small (averaged across RGB)
                new_conv1.weight[:, 3:] = resnet.conv1.weight.mean(dim=1, keepdim=True) * 0.1
            self.stem = nn.Sequential(new_conv1, resnet.bn1, resnet.relu)
        else:
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # /2, 64
        self.pool = resnet.maxpool                                          # /4
        self.layer1 = resnet.layer1                                         # /4,  64
        self.layer2 = resnet.layer2                                         # /8,  128
        self.layer3 = resnet.layer3                                         # /16, 256
        self.layer4 = resnet.layer4                                         # /32, 512

        c = decoder_ch
        self.up4 = self._up_block(512, c)
        self.fuse3 = self._fuse_block(256 + c, c)
        self.up3 = self._up_block(c, c)
        self.fuse2 = self._fuse_block(128 + c, c)
        self.up2 = self._up_block(c, c)
        self.fuse1 = self._fuse_block(64 + c, c)
        self.up1 = self._up_block(c, c)
        self.fuse0 = self._fuse_block(64 + c, c)
        self.up0 = self._up_block(c, c)
        self.head = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(c // 2, 1, 1),
        )
        if self.residual_da3:
            final = self.head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    @staticmethod
    def _fuse_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GELU(),
        )

    def forward(self, rgb: torch.Tensor, da3_log: torch.Tensor | None = None,
                da3_valid: torch.Tensor | None = None) -> torch.Tensor:
        """rgb: (B, 3, H, W) in [0,1]. Returns log-depth (B, 1, H, W).

        When use_da3_input=True, da3_log (B, 1, H, W) and da3_valid (B, 1, H, W)
        are concatenated to RGB. da3_log is the log of DA3 depth at FG; 0 elsewhere.
        """
        x = _imagenet_normalize(rgb)
        if self.use_da3_input:
            assert da3_log is not None and da3_valid is not None
            x = torch.cat([x, da3_log, da3_valid], dim=1)
        s0 = self.stem(x)              # H/2,  64
        p = self.pool(s0)               # H/4
        s1 = self.layer1(p)            # H/4,  64
        s2 = self.layer2(s1)           # H/8,  128
        s3 = self.layer3(s2)           # H/16, 256
        s4 = self.layer4(s3)           # H/32, 512

        u = self.up4(s4)                                       # H/16
        u = self.fuse3(torch.cat([u, s3], dim=1))
        u = self.up3(u)                                        # H/8
        u = self.fuse2(torch.cat([u, s2], dim=1))
        u = self.up2(u)                                        # H/4
        u = self.fuse1(torch.cat([u, s1], dim=1))
        u = self.up1(u)                                        # H/2
        u = self.fuse0(torch.cat([u, s0], dim=1))
        u = self.up0(u)                                        # H
        residual = self.head(u)
        if self.residual_da3:
            assert da3_log is not None and da3_valid is not None
            base = torch.where(da3_valid > 0.5, da3_log, torch.zeros_like(da3_log))
            return base + residual
        return residual


def _load_manifest(p: Path) -> dict:
    return json.loads(p.read_text())


def _depth_ids(obj_dir: Path, subdir: str) -> set[int]:
    base = depth_path_at(obj_dir, 0, subdir=subdir).parent
    out = set()
    if not base.exists():
        return out
    for p in base.glob("depth_*.npy"):
        stem = p.stem.removeprefix("depth_")
        if stem.isdigit():
            out.add(int(stem))
    return out


def _filter_entries_with_sidecars(root: Path,
                                  entries: list,
                                  cond_subdir: str,
                                  da3_subdir: str,
                                  min_views: int,
                                  require_da3: bool) -> list:
    min_views = max(int(min_views), 1)
    out = []
    for entry in entries:
        obj_dir = object_dir_for_entry(root, entry)
        try:
            frame_ids = set(available_frame_indices(obj_dir, subdir=cond_subdir))
            ids = frame_ids & _depth_ids(obj_dir, da3_subdir) if require_da3 else frame_ids
        except Exception:
            continue
        if len(ids) >= min_views:
            out.append(entry)
    return out


def _load_sample(obj_dir: Path, view_idx: int, train_scale: float, device: torch.device,
                 cond_subdir: str = "ltx_decoded",
                 da3_subdir: str | None = None):
    """Load one (RGB, GT-depth, mask, da3_log, da3_valid) sample at given view.

    When `da3_subdir` is None, the returned da3_log/da3_valid are zeros (unused).
    """
    rgb = load_views_at(obj_dir, [view_idx], subdir=cond_subdir)[0]      # (H, W, 3) [0,1]
    mask = load_masks_at(obj_dir, [view_idx])[0]                          # (H, W, 1) {0,1}
    gt = load_depth_view(obj_dir, view_idx)                               # (H, W) float32

    rgb = rgb.permute(2, 0, 1).unsqueeze(0).to(device)                    # (1, 3, H, W)
    mask = mask.permute(2, 0, 1).unsqueeze(0).to(device)                  # (1, 1, H, W)
    gt = gt.unsqueeze(0).unsqueeze(0).to(device)                          # (1, 1, H, W)

    if da3_subdir:
        da3 = load_depth_view_at(obj_dir, view_idx, subdir=da3_subdir).to(device)
        if da3.shape != rgb.shape[-2:]:
            da3 = F.interpolate(da3[None, None], size=rgb.shape[-2:], mode="bilinear",
                                align_corners=False)[0, 0]
        da3 = da3.unsqueeze(0).unsqueeze(0)
        # Valid where finite + within sane range
        da3_valid = (torch.isfinite(da3) & (da3 > 1e-3) & (da3 < 1e6)).float()
        # log-depth, with 0 where invalid (gated by valid channel)
        da3_log = torch.where(da3_valid > 0.5, torch.log(da3.clamp_min(LOG_EPS)),
                              torch.zeros_like(da3))
    else:
        da3_log = torch.zeros_like(gt)
        da3_valid = torch.zeros_like(gt)

    if train_scale < 0.999:
        size = (max(8, int(round(rgb.shape[-2] * train_scale))),
                max(8, int(round(rgb.shape[-1] * train_scale))))
        rgb = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=size, mode="area").clamp(0.0, 1.0)
        gt = F.interpolate(gt, size=size, mode="nearest")
        da3_log = F.interpolate(da3_log, size=size, mode="bilinear", align_corners=False)
        da3_valid = F.interpolate(da3_valid, size=size, mode="area").clamp(0.0, 1.0)

    # Valid: alpha is foreground AND GT depth is finite (not 1e10 sentinel)
    valid = (mask > 0.5) & torch.isfinite(gt) & (gt > 1e-3) & (gt < 1e6)
    return rgb, mask, gt, valid, da3_log, da3_valid


def _log_depth(d: torch.Tensor) -> torch.Tensor:
    return torch.log(d.clamp_min(LOG_EPS))


def _depth_from_log(ld: torch.Tensor) -> torch.Tensor:
    return torch.exp(ld)


def loss_fn(pred_log: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
            rgb: torch.Tensor, edge_w: float, grad_w: float):
    """Mask-restricted log-L1 + edge-aware gradient L1.

    pred_log: (B, 1, H, W) predicted log-depth
    gt:       (B, 1, H, W) GT depth in metric units
    valid:    (B, 1, H, W) boolean mask
    rgb:      (B, 3, H, W) RGB for edge-aware weighting
    """
    n_valid = valid.float().sum().clamp_min(1.0)

    # Log-depth L1 inside valid region
    gt_log = _log_depth(gt)
    diff = (pred_log - gt_log) * valid.float()
    l1 = diff.abs().sum() / n_valid

    # Edge-aware: penalize depth gradient where image is smooth (texture-aware)
    if edge_w > 0:
        dx = (pred_log[..., 1:] - pred_log[..., :-1]).abs()
        dy = (pred_log[..., 1:, :] - pred_log[..., :-1, :]).abs()
        # Image edge magnitude (high → don't penalize depth jump)
        rgb_dx = (rgb[..., 1:] - rgb[..., :-1]).abs().mean(dim=1, keepdim=True)
        rgb_dy = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(dim=1, keepdim=True)
        w_dx = torch.exp(-10.0 * rgb_dx)
        w_dy = torch.exp(-10.0 * rgb_dy)
        v_dx = valid[..., 1:].float() * valid[..., :-1].float()
        v_dy = valid[..., 1:, :].float() * valid[..., :-1, :].float()
        edge_loss = ((dx * w_dx * v_dx).sum() + (dy * w_dy * v_dy).sum()) \
                    / (v_dx.sum() + v_dy.sum()).clamp_min(1.0)
        l1 = l1 + edge_w * edge_loss

    # Gradient-matching: depth gradient should match GT gradient direction
    if grad_w > 0:
        pred_dx = pred_log[..., 1:] - pred_log[..., :-1]
        pred_dy = pred_log[..., 1:, :] - pred_log[..., :-1, :]
        gt_dx = gt_log[..., 1:] - gt_log[..., :-1]
        gt_dy = gt_log[..., 1:, :] - gt_log[..., :-1, :]
        v_dx = valid[..., 1:].float() * valid[..., :-1].float()
        v_dy = valid[..., 1:, :].float() * valid[..., :-1, :].float()
        grad_loss = (((pred_dx - gt_dx).abs() * v_dx).sum()
                     + ((pred_dy - gt_dy).abs() * v_dy).sum()) \
                    / (v_dx.sum() + v_dy.sum()).clamp_min(1.0)
        l1 = l1 + grad_w * grad_loss

    return l1


@torch.no_grad()
def eval_model(model, root: Path, entries: list, args, device) -> dict:
    model.eval()
    fg_l1_log = []
    fg_l1_metric = []
    da3_l1_metric = []
    n = 0
    use_da3 = bool(args.use_da3_input)
    da3_subdir = args.da3_subdir if use_da3 else None
    for entry in entries[: args.eval_objects]:
        obj_dir = object_dir_for_entry(root, entry)
        try:
            idxs = available_frame_indices(obj_dir, subdir=args.cond_subdir)
        except Exception:
            continue
        if not idxs:
            continue
        for view_idx in idxs[: args.eval_views_per_object]:
            try:
                rgb, mask, gt, valid, da3_log, da3_valid = _load_sample(
                    obj_dir, view_idx, 1.0, device,
                    cond_subdir=args.cond_subdir,
                    da3_subdir=da3_subdir,
                )
            except Exception:
                continue
            if not valid.any():
                continue
            if use_da3:
                pred_log = model(rgb, da3_log, da3_valid)
            else:
                pred_log = model(rgb)
            pred = _depth_from_log(pred_log)

            # Metric: L1 in log space on valid pixels
            err_log = ((pred_log - _log_depth(gt)) * valid.float()).abs().sum() / valid.float().sum()
            err_metric = ((pred - gt) * valid.float()).abs().sum() / valid.float().sum()
            fg_l1_log.append(err_log.item())
            fg_l1_metric.append(err_metric.item())

            # DA3 baseline comparison (always log it, regardless of input mode)
            try:
                da3_raw = load_depth_view_at(
                    obj_dir, view_idx, subdir=args.da3_subdir
                ).to(device)
                if da3_raw.shape != gt.shape[-2:]:
                    da3_raw = F.interpolate(da3_raw[None, None], size=gt.shape[-2:], mode="bilinear",
                                            align_corners=False)[0, 0]
                da3_raw = da3_raw.unsqueeze(0).unsqueeze(0)
                err_da3 = ((da3_raw - gt) * valid.float()).abs().sum() / valid.float().sum()
                da3_l1_metric.append(err_da3.item())
            except Exception:
                pass
            n += 1
    model.train()
    return {
        "n": n,
        "pred_l1_log": float(np.mean(fg_l1_log)) if fg_l1_log else float("nan"),
        "pred_l1_metric": float(np.mean(fg_l1_metric)) if fg_l1_metric else float("nan"),
        "da3_l1_metric": float(np.mean(da3_l1_metric)) if da3_l1_metric else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=os.environ.get("PHASE2_DATA_ROOT", ""))
    ap.add_argument("--manifest", default=os.environ.get("PHASE2_MANIFEST", "manifest.json"))
    ap.add_argument("--cond_subdir", default="ltx_decoded")
    ap.add_argument("--da3_subdir", default="da3_ltx")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--train_scale", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--edge_w", type=float, default=0.1)
    ap.add_argument("--grad_w", type=float, default=0.5)
    ap.add_argument("--decoder_ch", type=int, default=128)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--use_da3_input", type=int, default=1,
                    help="Concat DA3 log-depth + valid-mask to RGB input (5ch).")
    ap.add_argument("--residual_da3", type=int, default=1,
                    help="When DA3 input is enabled, predict a residual over DA3 log-depth.")
    ap.add_argument("--max_train_objects", type=int, default=0)   # 0 = use all
    ap.add_argument("--filter_missing_condition", type=int, default=0)
    ap.add_argument("--filter_missing_condition_min_views", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_at_start", type=int, default=1)
    ap.add_argument("--eval_objects", type=int, default=10)
    ap.add_argument("--eval_views_per_object", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/depth_anchor")
    args = ap.parse_args()

    if not args.dataset_root:
        raise ValueError("--dataset_root or PHASE2_DATA_ROOT is required")
    root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = _load_manifest(manifest_path)
    train_entries = list(manifest["train"])
    if args.max_train_objects > 0:
        train_entries = train_entries[: args.max_train_objects]
    heldout_entries = list(manifest.get("eval", [])) + list(manifest.get("test", []))
    if not heldout_entries:
        heldout_entries = train_entries[: args.eval_objects]
    if args.filter_missing_condition:
        before_train = len(train_entries)
        before_heldout = len(heldout_entries)
        require_da3 = bool(args.use_da3_input)
        train_entries = _filter_entries_with_sidecars(
            root, train_entries, args.cond_subdir, args.da3_subdir,
            args.filter_missing_condition_min_views, require_da3,
        )
        heldout_entries = _filter_entries_with_sidecars(
            root, heldout_entries, args.cond_subdir, args.da3_subdir,
            args.filter_missing_condition_min_views, require_da3,
        )
        print(
            f"[depth_anchor] filtered sidecars train={len(train_entries)}/{before_train} "
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
    model = DepthAnchorNet(decoder_ch=args.decoder_ch,
                           pretrained=not args.no_pretrained,
                           use_da3_input=bool(args.use_da3_input),
                           residual_da3=bool(args.residual_da3)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[depth_anchor] model params: {n_params:,}", flush=True)
    print(f"[depth_anchor] train={len(train_entries)} heldout={len(heldout_entries)}", flush=True)

    metrics_path = out / "metrics.jsonl"
    t0 = time.time()
    best_log_l1 = math.inf
    best_step = -1

    def save_ckpt(path: Path, step: int, ev: dict | None) -> None:
        torch.save({
            "step": step,
            "args": vars(args),
            "metrics": ev,
            "model": model.state_dict(),
        }, path)

    def run_eval(step: int) -> dict:
        nonlocal best_log_l1, best_step
        ev = eval_model(model, root, heldout_entries, args, device)
        row = {"step": step, **ev}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[depth_anchor] EVAL step {step} n={ev['n']} "
              f"pred_log_l1={ev['pred_l1_log']:.5f} "
              f"pred_metric_l1={ev['pred_l1_metric']:.5f} "
              f"da3_metric_l1={ev['da3_l1_metric']:.5f}", flush=True)
        if math.isfinite(ev["pred_l1_log"]) and ev["pred_l1_log"] < best_log_l1:
            best_log_l1 = ev["pred_l1_log"]
            best_step = step
            save_ckpt(out / "depth_anchor_best.pt", step, ev)
            print(f"[depth_anchor] best step {step} pred_log_l1={ev['pred_l1_log']:.5f}", flush=True)
        return ev

    if args.eval_at_start:
        run_eval(-1)

    model.train()
    use_da3 = bool(args.use_da3_input)
    da3_subdir_in = args.da3_subdir if use_da3 else None
    for step in range(args.steps):
        # Build a batch by sampling random (object, view) pairs
        batch_rgb, batch_mask, batch_gt, batch_valid = [], [], [], []
        batch_da3_log, batch_da3_valid = [], []
        attempts = 0
        while len(batch_rgb) < args.batch_size and attempts < args.batch_size * 4:
            attempts += 1
            entry = train_entries[rng.randrange(len(train_entries))]
            obj_dir = object_dir_for_entry(root, entry)
            try:
                idxs = available_frame_indices(obj_dir, subdir=args.cond_subdir)
            except Exception:
                continue
            if not idxs:
                continue
            view_idx = idxs[rng.randrange(len(idxs))]
            try:
                rgb, mask, gt, valid, da3_log, da3_valid = _load_sample(
                    obj_dir, view_idx, args.train_scale, device,
                    cond_subdir=args.cond_subdir,
                    da3_subdir=da3_subdir_in,
                )
            except Exception as ex:
                continue
            if not valid.any():
                continue
            batch_rgb.append(rgb)
            batch_mask.append(mask)
            batch_gt.append(gt)
            batch_valid.append(valid)
            batch_da3_log.append(da3_log)
            batch_da3_valid.append(da3_valid)
        if not batch_rgb:
            continue
        rgb_b = torch.cat(batch_rgb, dim=0)
        mask_b = torch.cat(batch_mask, dim=0)
        gt_b = torch.cat(batch_gt, dim=0)
        valid_b = torch.cat(batch_valid, dim=0)
        da3_log_b = torch.cat(batch_da3_log, dim=0)
        da3_valid_b = torch.cat(batch_da3_valid, dim=0)

        if use_da3:
            pred_log = model(rgb_b, da3_log_b, da3_valid_b)
        else:
            pred_log = model(rgb_b)
        loss = loss_fn(pred_log, gt_b, valid_b, rgb_b, args.edge_w, args.grad_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if step % max(args.log_every, 1) == 0:
            with torch.no_grad():
                pred = _depth_from_log(pred_log)
                metric_l1 = ((pred - gt_b) * valid_b.float()).abs().sum() / valid_b.float().sum().clamp_min(1)
            print(f"[depth_anchor] step {step} loss={loss.item():.5f} "
                  f"metric_l1={metric_l1.item():.5f} lr={sched.get_last_lr()[0]:.2e} "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
        if step > 0 and step % max(args.eval_every, 1) == 0:
            run_eval(step)

    # Final
    run_eval(args.steps - 1)
    save_ckpt(out / "depth_anchor_final.pt", args.steps - 1, None)
    print(f"[depth_anchor] saved final ckpt to {out}", flush=True)
    if best_step >= 0:
        print(f"[depth_anchor] BEST step={best_step} pred_log_l1={best_log_l1:.5f} "
              f"({out / 'depth_anchor_best.pt'})", flush=True)


if __name__ == "__main__":
    main()
