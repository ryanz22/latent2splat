"""Ceiling check, Method A: free the clean decoder's OUTPUT and optimize it directly.

We replace `raw = CleanGSDecoder(latent, ...)` with a learnable `raw` Parameter and run the
*identical* activate + ray-anchor + render + loss path (no latent, no network). This localizes
the Phase-1 overfit blur: if these 393k freely-optimized ray-anchored Gaussians get sharp, the
representation is fine and the blur is our decoder (upsampler-from-24x16 / optimization / loss);
if they ALSO plateau near the decoder, the pixel-aligned form/budget is itself the wall.

Note: in a single-object overfit the network can memorize, so this is NOT a test of the latent
(that is a Phase-2 / generalization question).

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python -m decoder.clean.fit_freegauss --lr 1e-2 --steps 2500
"""
from __future__ import annotations

import argparse
import torch

from decoder.clean.geometry import ray_dirs_world
from decoder.clean.network import LH, LW
from decoder.clean.geometry import depth_bounds
from decoder.clean.gaussians import activate

V4 = "/home/rrzhang/projects/data/Animals v4 Final"
MANIFEST = V4 + "/animals_v4_approved_encoded.json"


def free_gaussian_map(ref_K, ref_c2w, radius, ups_stages=5, half_frac=0.5,
                      seed=0, device="cpu"):
    """Method-A setup: the decoder's exact output form with the network removed.

    Returns (raw, anchor) where raw is a learnable nn.Parameter (N,12) and
    anchor = (origins, dirs, d_near, d_far); N = map_h*map_w = (LH·2^ups)·(LW·2^ups).
    raw is init'd small (~N(0,0.02^2)) so activate() reproduces the decoder's start
    (rgb~0.5, opacity~sigmoid(B_ALPHA), scale small, depth at the shell midpoint). The
    dirs come from the SAME ray_dirs_world call the decoder head uses, so raw[i] anchors
    to the same pixel ray as decoder head pixel i — same form, no network.
    """
    map_h, map_w = LH * 2 ** ups_stages, LW * 2 ** ups_stages
    gen = torch.Generator().manual_seed(seed)                       # CPU generator → reproducible
    raw = torch.nn.Parameter((0.02 * torch.randn(map_h * map_w, 12, generator=gen)).to(device))
    dirs = ray_dirs_world(ref_K, ref_c2w, map_h, map_w).to(device)  # (N,3) world, unit
    origins = ref_c2w[:3, 3].to(device).expand_as(dirs)            # (N,3)
    d_near, d_far = depth_bounds(ref_c2w, radius, half_frac)
    return raw, (origins, dirs, d_near, d_far)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="af3281f986cc40b9b3cbca1f72e77f46")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-2)   # free pre-activation params: far > the 4e-4 network LR
    ap.add_argument("--train_views", type=int, default=8)
    ap.add_argument("--fg_weight", type=float, default=10.0)
    ap.add_argument("--mask_weight", type=float, default=0.5)
    ap.add_argument("--scale_cap_frac", type=float, default=0.012)   # exp016 best
    ap.add_argument("--perceptual_weight", type=float, default=0.5)  # exp016 best
    ap.add_argument("--ups_stages", type=int, default=5)             # 5 → 393,216 Gaussians
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/ceiling_freegauss")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # heavy / GPU-only deps live here so the unit test can import this module gsplat-free
    from decoder.data import ObjaverseLatentDataset
    from decoder.render import render_views, _ssim
    from decoder.clean.losses import mask_alpha_l1, scale_hinge, VGGPerceptual
    from decoder.clean.metrics import fg_masked_psnr, sharpness_ratio

    ds = ObjaverseLatentDataset(V4, split="train", manifest_path=MANIFEST)
    idx0 = next(i for i, e in enumerate(ds.entries) if e["uid"] == args.uid)
    s = ds[idx0]
    K = s["K"].to(dev); c2w_gl = s["c2w_opengl"].to(dev)
    w2c = s["w2c"].to(dev)
    target = s["frames"].to(dev)                   # (V,H,W,3), white bg
    fg = s["masks"].to(dev)                        # (V,H,W,1)
    radius = float(s["radius"]); w, h = s["width"], s["height"]
    ref_K, ref_c2w = K[0], c2w_gl[0]

    # the one substitution: a free raw map in place of the decoder's network
    raw, (origins, dirs, dn, df) = free_gaussian_map(
        ref_K, ref_c2w, radius, ups_stages=args.ups_stages, seed=args.seed, device=dev)
    percep = VGGPerceptual().to(dev) if args.perceptual_weight > 0 else None
    # NO weight decay: that would pull raw toward its (gray) init — the decoder's wd=0.05
    # regularized network weights, not the output, so a fair ceiling fit uses wd=0.
    opt = torch.optim.AdamW([raw], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=100)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps - 100, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[100])

    V = target.shape[0]
    nv = min(args.train_views, V)
    fgb = fg.bool()

    def cur_params():
        return activate(raw, origins, dirs, dn, df, radius, args.scale_cap_frac)

    def eval_all(chunk: int = 8):
        """Honest fixed-white-bg render over ALL views → FG-PSNR, sharpness, ref obj/bg alpha."""
        rs, as_ = [], []
        with torch.no_grad():
            p = cur_params()
            for i in range(0, V, chunk):
                r, a = render_views(p, w2c[i:i + chunk], K[i:i + chunk], w, h, bg=1.0)
                rs.append(r); as_.append(a)
        render_all, alpha_all = torch.cat(rs, 0), torch.cat(as_, 0)
        fgp = fg_masked_psnr(render_all, target, fgb)
        sr = sharpness_ratio(render_all, target, fgb.squeeze(-1))
        a0, m0 = alpha_all[0, ..., 0], fgb[0, ..., 0]
        return fgp, sr, float(a0[m0].mean()), float(a0[~m0].mean()), render_all, alpha_all

    print(f"[free] uid={args.uid} radius={radius:.3f} N={raw.shape[0]:,} "
          f"train_views={nv}/{V} lr={args.lr} cap={args.scale_cap_frac} "
          f"percep={args.perceptual_weight}", flush=True)
    last_l1 = 0.0
    for step in range(args.steps):
        opt.zero_grad()
        idx = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                         torch.randperm(V - 1, device=dev)[:nv - 1] + 1])
        cur_bg = float(torch.rand(1))                          # random-bg compositing (C1)
        tgt = (target * fg + cur_bg * (1 - fg))[idx]
        p = cur_params()
        render, alpha = render_views(p, w2c[idx], K[idx], w, h, bg=cur_bg)
        # --- loss: byte-identical to train_clean.py at exp016 settings (matched protocol) ---
        per_px = (render - tgt).abs().mean(-1, keepdim=True)        # (v,H,W,1)
        wmap = 1.0 + (args.fg_weight - 1.0) * fg[idx]
        l1 = (per_px * wmap).sum() / (wmap.sum() + 1e-8)
        ssim = _ssim(render, tgt)
        mask = mask_alpha_l1(alpha, fg[idx])
        hinge = scale_hinge(p["scale"], s_min=0.005, s_max=0.05 * radius)
        loss = l1 + 0.2 * (1 - ssim) + args.mask_weight * mask + 0.01 * hinge
        if percep is not None:
            loss = loss + args.perceptual_weight * percep(render, tgt)
        # -------------------------------------------------------------------------------------
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            last_l1 = float(l1)
            del render, alpha, loss
            fgp, sr, obj_a, bg_a, render_all, alpha_all = eval_all()
            with torch.no_grad():
                op = cur_params()["opacity"].reshape(-1)
            # FG-PSNR + sharp every 100 = the convergence signal for the validity guard (§1.5):
            # only conclude "form is capped" once sharp has plateaued.
            print(f"[free] step {step:4d} FG-PSNR={fgp:.2f} sharp={sr:.3f} "
                  f"op_mean={float(op.mean()):.4f} op_p99={float(op.quantile(0.99)):.3f} "
                  f"alive={float((op > 0.01).float().mean()):.3f} "
                  f"obj_alpha={obj_a:.3f} bg_alpha={bg_a:.3f}", flush=True)

    if args.out_dir:
        from pathlib import Path
        from decoder.clean.viz import _viz
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        fgp, sr, _, _, render_all, alpha_all = eval_all()
        print(f"[free] FINAL lr={args.lr} cap={args.scale_cap_frac} ups={args.ups_stages} "
              f"FG-PSNR={fgp:.2f} sharpness_ratio={sr:.3f}", flush=True)
        ckpt = out / "overfit_free.pt"
        torch.save({"render": render_all.cpu(), "alpha": alpha_all.cpu(),
                    "target": target.cpu(), "raw": raw.detach().cpu(),
                    "result": {"uid": args.uid, "final_psnr": fgp, "sharpness": sr,
                               "last_l1": last_l1, "lr": args.lr}}, ckpt)
        _viz(str(ckpt), str(out))


if __name__ == "__main__":
    main()
