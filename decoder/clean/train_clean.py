"""Phase-1 gate: overfit the clean decoder on ONE object (K=1, reference anchor),
reporting the HONEST fixed-bg FG-masked PSNR (not the per-step random-bg number).

  ./.venv/bin/python -m decoder.clean.train_clean --uid af3281f986cc40b9b3cbca1f72e77f46
"""
from __future__ import annotations

import argparse
import torch

from decoder.data import ObjaverseLatentDataset
from decoder.render import render_views, _ssim
from decoder.clean.network import CleanGSDecoder
from decoder.clean.losses import mask_alpha_l1, scale_hinge, VGGPerceptual
from decoder.clean.metrics import fg_masked_psnr, sharpness_ratio

V4 = "/home/rrzhang/projects/data/Animals v4 Final"
MANIFEST = V4 + "/animals_v4_approved_encoded.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="af3281f986cc40b9b3cbca1f72e77f46")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--depth_layers", type=int, default=12)
    ap.add_argument("--train_views", type=int, default=8)   # §4.4 subsample to fit 32 GB
    ap.add_argument("--fg_weight", type=float, default=10.0)  # upweight object pixels
    ap.add_argument("--mask_weight", type=float, default=0.5)  # alpha-mask L1 weight
    ap.add_argument("--out_dir", default="runs/clean_best")    # save renders + viz here
    ap.add_argument("--scale_cap_frac", type=float, default=0.05)  # Gaussian scale cap (frac of radius)
    ap.add_argument("--seed", type=int, default=0)   # fix init + view/bg sampling so caps are comparable
    ap.add_argument("--perceptual_weight", type=float, default=0.0)  # VGG16 feature L1 (sharpness lever)
    ap.add_argument("--ups_stages", type=int, default=4)  # 4→98k Gaussians, 5→393k (~1/render-px)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    ds = ObjaverseLatentDataset(V4, split="train", manifest_path=MANIFEST)
    idx = next(i for i, e in enumerate(ds.entries) if e["uid"] == args.uid)
    s = ds[idx]
    latent = s["latent"][None].to(dev)
    K = s["K"].to(dev); c2w_gl = s["c2w_opengl"].to(dev)
    w2c = s["w2c"].to(dev)
    target = s["frames"].to(dev)                  # (V,H,W,3), white bg
    fg = s["masks"].to(dev)                        # (V,H,W,1)
    radius = float(s["radius"]); w, h = s["width"], s["height"]
    ref_K, ref_c2w = K[0], c2w_gl[0]

    model = CleanGSDecoder(depth=args.depth_layers, scale_cap_frac=args.scale_cap_frac,
                           ups_stages=args.ups_stages).to(dev)
    percep = VGGPerceptual().to(dev) if args.perceptual_weight > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=100)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps - 100, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[100])

    V = target.shape[0]
    nv = min(args.train_views, V)

    def eval_fg_psnr(params, chunk: int = 8) -> float:
        """Honest fixed-white-bg FG-masked PSNR over ALL views, rendered in chunks."""
        outs = []
        with torch.no_grad():
            for i in range(0, V, chunk):
                r, _ = render_views(params, w2c[i:i + chunk], K[i:i + chunk], w, h, bg=1.0)
                outs.append(r)
        return fg_masked_psnr(torch.cat(outs, 0), target, fg.bool())

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[clean] uid={args.uid} radius={radius:.3f} n_params={n_params:,} "
          f"train_views={nv}/{V}", flush=True)
    for step in range(args.steps):
        opt.zero_grad()
        # subsample views (always include the reference view 0) — §4.4, fits 32 GB
        idx = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                         torch.randperm(V - 1, device=dev)[:nv - 1] + 1])
        cur_bg = float(torch.rand(1))                          # random-bg compositing (C1)
        tgt = (target * fg + cur_bg * (1 - fg))[idx]
        p = {k: v[0] for k, v in model(latent, ref_K, ref_c2w, radius).items()}
        render, alpha = render_views(p, w2c[idx], K[idx], w, h, bg=cur_bg)
        # foreground-weighted L1: the object is ~18% of pixels; with random_bg a
        # transparent render trivially matches the 82% background, so plain L1
        # collapses opacity to 0. fg_weight upweights the object (Method-E lever).
        per_px = (render - tgt).abs().mean(-1, keepdim=True)        # (v,H,W,1)
        wmap = 1.0 + (args.fg_weight - 1.0) * fg[idx]
        l1 = (per_px * wmap).sum() / (wmap.sum() + 1e-8)
        ssim = _ssim(render, tgt)
        mask = mask_alpha_l1(alpha, fg[idx])
        hinge = scale_hinge(p["scale"], s_min=0.005, s_max=0.05 * radius)
        # NOTE: no GLOBAL opacity reg — it penalizes object Gaussians too and
        # collapses opacity (documented). mask-L1 + pruning handle empty space.
        loss = l1 + 0.2 * (1 - ssim) + args.mask_weight * mask + 0.01 * hinge
        pl = args.perceptual_weight * percep(render, tgt) if percep is not None else None
        if pl is not None:
            loss = loss + pl
        if step == 0:
            print(f"[clean] step0 components: l1={float(l1):.4f} ssim={float(0.2*(1-ssim)):.4f} "
                  f"mask={float(args.mask_weight*mask):.4f} "
                  f"percep={float(pl) if pl is not None else 0.0:.4f}", flush=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            del render, alpha, loss
            pe = {k: v.detach() for k, v in p.items()}
            fgp = eval_fg_psnr(pe)
            with torch.no_grad():
                op = pe["opacity"].reshape(-1)
                _, a_ref = render_views(pe, w2c[:1], K[:1], w, h, bg=1.0)  # ref-view alpha
                a = a_ref[0, ..., 0]; fg0 = fg[0, ..., 0].bool()
                obj_a, bg_a = float(a[fg0].mean()), float(a[~fg0].mean())
            print(f"[clean] step {step:4d} FG-PSNR={fgp:.2f} op_mean={float(op.mean()):.4f} "
                  f"op_p99={float(op.quantile(0.99)):.3f} alive={float((op>0.01).float().mean()):.3f} "
                  f"obj_alpha={obj_a:.3f} bg_alpha={bg_a:.3f}", flush=True)

    # save renders + a target|render|alpha viz grid (same format as Method E's _viz)
    if args.out_dir:
        from pathlib import Path
        from decoder.clean.viz import _viz
        pe = {k: v.detach() for k, v in p.items()}
        rs, as_ = [], []
        with torch.no_grad():
            for i in range(0, V, 8):
                r, a = render_views(pe, w2c[i:i + 8], K[i:i + 8], w, h, bg=1.0)  # fixed white bg
                rs.append(r.cpu()); as_.append(a.cpu())
        render_all = torch.cat(rs)
        sr = sharpness_ratio(render_all, target.cpu(), fg.cpu().bool().squeeze(-1))
        print(f"[clean] FINAL scale_cap_frac={args.scale_cap_frac} FG-PSNR={fgp:.2f} "
              f"sharpness_ratio={sr:.3f}", flush=True)
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        ckpt = out / "overfit_clean.pt"
        torch.save({"render": render_all, "alpha": torch.cat(as_), "target": target.cpu(),
                    "model": model.state_dict(),
                    "result": {"uid": args.uid, "final_psnr": fgp, "last_l1": 0.0}}, ckpt)
        _viz(str(ckpt), str(out))


if __name__ == "__main__":
    main()
