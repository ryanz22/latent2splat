"""Per-object free 3DGS ceiling with gsplat densification.

This is the control run the latent decoder is missing: remove the shared
network and the K=1 ray-anchor constraint, then optimize ordinary free 3DGS
parameters for one object with split/duplicate/prune enabled.

It answers a different question than fit_freegauss.py:
  * fit_freegauss.py: same ray-anchored representation, no network.
  * fit_densified_gs.py: free means + adaptive densification, no network.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from decoder.clean.geometry import ray_dirs_world

V4 = os.environ.get("PHASE2_DATA_ROOT", "/home/rrzhang/projects/data/Animals v4 Final")
MANIFEST = os.environ.get("PHASE2_MANIFEST", V4 + "/animals_v4_approved_encoded.json")


def _logit(x: torch.Tensor) -> torch.Tensor:
    return torch.logit(x.clamp(1e-4, 1 - 1e-4))


def _pick_sample(ds, uid: str | None) -> dict:
    if uid:
        idx = next(i for i, e in enumerate(ds.entries) if e["uid"] == uid)
    else:
        idx = 0
    return ds[idx]


def _init_from_ref_view(sample: dict, n: int, init_scale: float, init_opacity: float,
                        device: str, seed: int) -> torch.nn.ParameterDict:
    """Foreground-pixel lift from view 0, with random depths through the object shell."""
    gen = torch.Generator(device=device).manual_seed(seed)
    h, w = sample["height"], sample["width"]
    radius = float(sample["radius"])
    ref_K = sample["K"][0].to(device)
    ref_c2w = sample["c2w_opengl"][0].to(device)
    ref_img = sample["frames"][0].to(device)
    ref_mask = sample["masks"][0, ..., 0].to(device).bool()

    fg_idx = torch.where(ref_mask.reshape(-1))[0]
    if len(fg_idx) == 0:
        fg_idx = torch.arange(h * w, device=device)
    sel = fg_idx[torch.randint(len(fg_idx), (n,), generator=gen, device=device)]

    dirs = ray_dirs_world(ref_K, ref_c2w, h, w).to(device)[sel]
    origin = ref_c2w[:3, 3]
    # Conservative shell around the object center. This initializes front-view
    # pixels on plausible rays but still lets free means move anywhere.
    t = radius * (0.55 + 0.9 * torch.rand(n, 1, generator=gen, device=device))
    means = origin[None] + t * dirs
    means = means + 0.01 * radius * torch.randn(n, 3, generator=gen, device=device)

    colors = ref_img.reshape(-1, 3)[sel]
    scales = torch.full((n, 3), init_scale * radius, device=device)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((n,), init_opacity, device=device)

    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales.log()),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(_logit(opacities)),
        "colors": torch.nn.Parameter(_logit(colors)),
    })


def _render(params: torch.nn.ParameterDict, w2c: torch.Tensor, K: torch.Tensor,
            width: int, height: int, bg: float, packed: bool = True):
    from gsplat import rasterization

    colors, alphas, info = rasterization(
        means=params["means"],
        quats=F.normalize(params["quats"], dim=-1),
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.sigmoid(params["colors"]),
        viewmats=w2c,
        Ks=K,
        width=width,
        height=height,
        render_mode="RGB",
        packed=packed,
    )
    return (colors + (1.0 - alphas) * bg).clamp(0.0, 1.0), alphas, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--init_gaussians", type=int, default=8192)
    ap.add_argument("--train_views", type=int, default=4)
    ap.add_argument("--eval_views", type=int, default=0)       # 0 = all views
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_render_chunk", type=int, default=2)
    ap.add_argument("--lr_mean", type=float, default=1.6e-4)
    ap.add_argument("--lr_attr", type=float, default=2.5e-3)
    ap.add_argument("--init_scale_frac", type=float, default=0.01)
    ap.add_argument("--init_opacity", type=float, default=0.05)
    ap.add_argument("--fg_weight", type=float, default=10.0)
    ap.add_argument("--mask_weight", type=float, default=0.5)
    ap.add_argument("--refine_start", type=int, default=100)
    ap.add_argument("--refine_stop", type=int, default=900)
    ap.add_argument("--refine_every", type=int, default=50)
    ap.add_argument("--reset_every", type=int, default=3000)
    ap.add_argument("--max_gaussians", type=int, default=250000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_free_vram_gb", type=float, default=8.0)
    ap.add_argument("--out_dir", default="runs/densified_freegs")
    args = ap.parse_args()

    dev = "cuda"
    free_b, _ = torch.cuda.mem_get_info()
    free_gb = free_b / (1024 ** 3)
    if free_gb < args.min_free_vram_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB CUDA memory is free; refusing to start. "
            f"Lower --min_free_vram_gb or stop other GPU containers."
        )
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    from decoder.data import ObjaverseLatentDataset
    from decoder.render import _ssim
    from decoder.clean.losses import mask_alpha_l1
    from decoder.clean.metrics import fg_masked_psnr, sharpness_ratio
    from gsplat.strategy import DefaultStrategy

    ds = ObjaverseLatentDataset(V4, "train", manifest_path=MANIFEST)
    s = _pick_sample(ds, args.uid)
    w, h = s["width"], s["height"]
    w2c, K = s["w2c"].to(dev), s["K"].to(dev)
    target = s["frames"].to(dev)
    fg = s["masks"].to(dev)
    fgb = fg.bool()
    v_total = target.shape[0]
    v_eval = v_total if args.eval_views <= 0 else min(args.eval_views, v_total)

    params = _init_from_ref_view(
        s, args.init_gaussians, args.init_scale_frac, args.init_opacity, dev, args.seed
    )
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=args.lr_mean),
        "scales": torch.optim.Adam([params["scales"]], lr=args.lr_attr),
        "quats": torch.optim.Adam([params["quats"]], lr=args.lr_attr),
        "opacities": torch.optim.Adam([params["opacities"]], lr=args.lr_attr),
        "colors": torch.optim.Adam([params["colors"]], lr=args.lr_attr),
    }
    strategy = DefaultStrategy(
        refine_start_iter=args.refine_start,
        refine_stop_iter=args.refine_stop,
        refine_every=args.refine_every,
        reset_every=args.reset_every,
        verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=float(s["radius"]))

    def eval_all():
        rs, alphas = [], []
        with torch.no_grad():
            for i in range(0, v_eval, args.eval_render_chunk):
                r, a, _ = _render(params, w2c[i:i + args.eval_render_chunk],
                                  K[i:i + args.eval_render_chunk], w, h, bg=1.0)
                rs.append(r); alphas.append(a)
        render_all = torch.cat(rs, 0)
        alpha_all = torch.cat(alphas, 0)
        return (
            fg_masked_psnr(render_all, target[:v_eval], fgb[:v_eval]),
            sharpness_ratio(render_all, target[:v_eval], fgb[:v_eval].squeeze(-1)),
            float(alpha_all[0, ..., 0][fgb[0, ..., 0]].mean()),
            render_all,
            alpha_all,
        )

    print(f"[densified] uid={s['uid']} init={args.init_gaussians:,} views={args.train_views}/{v_total} "
          f"eval_views={v_eval} radius={float(s['radius']):.3f}", flush=True)

    for step in range(args.steps):
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        nv = min(args.train_views, v_total)
        idx = torch.cat([
            torch.zeros(1, dtype=torch.long, device=dev),
            torch.randperm(v_total - 1, device=dev)[:nv - 1] + 1,
        ])
        cur_bg = float(torch.rand(1))
        tgt = (target * fg + cur_bg * (1 - fg))[idx]
        render, alpha, info = _render(params, w2c[idx], K[idx], w, h, bg=cur_bg)
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        per_px = (render - tgt).abs().mean(-1, keepdim=True)
        wmap = 1.0 + (args.fg_weight - 1.0) * fg[idx]
        l1 = (per_px * wmap).sum() / (wmap.sum() + 1e-8)
        ssim = _ssim(render, tgt)
        mask = mask_alpha_l1(alpha, fg[idx])
        loss = l1 + 0.2 * (1 - ssim) + args.mask_weight * mask
        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)
        for opt in optimizers.values():
            opt.step()
        if len(params["means"]) > args.max_gaussians:
            print(f"[densified] hit max_gaussians={args.max_gaussians:,}; stopping", flush=True)
            break
        if step % args.eval_every == 0 or step == args.steps - 1:
            psnr, sharp, obj_alpha, _, _ = eval_all()
            op = torch.sigmoid(params["opacities"].detach())
            print(f"[densified] step {step:4d} N={len(params['means']):,} "
                  f"loss={float(loss.detach()):.4f} FG-PSNR={psnr:.2f} sharp={sharp:.3f} "
                  f"op={float(op.mean()):.4f} op99={float(op.quantile(0.99)):.3f} "
                  f"obj_alpha={obj_alpha:.3f}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    psnr, sharp, obj_alpha, render_all, alpha_all = eval_all()
    ckpt = {
        "params": {k: v.detach().cpu() for k, v in params.items()},
        "render": render_all.cpu(),
        "alpha": alpha_all.cpu(),
        "target": target[:v_eval].cpu(),
        "result": {
            "uid": s["uid"],
            "final_psnr": psnr,
            "sharpness": sharp,
            "obj_alpha": obj_alpha,
            "n_gaussians": len(params["means"]),
            "args": vars(args),
        },
    }
    torch.save(ckpt, out / "densified_freegs.pt")
    print(f"[densified] FINAL N={len(params['means']):,} FG-PSNR={psnr:.2f} "
          f"sharp={sharp:.3f} obj_alpha={obj_alpha:.3f}", flush=True)


if __name__ == "__main__":
    main()
