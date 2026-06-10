"""Training loop. v1 supports the overfit-one-object sanity gate.

Overfit success criterion (the gate before any multi-object run): L1 drops
>10x from step 0 and the re-rendered orbit is visually recognizable.
"""
from __future__ import annotations

from pathlib import Path

import torch

from decoder.data import ObjaverseLatentDataset
from .decoder_a import TokenAlignedDecoder
from .decoder_b import LearnedQueryDecoder
from .decoder_c import PixelAlignedDecoder
from .decoder_d import PixelAlignedDecoderD
from .decoder_e import PixelAlignedDecoderE
from decoder.render import photometric_loss, render_views


def build_decoder(arch: str, **kw) -> torch.nn.Module:
    if arch == "a":
        return TokenAlignedDecoder(**kw)
    if arch == "b":
        return LearnedQueryDecoder(**kw)
    if arch == "c":
        return PixelAlignedDecoder(**kw)
    if arch == "d":
        return PixelAlignedDecoderD(**kw)
    if arch == "e":
        return PixelAlignedDecoderE(**kw)
    raise ValueError(f"unknown arch {arch!r} (expected 'a', 'b', 'c', 'd', or 'e')")


def train_from_config(cfg) -> dict:
    """Canonical config-driven training/overfit loop (see legacy/config.py + legacy/configs/).

    Loss terms are toggled + weighted by cfg.loss (weight 0 = off); depth + GT-mask
    supervision are used when the dataset provides them (Trial R4 / animals_v3) and
    the config enables them. `overfit_one_object` below is the legacy kwargs CLI path
    (photometric only); this is the modular entry meant for the full pipeline.
    """
    import torch.nn.functional as F
    from decoder.data import depth_target_on_grid
    from decoder.render import depth_loss, foreground_mask

    d, mc, oc, lc = cfg.data, cfg.model, cfg.optim, cfg.loss
    dev = cfg.device
    ds = ObjaverseLatentDataset(d.dataset_root, split=d.split,
                                manifest_path=d.manifest, bg_variant=d.bg_variant,
                                load_depths=lc.depth > 0)
    idx = 0 if not d.uid else next(i for i, e in enumerate(ds.entries) if e["uid"] == d.uid)
    sample = ds[idx]
    target = sample["frames"].to(dev)
    w2c, K = sample["w2c"].to(dev), sample["K"].to(dev)
    w, h = sample["width"], sample["height"]
    radius = sample.get("radius") or float(sample["c2w_opengl"][0][:3, 3].norm())

    if "latent" in sample:
        latent = sample["latent"][None].to(dev)
    elif d.synthetic_latent:
        latent = torch.randn(1, 128, 2, 24, 16, device=dev)
        print("[train] WARNING: no latent on disk — using a SYNTHETIC latent (dry-run only).", flush=True)
    else:
        raise ValueError(f"object {sample['uid']} has no latent.npy (format-preview dataset). "
                         "Set data.synthetic_latent=true to dry-run the config.")

    pixel_aligned = mc.arch in ("c", "d", "e")
    decoder_kw: dict = {"radius": radius} if pixel_aligned else {"latent_t": latent.shape[2]}
    if mc.arch == "e":
        decoder_kw["opacity_mode"] = mc.opacity_mode
    model = build_decoder(mc.arch, **decoder_kw).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    if mc.arch in ("d", "e"):
        opt = torch.optim.AdamW(model.parameters(), lr=oc.lr, betas=(0.9, 0.95), weight_decay=0.05)
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=oc.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(oc.steps - oc.warmup_steps, 1))
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[oc.warmup_steps])
    else:
        opt, sched = torch.optim.Adam(model.parameters(), lr=oc.lr), None

    if pixel_aligned:
        ref_K, ref_c2w = K[0], sample["c2w_opengl"][0].to(dev)
        def run_model():
            return model(latent, ref_K, ref_c2w)
    else:
        def run_model():
            return model(latent)

    # foreground silhouette: exact GT masks (Trial R4+) when present + requested, else
    # the bg-color threshold. Used for the fg-weighted L1, BCE, random-bg recomposite.
    have_masks = "masks" in sample
    use_gt = lc.fg_source == "gt" and have_masks
    fg_full = sample["masks"].to(dev) if use_gt else foreground_mask(target, bg=d.bg)  # (V,H,W,1)

    # depth target on the decoder's Gaussian grid (reference frame 0), if depth loss on
    depth_tgt = depth_valid = None
    if lc.depth > 0 and "depth" in sample and mc.arch == "e":
        depth_tgt, depth_valid = depth_target_on_grid(
            sample["depth"][0].to(dev), fg_full[0, ..., 0], K[0], model.up_h, model.up_w)
        depth_tgt, depth_valid = depth_tgt.to(dev), depth_valid.to(dev)

    # background-anchored Gaussian mask for the masked opacity reg (frame 0)
    opacity_reg_mask = None
    if lc.opacity_reg > 0 and lc.opacity_reg_masked and mc.arch == "e":
        fg_grid = F.adaptive_avg_pool2d(fg_full[0].permute(2, 0, 1)[None], (model.up_h, model.up_w))[0, 0]
        opacity_reg_mask = (fg_grid.reshape(-1) < 0.5).float().to(dev)

    on = [k for k, v in [("l1", lc.l1), ("ssim", lc.ssim), ("mask", lc.mask_bce),
                         ("depth", depth_tgt is not None), ("opreg", lc.opacity_reg),
                         ("scalereg", lc.scale_reg)] if v]
    print(f"[train] uid={sample['uid']} arch={mc.arch} opacity={mc.opacity_mode} radius={radius:.3f} "
          f"views={target.shape[0]} n_params={n_params:,} fg={'gt' if use_gt else 'threshold'} "
          f"loss_on={on}", flush=True)

    history, first_l1 = [], 0.0
    for step in range(oc.steps):
        opt.zero_grad()
        cur_bg = float(torch.rand(1).item()) if d.random_bg else d.bg
        tgt = target * fg_full + cur_bg * (1.0 - fg_full) if d.random_bg else target
        params = {k: v[0] for k, v in run_model().items()}
        render, alpha = render_views(params, w2c, K, w, h, bg=cur_bg)
        loss, comp = photometric_loss(
            render, tgt, params, alpha=(alpha if lc.mask_bce > 0 else None), bg=cur_bg,
            l1_weight=lc.l1, ssim_weight=lc.ssim, mask_weight=lc.mask_bce,
            scale_reg_weight=lc.scale_reg, fg_weight=lc.fg_weight,
            opacity_reg_weight=lc.opacity_reg,
            fg_mask=(fg_full if use_gt else None), opacity_reg_mask=opacity_reg_mask)
        if depth_tgt is not None:
            dl = depth_loss(params["depth"], depth_tgt, depth_valid, delta=lc.depth_delta)
            loss = loss + lc.depth * dl
            comp["depth"] = dl.detach()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), oc.grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
        if step == 0:
            first_l1 = comp["l1"].item()
        if step % oc.log_every == 0 or step == oc.steps - 1:
            with torch.no_grad():
                op = params["opacity"]
                rec = {"step": step, **{k: float(v) for k, v in comp.items()},
                       "n_gaussians": params["mean"].shape[0],
                       "opacity_mean": float(op.mean()), "alive_frac": float((op > 0.01).float().mean()),
                       "scale_mean": float(params["scale"].mean()), "render_std": float(render.std()),
                       "mean_absmax": float(params["mean"].abs().max()), "grad_norm": float(grad_norm)}
            history.append(rec)
            print(f"[train] step {step:4d} l1={rec['l1']:.4f} psnr={rec['psnr']:.2f} ssim={rec['ssim']:.3f} "
                  f"mask={rec.get('mask', 0):.3f} depth={rec.get('depth', 0):.4f} op={rec['opacity_mean']:.3f} "
                  f"alive={rec['alive_frac']:.3f} sc={rec['scale_mean']:.4f} rstd={rec['render_std']:.3f} "
                  f"grad={rec['grad_norm']:.2e}", flush=True)

    last_l1 = history[-1]["l1"]
    result = {"arch": mc.arch, "uid": sample["uid"], "n_params": n_params,
              "first_l1": first_l1, "last_l1": last_l1,
              "l1_drop_ratio": first_l1 / max(last_l1, 1e-8),
              "final_psnr": history[-1]["psnr"], "history": history}
    if cfg.out_dir:
        out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            params = {k: v[0] for k, v in run_model().items()}
            render, alpha = render_views(params, w2c, K, w, h, bg=d.bg)
        torch.save({"render": render.cpu(), "alpha": alpha.cpu(), "target": target.cpu(),
                    "result": result}, out / f"overfit_{mc.arch}.pt")
    return result


def overfit_one_object(
    dataset_root: str | Path,
    arch: str = "a",
    steps: int = 1500,
    lr: float = 3e-4,
    grad_clip: float = 1.0,
    device: str = "cuda",
    log_every: int = 50,
    out_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    bg: float = 0.0,
    mask_weight: float = 0.0,
    fg_weight: float = 10.0,
    scale_reg_weight: float = 0.0,
    opacity_mode: str = "pdf",
    opacity_reg_weight: float = 0.0,
    random_bg: bool = False,
    uid: str = "",
) -> dict:
    """Train a decoder to memorize a single object's orbit views.

    Input latent encodes the orbit (animals_v1: 49 frames → T=7); supervision
    renders the predicted 3DGS at ALL the dataset's orbit cameras and compares
    to the rendered frames. `bg` is the dataset background (animals_v1 = 0 black).
    Camera radius for the pixel-aligned decoder is measured from the data.
    """
    ds = ObjaverseLatentDataset(dataset_root, split="train", manifest_path=manifest_path)
    idx = 0
    if uid:
        idx = next(i for i, e in enumerate(ds.entries) if e["uid"] == uid)
    sample = ds[idx]
    latent = sample["latent"][None].to(device)
    target = sample["frames"].to(device)
    w2c = sample["w2c"].to(device)
    K = sample["K"].to(device)
    w, h = sample["width"], sample["height"]
    latent_t = latent.shape[2]   # temporal dim from the data (animals_v1=7)
    # radius = distance from camera center to origin (object is centered there)
    radius = float(sample["c2w_opengl"][0][:3, 3].norm())

    # A/B size their token grid by T; C/D are pixel-aligned (T-agnostic) and
    # take the data-measured radius to set their depth shell.
    pixel_aligned = arch in ("c", "d", "e")
    decoder_kw: dict = {"radius": radius} if pixel_aligned else {"latent_t": latent_t}
    if arch == "e":
        decoder_kw["opacity_mode"] = opacity_mode
    model = build_decoder(arch, **decoder_kw).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # AdamW + linear warmup (GS-LRM/Lyra recipe) — warmup so early steps can't
    # overshoot. Plain Adam for the legacy A/B/C paths.
    if arch in ("d", "e"):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        warmup_steps = 100
        sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=warmup_steps)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = None
    print(f"[overfit-{arch}] uid={sample['uid']} latent_t={latent_t} radius={radius:.3f} "
          f"views={target.shape[0]} bg={bg} n_params={n_params:,}", flush=True)

    # C/D are pixel-aligned to a reference frame (frame 0); they need that
    # frame's intrinsics + OpenGL c2w to unproject. A/B take only the latent.
    if pixel_aligned:
        ref_K = K[0]
        ref_c2w = sample["c2w_opengl"][0].to(device)
        def run_model():
            return model(latent, ref_K, ref_c2w)
    else:
        def run_model():
            return model(latent)

    # Foreground mask (object pixels) from the fixed-bg targets — used to
    # recomposite onto a RANDOM background each iter when random_bg=True. This
    # attacks the "render-nothing matches the background" trivial minimum
    # (Faster-GS): a varying bg forces Gaussians to explain object color, not
    # hide in a constant background. The target's true bg is `bg` (e.g. black).
    from decoder.render import foreground_mask
    fg = foreground_mask(target, bg=bg)  # (V,H,W,1) 1=object

    history = []
    first_l1 = 0.0
    for step in range(steps):
        opt.zero_grad()
        cur_bg = float(torch.rand(1).item()) if random_bg else bg
        # recomposite target onto the current bg so render and target share it
        tgt = target * fg + cur_bg * (1.0 - fg) if random_bg else target
        params = {k: v[0] for k, v in run_model().items()}  # drop batch dim -> (N,...)
        render, alpha = render_views(params, w2c, K, w, h, bg=cur_bg)
        loss, comp = photometric_loss(render, tgt, params, alpha=alpha, bg=cur_bg,
                                      mask_weight=mask_weight, fg_weight=fg_weight,
                                      scale_reg_weight=scale_reg_weight,
                                      opacity_reg_weight=opacity_reg_weight)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
        if step == 0:
            first_l1 = comp["l1"].item()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                op = params["opacity"]
                rec = {"step": step, **{k: float(v) for k, v in comp.items()},
                       "n_gaussians": params["mean"].shape[0],
                       "opacity_mean": float(op.mean()),
                       "alive_frac": float((op > 0.01).float().mean()),
                       "scale_mean": float(params["scale"].mean()),
                       "render_mean": float(render.mean()),
                       "render_std": float(render.std()),
                       "mean_absmax": float(params["mean"].abs().max()),
                       "grad_norm": float(grad_norm)}
            history.append(rec)
            print(f"[overfit-{arch}] step {step:4d} l1={rec['l1']:.4f} "
                  f"psnr={rec['psnr']:.2f} ssim={rec['ssim']:.3f} mask={rec.get('mask', 0):.3f} "
                  f"op={rec['opacity_mean']:.3f} alive={rec['alive_frac']:.3f} "
                  f"sc={rec['scale_mean']:.4f} rstd={rec['render_std']:.3f} "
                  f"|mu|max={rec['mean_absmax']:.2f} grad={rec['grad_norm']:.2e}", flush=True)

    last_l1 = history[-1]["l1"]
    result = {
        "arch": arch, "uid": sample["uid"], "n_params": n_params,
        "first_l1": first_l1, "last_l1": last_l1,
        "l1_drop_ratio": first_l1 / max(last_l1, 1e-8),
        "final_psnr": history[-1]["psnr"], "history": history,
    }
    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            params = {k: v[0] for k, v in run_model().items()}
            render, alpha = render_views(params, w2c, K, w, h, bg=bg)
        torch.save({"render": render.cpu(), "alpha": alpha.cpu(), "target": target.cpu(),
                    "result": result}, out / f"overfit_{arch}.pt")
    return result


def overfit_freegs(
    dataset_root: str | Path,
    n_gaussians: int = 30000,
    steps: int = 1000,
    device: str = "cuda",
    log_every: int = 50,
    out_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    bg: float = 0.0,
    fg_weight: float = 10.0,
    uid: str = "",
) -> dict:
    """Decoder-FREE baseline: optimize raw Gaussian params directly, per the
    canonical gsplat simple_trainer recipe (log-scale + logit-opacity, separate
    per-parameter Adam groups). Decisive sanity check: if free Gaussians render
    the object, the renderer/camera/loss/data are correct and the problem is the
    decoder parameterization (shared output layer collapses all Gaussians at
    once). If this also fails, the problem is upstream.
    """
    from decoder.render import photometric_loss, render_views

    ds = ObjaverseLatentDataset(dataset_root, split="train", manifest_path=manifest_path)
    idx = 0 if not uid else next(i for i, e in enumerate(ds.entries) if e["uid"] == uid)
    sample = ds[idx]
    target = sample["frames"].to(device)
    w2c, K = sample["w2c"].to(device), sample["K"].to(device)
    w, h = sample["width"], sample["height"]

    N = n_gaussians
    g = torch.Generator(device=device).manual_seed(0)
    # gsplat init: means random in a ball ~object size; log-scale ~exp(-3)≈0.05;
    # logit-opacity 0 -> sigmoid 0.5; colors small random. Object is a unit-ish
    # sphere at origin, so init means within radius ~1.0 of origin.
    raw = {
        "means": torch.nn.Parameter((torch.rand(N, 3, generator=g, device=device) - 0.5) * 1.2),
        "scales": torch.nn.Parameter(torch.full((N, 3), -3.0, device=device)
                                     + 0.1 * torch.randn(N, 3, generator=g, device=device)),
        "quats": torch.nn.Parameter(torch.randn(N, 4, generator=g, device=device)),
        "opacities": torch.nn.Parameter(torch.zeros(N, device=device)),
        "colors": torch.nn.Parameter(0.5 + 0.1 * torch.randn(N, 3, generator=g, device=device)),
    }
    # per-parameter-group Adam (canonical 3DGS LRs: means low, others higher)
    opt = torch.optim.Adam([
        {"params": [raw["means"]], "lr": 1.6e-4},
        {"params": [raw["scales"]], "lr": 5e-3},
        {"params": [raw["quats"]], "lr": 1e-3},
        {"params": [raw["opacities"]], "lr": 5e-2},
        {"params": [raw["colors"]], "lr": 2.5e-3},
    ])

    def activate():
        return {
            "mean": raw["means"],
            "quat": torch.nn.functional.normalize(raw["quats"], dim=-1),
            "scale": torch.exp(raw["scales"]),
            "opacity": torch.sigmoid(raw["opacities"]).unsqueeze(-1),
            "rgb": torch.sigmoid(raw["colors"]),
        }

    history, first_l1 = [], 0.0
    for step in range(steps):
        opt.zero_grad()
        params = activate()
        render, alpha = render_views(params, w2c, K, w, h, bg=bg)
        loss, comp = photometric_loss(render, target, params, alpha=None, bg=bg,
                                      mask_weight=0.0, fg_weight=fg_weight,
                                      scale_reg_weight=0.0)
        loss.backward()
        opt.step()
        if step == 0:
            first_l1 = comp["l1"].item()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                op = params["opacity"]
                rec = {"step": step, **{k: float(v) for k, v in comp.items()},
                       "opacity_mean": float(op.mean()),
                       "alive_frac": float((op > 0.01).float().mean()),
                       "scale_mean": float(params["scale"].mean()),
                       "render_std": float(render.std())}
            history.append(rec)
            print(f"[freegs] step {step:4d} l1={rec['l1']:.4f} psnr={rec['psnr']:.2f} "
                  f"ssim={rec['ssim']:.3f} op={rec['opacity_mean']:.3f} "
                  f"alive={rec['alive_frac']:.3f} sc={rec['scale_mean']:.4f} "
                  f"rstd={rec['render_std']:.3f}", flush=True)

    last_l1 = history[-1]["l1"]
    result = {"mode": "freegs", "uid": sample["uid"], "n_gaussians": N,
              "first_l1": first_l1, "last_l1": last_l1,
              "l1_drop_ratio": first_l1 / max(last_l1, 1e-8),
              "final_psnr": history[-1]["psnr"], "history": history}
    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            render, alpha = render_views(activate(), w2c, K, w, h, bg=bg)
        torch.save({"render": render.cpu(), "alpha": alpha.cpu(), "target": target.cpu(),
                    "result": result}, out / "overfit_freegs.pt")
    return result
