"""Decode stored LTX VAE latents to cached per-object conditioning frames.

The decoder trainer can then use these frames with:

  --condition_source fixed --cond_subdir ltx_decoded --cond_view_indices available

Frames are saved with original camera-view indices, e.g. a 9-frame decoded clip
from a 49-view orbit defaults to frame_000, frame_006, ..., frame_048.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from decoder.data import entry_relpath, entry_uid, latent_path_for_entry, object_dir_for_entry

def _add_ltx_paths(repo_root: Path) -> None:
    packages = repo_root / "vendor" / "LTX-2" / "packages"
    for rel in ("ltx-core/src", "ltx-trainer/src"):
        p = str(packages / rel)
        if p not in sys.path:
            sys.path.insert(0, p)


def _install_loader_stub() -> None:
    """Avoid the vendored LTX loader/fuse_loras circular import for VAE-only use."""
    name = "ltx_core.loader.fuse_loras"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)

    def _unused_apply_loras(*_args, **_kwargs):
        raise RuntimeError("apply_loras is not available in the VAE-only decode path")

    mod.apply_loras = _unused_apply_loras
    sys.modules[name] = mod


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _default_ltx23_vae_config(timestep_conditioning: bool = False) -> dict:
    # Reconstructed from the local LTX-2.3 VAE-only checkpoint shapes. This
    # avoids storing the full 46 GB DiT checkpoint when all we need is the VAE.
    return {
        "dims": 3,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 128,
        "patch_size": 4,
        "norm_layer": "pixel_norm",
        "spatial_padding_mode": "reflect",
        "causal_decoder": False,
        "timestep_conditioning": timestep_conditioning,
        "decoder_base_channels": 128,
        "decoder_blocks": [
            ["res_x", 4],
            ["compress_space", {"multiplier": 2}],
            ["res_x", 6],
            ["compress_time", {"multiplier": 2}],
            ["res_x", 4],
            ["compress_all", {"multiplier": 1}],
            ["res_x", 2],
            ["compress_all", {"multiplier": 2}],
            ["res_x", 2],
        ],
    }


def _dequantize_raw_vae_weights(raw: dict[str, torch.Tensor], key: str,
                                prefix: str, dtype: torch.dtype) -> torch.Tensor:
    value = raw[key]
    if value.dtype == torch.int8:
        scale = raw.get(f"{key}_scale")
        if scale is None:
            raise RuntimeError(f"int8 VAE weight is missing scale tensor: {key}_scale")
        view_shape = (scale.shape[0],) + (1,) * (value.ndim - 1)
        value = value.float() * scale.float().reshape(view_shape)
    return value.to(dtype=dtype)


def _load_raw_vae(model_path: str, config_json: str | None, raw_prefix: str,
                  raw_timestep_conditioning: bool,
                  device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    from safetensors.torch import load_file

    from ltx_core.model.video_vae import VideoDecoderConfigurator

    if config_json:
        cfg = json.loads(Path(config_json).read_text())
        vae_cfg = cfg.get("vae", cfg)
    else:
        vae_cfg = _default_ltx23_vae_config(
            timestep_conditioning=raw_timestep_conditioning
        )
    vae = VideoDecoderConfigurator.from_config({"vae": vae_cfg}).to(device=device, dtype=dtype)
    raw = load_file(model_path, device="cpu")
    sd = {}
    dec_prefix = f"{raw_prefix}vae.decoder."
    stat_prefix = f"{raw_prefix}vae.per_channel_statistics."
    for key, value in raw.items():
        if key.endswith("_scale"):
            continue
        if key.startswith(dec_prefix):
            sd[key.replace(dec_prefix, "", 1)] = _dequantize_raw_vae_weights(
                raw, key, raw_prefix, dtype
            )
        elif key.startswith(stat_prefix):
            sd[key.replace(stat_prefix, "per_channel_statistics.", 1)] = value.to(dtype=dtype)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected VAE keys: {unexpected[:8]}")
    if missing:
        print(f"[decode_ltx] warning: missing VAE keys: {missing[:8]}", flush=True)
    return vae.eval()


def _load_vae(model_path: str, config_json: str | None, raw_prefix: str,
              raw_timestep_conditioning: bool,
              device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    if raw_prefix:
        return _load_raw_vae(
            model_path, config_json, raw_prefix, raw_timestep_conditioning, device, dtype
        )
    from ltx_trainer.model_loader import load_video_vae_decoder

    return load_video_vae_decoder(model_path, device=device, dtype=dtype).eval()


def _entries(manifest_path: Path, split: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    if split == "all":
        out = []
        for key in ("train", "eval", "test"):
            out.extend(manifest.get(key, []))
        return out
    return manifest[split]


def _view_counts(entry, obj_dir: Path) -> tuple[int, int]:
    n_views = None
    n_orbit = None
    if isinstance(entry, dict):
        n_views = entry.get("num_views")
        n_orbit = entry.get("num_orbit_views")
    if n_views is None or n_orbit is None:
        cameras_path = obj_dir / "cameras.json"
        if cameras_path.exists():
            cameras = json.loads(cameras_path.read_text())
            frames = cameras.get("frames", [])
            n_views = n_views or len(frames)
            n_orbit = n_orbit or cameras.get("num_orbit_views")
    n_views = int(n_views or 49)
    n_orbit = int(n_orbit or min(n_views, 49))
    return n_views, n_orbit


def default_decode_view_indices(n_views: int, decoded_n: int,
                                n_orbit_views: int | None = None) -> list[int]:
    """Map decoded VAE frames back to the source frames used for encoding.

    LTX encodes the few input frames with inclusive endpoint spacing. For the
    current 49-frame orbit and 9-frame latent clip, that is
    [0, 6, 12, 18, 24, 30, 36, 42, 48]. Using the non-endpoint training anchor
    sampler here silently misregisters decoded RGB to the wrong cameras.
    """
    if decoded_n <= 0:
        return []
    limit = min(n_views, n_orbit_views or n_views)
    if decoded_n == 1:
        return [0]
    vals = torch.linspace(0, limit - 1, decoded_n).round().long().tolist()
    # Rounding linspace should be unique for the small decoded_n we use, but make
    # failures explicit instead of writing duplicate filenames.
    if len(set(vals)) != decoded_n:
        raise ValueError(
            f"decoded frame mapping is not unique: n_views={n_views} "
            f"n_orbit_views={n_orbit_views} decoded_n={decoded_n} -> {vals}"
        )
    return vals


def _save_frame(frame: torch.Tensor, path: Path) -> None:
    arr = (frame.float().cpu().numpy().clip(0.0, 1.0) * 255.0 + 0.5).astype("uint8")
    Image.fromarray(arr).save(path)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", default="all")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_config_json", default=None)
    ap.add_argument("--raw_prefix", default="")
    ap.add_argument("--raw_timestep_conditioning", type=int, default=0)
    ap.add_argument("--out_subdir", default="ltx_decoded")
    ap.add_argument("--out_root", default="",
                    help="If set, write <out_root>/<tier>/<uid>/<out_subdir>/ sidecars.")
    ap.add_argument("--view_indices", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=0,
                    help="Stop after writing this many new objects; skipped existing objects do not count.")
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = Path(args.dataset_root or Path.cwd())
    manifest_path = Path(args.manifest or dataset_root / "manifest.json")
    _add_ltx_paths(repo_root)
    _install_loader_stub()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    vae = _load_vae(
        args.model_path, args.model_config_json, args.raw_prefix,
        bool(args.raw_timestep_conditioning), device, dtype
    )

    entries = _entries(manifest_path, args.split)
    if args.offset > 0:
        entries = entries[args.offset:]
    if args.limit > 0:
        entries = entries[:args.limit]
    print(
        f"[decode_ltx] objects={len(entries)} out_subdir={args.out_subdir} "
        f"out_root={args.out_root or '<dataset>'} offset={args.offset} "
        f"max_new={args.max_new}",
        flush=True,
    )
    out_root = Path(args.out_root) if args.out_root else None

    written = 0
    skipped_existing = 0
    skipped_missing_latent = 0
    for n, entry in enumerate(entries, 1):
        uid = entry_uid(entry)
        obj_dir = object_dir_for_entry(dataset_root, entry)
        latent_path = latent_path_for_entry(dataset_root, entry, obj_dir)
        if not latent_path.exists():
            skipped_missing_latent += 1
            print(f"[decode_ltx] skip {uid[:8]}: missing latent.npy", flush=True)
            continue
        out_dir = (
            out_root / entry_relpath(entry) / args.out_subdir
            if out_root else obj_dir / args.out_subdir
        )
        meta_path = out_dir / "metadata.json"
        if meta_path.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"[decode_ltx] {n}/{len(entries)} skip {uid[:8]}: exists", flush=True)
            continue

        latent_np = np.load(latent_path).astype("float32")
        latent = torch.from_numpy(latent_np)[None].to(device=device, dtype=dtype)
        video = vae(latent)
        video = ((video[0].permute(1, 2, 3, 0) + 1.0) / 2.0).clamp(0.0, 1.0)
        decoded_n = int(video.shape[0])

        if args.view_indices:
            view_indices = [int(x) for x in args.view_indices.split(",") if x.strip()]
        else:
            n_views, n_orbit = _view_counts(entry, obj_dir)
            view_indices = default_decode_view_indices(n_views, decoded_n, n_orbit)
        if len(view_indices) != decoded_n:
            raise ValueError(
                f"{uid}: decoded {decoded_n} frames but view_indices has {len(view_indices)}"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for stale in out_dir.glob("frame_*.png"):
                stale.unlink()
        for frame, view_i in zip(video, view_indices):
            _save_frame(frame, out_dir / f"frame_{view_i:03d}.png")
        meta = {
            "uid": uid,
            "model_path": str(args.model_path),
            "latent_shape": list(latent_np.shape),
            "decoded_frames": decoded_n,
            "view_indices": view_indices,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        written += 1
        print(f"[decode_ltx] {n}/{len(entries)} wrote {uid[:8]} frames={view_indices}", flush=True)
        if args.max_new > 0 and written >= args.max_new:
            print(f"[decode_ltx] reached max_new={args.max_new}", flush=True)
            break
    print(
        f"[decode_ltx] done written={written} skipped_existing={skipped_existing} "
        f"skipped_missing_latent={skipped_missing_latent}",
        flush=True,
    )


if __name__ == "__main__":
    main()
