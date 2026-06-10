"""Modal harness for the current Phase 1/2/3 decoder experiments.

This targets Modal RTX PRO 6000 Blackwell and the objaverse_v5_final dataset
with the separate depth archive.

Typical setup:

  modal run modal_phase123.py --action upload
  modal run modal_phase123.py --action prepare
  modal run modal_phase123.py --action smoke

Short training:

  modal run modal_phase123.py --action phase1_depth_probe --steps 50
  modal run modal_phase123.py --action phase3_offset_probe --steps 50
  modal run modal_phase123.py --action phase1_depth_pilot
  modal run modal_phase123.py --action phase3_offset_pilot
  modal run modal_phase123.py --action densified_smoke

Custom:

  modal run modal_phase123.py --action train --extra "--steps 1000 ..."
"""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import modal

APP_NAME = "latent2splat-phase123"
GPU = "RTX-PRO-6000"

DATA_DIR = "/data"
RENDERS_DIR = "/renders"
LATENTS_V8_DIR = "/latents_v8"
CACHE_DIR = "/cache"
RUNS_DIR = "/runs"
WEIGHTS_DIR = "/weights"
WORKDIR = "/workspace/latent2splat"
DATASET = "objaverse_v5_final"
MANIFEST = "manifest.json"
LTX_CKPT = f"{WEIGHTS_DIR}/ltx23/ltx-2.3-22b-distilled.safetensors"
LTX_RAW_VAE = f"{WEIGHTS_DIR}/ltx23/plain_int8-vae-bf16-backup.safetensors"
LTX_JASON_CKPT = f"{WEIGHTS_DIR}/ltx-2.3-22b-distilled-1.1.safetensors"
HF_LTX_REPO = "Lightricks/LTX-2.3"
HF_LTX_FILE = "ltx-2.3-22b-distilled.safetensors"

LOCAL_DATA_ROOT = Path("/mnt/data1/latent2splat-datasets")
LOCAL_RGB_ZIP = LOCAL_DATA_ROOT / "downloads" / "objaverse_v5_final.zip"
LOCAL_DEPTH_ZIP = LOCAL_DATA_ROOT / "downloads" / "objaverse_v5_depth.zip"
LOCAL_MANIFEST = LOCAL_DATA_ROOT / DATASET / MANIFEST
LOCAL_LTX_RAW_VAE = Path("/mnt/data1/weights/ltx2-tpu/denoise/transformer/plain_int8-vae-bf16-backup.safetensors")
DATA_VOLUME_NAME = os.environ.get("L2S_PHASE_DATA_VOLUME", "latent2splat-phase123-data")
RENDERS_VOLUME_NAME = os.environ.get("L2S_RENDERS_VOLUME", "latent2splat-renders")
CACHE_VOLUME_NAME = os.environ.get("L2S_PHASE_CACHE_VOLUME", "latent2splat-phase123-cache")
RUNS_VOLUME_NAME = os.environ.get("L2S_PHASE_RUNS_VOLUME", "latent2splat-phase123-runs")
WEIGHTS_VOLUME_NAME = os.environ.get("L2S_WEIGHTS_VOLUME", "latent2splat-weights")
LATENTS_V8_VOLUME_NAME = os.environ.get("L2S_LATENTS_V8_VOLUME", "latent2splat-renders-v8")

repo = Path(__file__).resolve().parent


def _is_combined_dataset(dataset: str) -> bool:
    return dataset in {"combined_v7_v8", "v7_v8", "objaverse_v7_v8"}


def _requested_action() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--action" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--action="):
            return arg.split("=", 1)[1]
    return "info"


REQUESTED_ACTION = _requested_action()
NEEDS_LTX_IMAGE = REQUESTED_ACTION.startswith("decode_ltx")
NEEDS_DEPTH_ANYTHING_IMAGE = REQUESTED_ACTION == "predict_depth"
NEEDS_VGGT_IMAGE = REQUESTED_ACTION == "predict_vggt_depth"
NEEDS_DA3_IMAGE = REQUESTED_ACTION.startswith("predict_da3_depth")
NEEDS_SPCONV_IMAGE = (
    REQUESTED_ACTION in {"spconv_smoke", "sparse_voxel_fusion_smoke"}
    or REQUESTED_ACTION.startswith("quality_sparse_voxel")
)


def _ignore_python_cache(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
renders_volume = modal.Volume.from_name(RENDERS_VOLUME_NAME, create_if_missing=False)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
latents_v8_volume = modal.Volume.from_name(LATENTS_V8_VOLUME_NAME, create_if_missing=True)

utility_image = modal.Image.debian_slim(python_version="3.12")
hf_utility_image = utility_image.pip_install("huggingface_hub[hf_transfer]").env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
)
inspect_image = (
    utility_image
    .pip_install("numpy==1.26.4", "Pillow")
    .add_local_dir(str(repo / "decoder"), f"{WORKDIR}/decoder", copy=True,
                   ignore=_ignore_python_cache)
)

decoder_base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("build-essential", "git", "libgl1", "libglib2.0-0", "ninja-build")
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands(
        "python -m pip install --no-cache-dir "
        "torch==2.10.0+cu130 torchvision==0.25.0+cu130 "
        "--extra-index-url https://download.pytorch.org/whl/cu130"
    )
    .run_commands(
        "TORCH_CUDA_ARCH_LIST=12.0 python -m pip install "
        "--no-cache-dir --no-build-isolation gsplat==1.4.0"
    )
    .pip_install("numpy==1.26.4", "Pillow", "pytest", "scipy", "tqdm", "wandb")
    .env(
        {
            "PYTHONPATH": WORKDIR,
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MAX_JOBS": "1",
            "CMAKE_BUILD_PARALLEL_LEVEL": "1",
            "NINJAFLAGS": "-j1",
            "TORCH_EXTENSIONS_DIR": f"{CACHE_DIR}/torch-extensions",
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "WANDB_DIR": f"{RUNS_DIR}/wandb",
            "PHASE2_DATA_ROOT": f"{DATA_DIR}/{DATASET}",
            "PHASE2_MANIFEST": f"{DATA_DIR}/{DATASET}/{MANIFEST}",
            "L2S_LATENTS_V8_ROOT": LATENTS_V8_DIR,
        }
    )
)

decoder_image = (
    decoder_base_image
    .add_local_dir(str(repo / "decoder"), f"{WORKDIR}/decoder", copy=True,
                   ignore=_ignore_python_cache)
)

ltx_decode_image = (
    decoder_base_image
    .pip_install("einops", "safetensors", "transformers==4.57.6", "accelerate")
    .add_local_dir(str(repo / "decoder"), f"{WORKDIR}/decoder", copy=True,
                   ignore=_ignore_python_cache)
    .add_local_dir(str(repo / "vendor" / "LTX-2"), f"{WORKDIR}/vendor/LTX-2", copy=True)
    .add_local_file(
        str(repo / "infra" / "experimental" / "patch_ltx_circular_import.py"),
        "/opt/patch_ltx_circular_import.py",
        copy=True,
    )
    .run_commands(f"python /opt/patch_ltx_circular_import.py {WORKDIR}/vendor/LTX-2")
) if NEEDS_LTX_IMAGE else utility_image

depth_prior_image = (
    decoder_base_image
    .pip_install("transformers==4.57.6", "accelerate", "safetensors")
    .add_local_dir(str(repo / "decoder"), f"{WORKDIR}/decoder", copy=True,
                   ignore=_ignore_python_cache)
) if NEEDS_DEPTH_ANYTHING_IMAGE else utility_image
vggt_depth_image = (
    decoder_base_image
    .pip_install("git+https://github.com/facebookresearch/vggt.git")
    .add_local_dir(str(repo / "decoder"), f"{WORKDIR}/decoder", copy=True,
                   ignore=_ignore_python_cache)
) if NEEDS_VGGT_IMAGE else utility_image
da3_depth_image = (
    decoder_base_image
    .pip_install("git+https://github.com/ByteDance-Seed/Depth-Anything-3.git")
    .add_local_dir(str(repo / "decoder"), f"{WORKDIR}/decoder", copy=True,
                   ignore=_ignore_python_cache)
) if NEEDS_DA3_IMAGE else utility_image

# Sparse-voxel 3D conv smoke image: cu126 prebuilt + NVRTC sm_120 fallback.
# Kept separate from decoder_image so a smoke failure doesn't bust caching.
spconv_smoke_image = (
    decoder_base_image
    .pip_install("spconv-cu126")
) if NEEDS_SPCONV_IMAGE else utility_image

# Full decoder image + spconv: the image the sparse-voxel fusion module
# smoke and (future) training runs use.  Inherits gsplat/etc from decoder_image.
decoder_spconv_image = (
    decoder_image
    .pip_install("spconv-cu126")
) if NEEDS_SPCONV_IMAGE else utility_image

app = modal.App(APP_NAME, image=utility_image)


def _volumes() -> dict[str, modal.Volume]:
    vols = {
        DATA_DIR: data_volume,
        LATENTS_V8_DIR: latents_v8_volume,
        CACHE_DIR: cache_volume,
        RUNS_DIR: runs_volume,
    }
    if RENDERS_VOLUME_NAME != DATA_VOLUME_NAME:
        vols[RENDERS_DIR] = renders_volume
    return vols


def _renders_mount_dir() -> str:
    return DATA_DIR if RENDERS_VOLUME_NAME == DATA_VOLUME_NAME else RENDERS_DIR


def _dataset_root(dataset: str) -> str:
    if _is_combined_dataset(dataset):
        return _renders_mount_dir()
    return f"{DATA_DIR}/{dataset}"


def _dataset_manifest(dataset: str, manifest: str = MANIFEST) -> str:
    if Path(manifest).is_absolute():
        return manifest
    if _is_combined_dataset(dataset):
        if manifest == MANIFEST:
            return f"{RUNS_DIR}/manifests/combined_v7_v8.json"
        m = manifest
        if m.startswith("data/"):
            m = m[len("data/"):]
        return f"{DATA_DIR}/{m}"
    return f"{_dataset_root(dataset)}/{manifest}"


def _run_module(module: str, args: list[str], dataset: str = DATASET,
                manifest: str = MANIFEST) -> None:
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PHASE2_DATA_ROOT"] = _dataset_root(dataset)
    env["PHASE2_MANIFEST"] = _dataset_manifest(dataset, manifest)
    env["L2S_LATENTS_V8_ROOT"] = LATENTS_V8_DIR
    env["L2S_COND_OVERRIDE_ROOT"] = f"{RUNS_DIR}/condition"
    env["WANDB_DIR"] = f"{RUNS_DIR}/wandb"
    # Depth-anchor baked depths live on the runs volume when render volume is full.
    depth_override = (
        f"{RUNS_DIR}/depth" if _is_combined_dataset(dataset)
        else f"{RUNS_DIR}/da3_anchor_v7"
    )
    env.setdefault("L2S_DEPTH_OVERRIDE_ROOT", depth_override)
    if env.get("TORCH_EXTENSIONS_DIR"):
        os.makedirs(env["TORCH_EXTENSIONS_DIR"], exist_ok=True)
    cmd = [sys.executable, "-m", module, *args]
    print("[modal_phase123] run:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=WORKDIR, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.function(image=utility_image, volumes={DATA_DIR: data_volume}, timeout=60 * 60, cpu=4, memory=8192)
def prepare_dataset(force: bool = False, dataset: str = DATASET,
                    archive_name: str = "", depth_archive_name: str = "",
                    manifest_name: str = MANIFEST) -> dict:
    """Extract RGB/depth archives already uploaded to the Modal data volume."""
    import json
    import shutil
    import zipfile
    from pathlib import Path

    def _extract_archive(zip_path: Path, dst_root: Path) -> None:
        """Extract either a rootless dataset archive or one containing dataset/."""
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if n and not n.endswith("/")]
            tops = {n.split("/", 1)[0] for n in names if "/" in n}
            nested = len(tops) == 1 and any(n.endswith(f"/{manifest_name}") for n in names)
            if nested:
                tmp = Path(DATA_DIR) / f"_extract_{dataset}"
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True)
                z.extractall(tmp)
                only = tmp / next(iter(tops))
                for p in only.iterdir():
                    shutil.move(str(p), dst_root / p.name)
                shutil.rmtree(tmp)
            else:
                z.extractall(dst_root)

    root = Path(DATA_DIR) / dataset
    uploads = Path(DATA_DIR) / "uploads"
    archive_name = archive_name or "objaverse_v5_final.zip"
    depth_archive_name = depth_archive_name or ("objaverse_v5_depth.zip" if dataset == DATASET else "")
    rgb_zip = uploads / archive_name
    depth_zip = uploads / depth_archive_name if depth_archive_name else None
    manifest_src = uploads / manifest_name
    if not rgb_zip.exists():
        raise FileNotFoundError(f"missing uploaded archive: {rgb_zip}")
    if force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    if force or not (root / manifest_name).exists():
        print(f"[prepare] extracting {rgb_zip.name} -> {root}", flush=True)
        _extract_archive(rgb_zip, root)
    if depth_zip and depth_zip.exists() and not any(root.glob("*/depth/depth_*.npy")):
        print(f"[prepare] extracting depth archive {depth_zip.name}", flush=True)
        _extract_archive(depth_zip, root)
    if manifest_src.exists():
        shutil.copyfile(manifest_src, root / manifest_name)
    if not (root / manifest_name).exists():
        raise FileNotFoundError(f"manifest not found after extraction: {root / manifest_name}")

    manifest = json.loads((root / manifest_name).read_text())
    n_objs = sum(len(v) for v in manifest.values())
    n_depth = len(list(root.glob("*/depth/depth_*.npy")))
    n_ltx = len(list(root.glob("*/ltx_decoded/frame_*.png")))
    n_da3 = len(list(root.glob("*/da3_ltx/depth_*.npy")))
    data_volume.commit()
    out = {
        "root": str(root),
        "objects": n_objs,
        "splits": {k: len(v) for k, v in manifest.items()},
        "depth_files": n_depth,
        "ltx_decoded_frames": n_ltx,
        "da3_depth_files": n_da3,
    }
    print("[prepare]", out, flush=True)
    return out


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=45 * 60, cpu=8, memory=98304)
def smoke(dataset: str = DATASET, manifest: str = MANIFEST) -> None:
    """GPU smoke for image, data, gsplat, and a one-step depth run."""
    _run_module(
        "pytest",
        [
            "decoder/tests/test_clean_gaussians.py",
            "decoder/tests/test_clean_network.py",
            "decoder/tests/test_clean_render.py",
            "-q",
        ],
    )
    _run_module(
        "decoder.clean.train_phase2",
        [
            "--steps", "1",
            "--accum", "1",
            "--k_views", "1",
            "--ups_stages", "2",
            "--num_workers", "0",
            "--n_train_eval", "1",
            "--n_heldout_eval", "1",
            "--eval_render_chunk", "1",
            "--eval_at_step0", "0",
            "--log_every", "1",
            "--wandb_mode", "disabled",
            "--perceptual_weight", "0",
            "--depth_weight", "1.0",
            "--depth_abs_weight", "0.05",
            "--depth_render_mode", "ED",
            "--depth_render_scale", "0.25",
            "--min_free_vram_gb", "40",
            "--out_dir", f"{RUNS_DIR}/smoke_depth",
        ],
        dataset=dataset,
        manifest=manifest,
    )
    cache_volume.commit()
    runs_volume.commit()


@app.function(image=hf_utility_image, volumes={WEIGHTS_DIR: weights_volume},
              timeout=2 * 60 * 60, cpu=4, memory=8192)
def download_ltx_checkpoint() -> dict:
    """Download only the public LTX checkpoint needed for VAE decode, not Gemma."""
    import os
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    out_dir = Path(WEIGHTS_DIR) / "ltx23"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / HF_LTX_FILE
    if out.exists():
        result = {"status": "exists", "path": str(out), "size_gb": out.stat().st_size / 1e9}
    else:
        hf_hub_download(repo_id=HF_LTX_REPO, filename=HF_LTX_FILE, local_dir=str(out_dir))
        result = {"status": "downloaded", "path": str(out), "size_gb": os.path.getsize(out) / 1e9}
    weights_volume.commit()
    print("[download_ltx_checkpoint]", result, flush=True)
    return result


@app.function(image=decoder_image, volumes={CACHE_DIR: cache_volume},
              timeout=2 * 60 * 60, cpu=16, memory=131072)
def compile_gsplat() -> dict:
    """Compile gsplat CUDA extensions into the shared cache volume without a GPU."""
    import os
    import time
    from pathlib import Path

    ext_dir = Path(CACHE_DIR) / "torch-extensions"
    ext_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    os.environ["MAX_JOBS"] = "4"
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "4"
    os.environ["NINJAFLAGS"] = "-j4"
    t0 = time.time()
    import torch
    print(
        f"[compile_gsplat] torch={torch.__version__} "
        f"cuda={torch.version.cuda} ext_dir={ext_dir}",
        flush=True,
    )
    from gsplat.cuda import _backend  # noqa: F401

    elapsed = round(time.time() - t0, 1)
    files = sum(1 for p in ext_dir.rglob("*") if p.is_file())
    cache_volume.commit()
    out = {"elapsed_sec": elapsed, "ext_dir": str(ext_dir), "files": files}
    print("[compile_gsplat]", out, flush=True)
    return out


@app.function(image=spconv_smoke_image, gpu=GPU, timeout=15 * 60, cpu=4, memory=16384)
def spconv_smoke() -> dict:
    """Verify spconv builds/runs on sm_120 (Blackwell RTX PRO 6000).

    Uses the cu126 prebuilt wheel and relies on spconv's NVRTC runtime
    fallback to compile sm_120 kernels on first use. This gates whether the
    sparse-voxel 3D fusion decoder is buildable on our hardware.
    """
    import time
    import traceback

    out: dict = {"ok": False, "step": "start"}
    try:
        import torch

        out["torch"] = torch.__version__
        out["torch_cuda"] = torch.version.cuda
        out["device_cap"] = list(torch.cuda.get_device_capability())
        out["device_name"] = torch.cuda.get_device_name(0)

        out["step"] = "import_spconv"
        import spconv.pytorch as spconv
        from spconv.pytorch import SparseConvTensor, SubMConv3d

        out["spconv_version"] = getattr(spconv, "__version__", "unknown")

        out["step"] = "build_sparse_tensor"
        device = "cuda"
        N = 5000
        torch.manual_seed(0)
        feats = torch.randn(N, 16, device=device, requires_grad=True)
        coords = torch.zeros(N, 4, dtype=torch.int32, device=device)
        coords[:, 1:] = torch.randint(0, 64, (N, 3), dtype=torch.int32, device=device)
        sp = SparseConvTensor(feats, coords, spatial_shape=[64, 64, 64], batch_size=1)
        out["in_features"] = list(sp.features.shape)

        out["step"] = "forward"
        conv = SubMConv3d(16, 32, kernel_size=3, bias=False).to(device)
        t0 = time.time()
        y = conv(sp)
        torch.cuda.synchronize()
        out["forward_sec"] = round(time.time() - t0, 3)
        out["out_features"] = list(y.features.shape)

        out["step"] = "backward"
        t1 = time.time()
        y.features.sum().backward()
        torch.cuda.synchronize()
        out["backward_sec"] = round(time.time() - t1, 3)

        out["step"] = "done"
        out["ok"] = True
    except Exception as exc:
        out["error"] = repr(exc)
        out["traceback"] = traceback.format_exc()

    print("[spconv_smoke]", out, flush=True)
    return out


@app.function(image=decoder_spconv_image, gpu=GPU, volumes=_volumes(), timeout=12 * 60 * 60, cpu=12, memory=98304)
def train_sparse_voxel(args: list[str], dataset: str = DATASET, manifest: str = MANIFEST) -> None:
    """Training entrypoint with spconv-enabled image (for --use_sparse_voxel_fusion runs)."""
    _run_module("decoder.clean.train_phase2", args, dataset=dataset, manifest=manifest)
    cache_volume.commit()
    runs_volume.commit()


@app.function(image=decoder_spconv_image, gpu=GPU, timeout=15 * 60, cpu=4, memory=16384)
def sparse_voxel_fusion_smoke() -> dict:
    """Smoke-test the SparseVoxelFusion module: instantiate + forward + verify
    step-0 ≈ prior on a synthetic fused dict (the non-negotiable invariant)."""
    import time
    import traceback

    out: dict = {"ok": False, "step": "start"}
    try:
        import torch
        out["torch"] = torch.__version__
        out["device_cap"] = list(torch.cuda.get_device_capability())

        out["step"] = "import_module"
        from decoder.clean.sparse_voxel_fusion import SparseVoxelFusion

        out["step"] = "instantiate"
        head = SparseVoxelFusion(hidden=32).cuda()
        head.eval()
        n_params = sum(p.numel() for p in head.parameters())
        out["param_count"] = n_params

        out["step"] = "build_synthetic_fused_dict"
        # Synthetic ~5000-voxel fused dict, world coords in radius=1 box.
        M = 5000
        torch.manual_seed(0)
        radius = 1.0
        voxel_size = 0.012  # ~83^3 grid; ~5000 cells inside
        mean = torch.randn(M, 3, device="cuda") * 0.4
        rgb = torch.sigmoid(torch.randn(M, 3, device="cuda"))
        opacity = torch.sigmoid(torch.randn(M, 1, device="cuda") - 2.0)
        scale = torch.exp(torch.randn(M, 3, device="cuda") * 0.3 - 4.0)
        quat = torch.nn.functional.normalize(torch.randn(M, 4, device="cuda"), dim=-1)
        fused = {"mean": mean, "rgb": rgb, "opacity": opacity,
                 "scale": scale, "quat": quat}

        out["step"] = "forward"
        t0 = time.time()
        with torch.no_grad():
            refined = head.refine(fused, voxel_size=voxel_size, radius=radius)
        torch.cuda.synchronize()
        out["forward_sec"] = round(time.time() - t0, 3)

        out["step"] = "check_step0_invariant"
        # Step-0 invariant: with zero-init weights + suppress bias -10,
        # refined output should equal the input to within float noise.
        opacity_l1 = (refined["opacity"] - fused["opacity"]).abs().mean().item()
        rgb_l1 = (refined["rgb"] - fused["rgb"]).abs().mean().item()
        mean_l1 = (refined["mean"] - fused["mean"]).abs().mean().item()
        out["zero_init_opacity_l1"] = opacity_l1
        out["zero_init_rgb_l1"] = rgb_l1
        out["zero_init_mean_l1"] = mean_l1
        # Tolerance: opacity dip from sigmoid(-10) ≈ 4.5e-5; with opacity≈0.05
        # input scale, expect L1 ≤ ~5e-6.  Loose tolerance for safety.
        out["zero_init_ok"] = (opacity_l1 < 1e-3 and rgb_l1 < 1e-3 and mean_l1 < 1e-3)

        out["step"] = "done"
        out["ok"] = True
    except Exception as exc:
        out["error"] = repr(exc)
        out["traceback"] = traceback.format_exc()

    print("[sparse_voxel_fusion_smoke]", out, flush=True)
    return out


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=12 * 60 * 60, cpu=12, memory=98304)
def train(args: list[str], dataset: str = DATASET, manifest: str = MANIFEST) -> None:
    _run_module("decoder.clean.train_phase2", args, dataset=dataset, manifest=manifest)
    cache_volume.commit()
    runs_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=12 * 60 * 60, cpu=12, memory=98304)
def train_string(args_string: str, dataset: str = DATASET, manifest: str = MANIFEST) -> None:
    _run_module(
        "decoder.clean.train_phase2",
        shlex.split(args_string),
        dataset=dataset,
        manifest=manifest,
    )
    cache_volume.commit()
    runs_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=6 * 60 * 60, cpu=8, memory=65536)
def pretrain_depth_refine(args: list[str], dataset: str = DATASET,
                          manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.pretrain_depth_refine", [*base, *args],
                dataset=dataset, manifest=manifest)
    runs_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=8 * 60 * 60, cpu=8, memory=65536)
def train_depth_anchor(args: list[str], dataset: str = DATASET,
                       manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.train_depth_anchor", [*base, *args],
                dataset=dataset, manifest=manifest)
    runs_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=4 * 60 * 60, cpu=8, memory=65536)
def bake_depth_anchor(args: list[str], dataset: str = DATASET,
                      manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.bake_depth_anchor", [*base, *args],
                dataset=dataset, manifest=manifest)
    runs_volume.commit()
    data_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=4 * 60 * 60, cpu=8, memory=65536)
def bake_depth_refine(args: list[str], dataset: str = DATASET,
                      manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.bake_depth_refine", [*base, *args],
                dataset=dataset, manifest=manifest)
    runs_volume.commit()
    data_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=6 * 60 * 60, cpu=8, memory=65536)
def pretrain_condition_rgb(args: list[str], dataset: str = DATASET,
                           manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.pretrain_condition_rgb", [*base, *args],
                dataset=dataset, manifest=manifest)
    runs_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=8 * 60 * 60, cpu=8, memory=65536)
def pretrain_condition_rgbd(args: list[str], dataset: str = DATASET,
                            manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.pretrain_condition_rgbd", [*base, *args],
                dataset=dataset, manifest=manifest)
    runs_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=6 * 60 * 60, cpu=8, memory=65536)
def pretrain_condition_mask(args: list[str], dataset: str = DATASET,
                            manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.pretrain_condition_mask", [*base, *args],
                dataset=dataset, manifest=manifest)
    runs_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=6 * 60 * 60, cpu=8, memory=65536)
def pretrain_surface_confidence(args: list[str], dataset: str = DATASET,
                                manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.pretrain_surface_confidence", [*base, *args],
                dataset=dataset, manifest=manifest)
    runs_volume.commit()


@app.function(image=decoder_image, volumes=_volumes(), timeout=60 * 60, cpu=4, memory=8192)
def condition_oracle(args: list[str], dataset: str = DATASET,
                     manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.eval_condition_oracle", [*base, *args],
                dataset=dataset, manifest=manifest)


@app.function(image=inspect_image, volumes=_volumes(), timeout=60 * 60, cpu=4, memory=8192)
def inspect_dataset_remote(args: list[str], dataset: str = DATASET,
                           manifest: str = MANIFEST) -> None:
    base = [
        "--dataset_root", _dataset_root(dataset),
        "--manifest", _dataset_manifest(dataset, manifest),
    ]
    _run_module("decoder.clean.inspect_dataset", [*base, *args],
                dataset=dataset, manifest=manifest)


@app.function(image=inspect_image, volumes=_volumes(), timeout=60 * 60, cpu=4, memory=8192)
def build_combined_manifest(args: list[str]) -> None:
    base = [
        "--dataset_root", _renders_mount_dir(),
        "--latents_v8_root", LATENTS_V8_DIR,
        "--out", f"{RUNS_DIR}/manifests/combined_v7_v8.json",
    ]
    _run_module("decoder.clean.build_combined_manifest", [*base, *args],
                dataset="combined_v7_v8", manifest=f"{RUNS_DIR}/manifests/combined_v7_v8.json")
    runs_volume.commit()


@app.function(image=decoder_image, gpu=GPU, volumes=_volumes(), timeout=12 * 60 * 60, cpu=12, memory=98304)
def densified(args: list[str], dataset: str = DATASET, manifest: str = MANIFEST) -> None:
    _run_module("decoder.clean.fit_densified_gs", args, dataset=dataset, manifest=manifest)
    cache_volume.commit()
    runs_volume.commit()


@app.function(
    image=ltx_decode_image,
    gpu=GPU,
    volumes={**_volumes(), WEIGHTS_DIR: weights_volume},
    timeout=2 * 60 * 60,
    cpu=4,
    memory=32768,
)
def decode_ltx(split: str = "all", limit: int = 0, overwrite: bool = False,
               extra_args: list[str] | None = None, dataset: str = DATASET,
               manifest: str = MANIFEST) -> None:
    from pathlib import Path

    candidates = [Path(LTX_RAW_VAE), Path(LTX_CKPT), Path(LTX_JASON_CKPT)]
    model_path = next((str(p) for p in candidates if p.exists()), "")
    if not model_path:
        raise FileNotFoundError(f"no LTX checkpoint found in: {[str(p) for p in candidates]}")
    extra = ["--raw_prefix", "global."] if model_path == LTX_RAW_VAE else []
    out_root_args = (
        ["--out_root", f"{RUNS_DIR}/condition"]
        if _is_combined_dataset(dataset) else []
    )
    _run_module(
        "decoder.clean.decode_ltx_latents",
        [
            "--dataset_root", _dataset_root(dataset),
            "--manifest", _dataset_manifest(dataset, manifest),
            "--split", split,
            "--model_path", model_path,
            "--out_subdir", "ltx_decoded",
            "--limit", str(limit),
            "--overwrite", "1" if overwrite else "0",
            *out_root_args,
            *extra,
            *(extra_args or []),
        ],
        dataset=dataset,
        manifest=manifest,
    )
    runs_volume.commit()
    data_volume.commit()


@app.function(
    image=depth_prior_image,
    gpu=GPU,
    volumes=_volumes(),
    timeout=2 * 60 * 60,
    cpu=4,
    memory=32768,
)
def predict_depth_anything(split: str = "all", limit: int = 0,
                           overwrite: bool = False,
                           extra_args: list[str] | None = None,
                           dataset: str = DATASET,
                           manifest: str = MANIFEST) -> None:
    out_root_args = (
        ["--out_root", f"{RUNS_DIR}/depth"]
        if _is_combined_dataset(dataset) else []
    )
    image_subdir = "@ltx_decoded" if _is_combined_dataset(dataset) else "ltx_decoded"
    _run_module(
        "decoder.clean.predict_depth_anything",
        [
            "--dataset_root", _dataset_root(dataset),
            "--manifest", _dataset_manifest(dataset, manifest),
            "--split", split,
            "--image_subdir", image_subdir,
            "--out_subdir", "depth_anything_ltx",
            "--limit", str(limit),
            "--overwrite", "1" if overwrite else "0",
            *out_root_args,
            *(extra_args or []),
        ],
        dataset=dataset,
        manifest=manifest,
    )
    runs_volume.commit()
    data_volume.commit()


@app.function(
    image=vggt_depth_image,
    gpu=GPU,
    volumes=_volumes(),
    timeout=3 * 60 * 60,
    cpu=4,
    memory=49152,
)
def predict_depth_vggt(split: str = "all", limit: int = 0,
                       overwrite: bool = False,
                       extra_args: list[str] | None = None,
                       dataset: str = DATASET,
                       manifest: str = MANIFEST) -> None:
    out_root_args = (
        ["--out_root", f"{RUNS_DIR}/depth"]
        if _is_combined_dataset(dataset) else []
    )
    image_subdir = "@ltx_decoded" if _is_combined_dataset(dataset) else "ltx_decoded"
    _run_module(
        "decoder.clean.predict_depth_vggt",
        [
            "--dataset_root", _dataset_root(dataset),
            "--manifest", _dataset_manifest(dataset, manifest),
            "--split", split,
            "--image_subdir", image_subdir,
            "--out_subdir", "vggt_ltx",
            "--limit", str(limit),
            "--overwrite", "1" if overwrite else "0",
            *out_root_args,
            *(extra_args or []),
        ],
        dataset=dataset,
        manifest=manifest,
    )
    runs_volume.commit()
    data_volume.commit()


@app.function(
    image=da3_depth_image,
    gpu=GPU,
    volumes=_volumes(),
    timeout=3 * 60 * 60,
    cpu=4,
    memory=49152,
)
def predict_depth_da3(split: str = "all", limit: int = 0,
                      overwrite: bool = False,
                      extra_args: list[str] | None = None,
                      dataset: str = DATASET,
                      manifest: str = MANIFEST) -> None:
    out_root_args = (
        ["--out_root", f"{RUNS_DIR}/depth"]
        if _is_combined_dataset(dataset) else []
    )
    image_subdir = "@ltx_decoded" if _is_combined_dataset(dataset) else "ltx_decoded"
    _run_module(
        "decoder.clean.predict_depth_da3",
        [
            "--dataset_root", _dataset_root(dataset),
            "--manifest", _dataset_manifest(dataset, manifest),
            "--split", split,
            "--image_subdir", image_subdir,
            "--out_subdir", "da3_ltx",
            "--limit", str(limit),
            "--overwrite", "1" if overwrite else "0",
            *out_root_args,
            *(extra_args or []),
        ],
        dataset=dataset,
        manifest=manifest,
    )
    runs_volume.commit()
    data_volume.commit()


def _phase1_depth_args(steps: int, name: str | None = None, eval_every: int = 250) -> list[str]:
    run_name = name or f"phase1_depth_{steps}"
    return [
        "--ups_stages", "4",
        "--scale_cap_frac", "0.012",
        "--lr", "5e-5",
        "--accum", "8",
        "--k_views", "2",
        "--num_workers", "0",
        "--warmup", "100",
        "--steps", str(steps),
        "--depth_weight", "1.0",
        "--depth_abs_weight", "0.05",
        "--depth_render_mode", "ED",
        "--depth_render_scale", "0.25",
        "--perceptual_weight", "0",
        "--eval_every", str(eval_every),
        "--log_every", "20",
        "--n_train_eval", "2",
        "--n_heldout_eval", "2",
        "--eval_render_chunk", "1",
        "--wandb_mode", "offline",
        "--min_free_vram_gb", "40",
        "--out_dir", f"{RUNS_DIR}/{run_name}",
    ]


def _phase3_offset_args(steps: int, name: str | None = None, eval_every: int = 250) -> list[str]:
    args = _phase1_depth_args(steps, name or f"phase3_offset_{steps}", eval_every)
    args.extend(["--mean_offset_frac", "0.05"])
    return args


def _quality_base_args(dataset: str, steps: int, name: str, eval_every: int,
                       lr: float = 0.0, max_train_objects: int = 0) -> list[str]:
    cond_subdir = "@ltx_decoded" if _is_combined_dataset(dataset) else "ltx_decoded"
    cond_depth_subdir = "@da3_ltx" if _is_combined_dataset(dataset) else "da3_ltx"
    return [
        "--steps", str(steps),
        "--accum", "1",
        "--k_views", "2",
        "--anchor_views", "9",
        "--anchor_render_mode", "iblend_fill",
        "--anchor_blend_topk", "2",
        "--anchor_blend_temp", "0.75",
        "--anchor_iblend_alpha_power", "1.0",
        "--anchor_iblend_view_weight", "1",
        "--anchor_iblend_color_mode", "maxweight",
        "--anchor_fill_alpha_power", "1.0",
        "--ups_stages", "5",
        "--scale_cap_frac", "0.0008",
        "--condition_source", "fixed",
        "--cond_subdir", cond_subdir,
        "--cond_depth_subdir", cond_depth_subdir,
        "--cond_view_indices", "available",
        "--filter_missing_condition", "1",
        "--filter_missing_condition_min_views", "9",
        "--image_condition", "1",
        "--image_head_skip", "1",
        "--image_depth_condition", "1",
        "--image_depth_skip", "1",
        "--image_normal_condition", "1",
        "--image_normal_quat", "1",
        "--image_camera_quat", "0",
        "--image_scale_frac", "0.00045",
        "--image_normal_scale_frac", "0.00008",
        "--image_opacity_fg", "0.95",
        "--image_opacity_bg", "0.0001",
        "--image_geom_residual_scale", "0.0",
        "--fusion_voxel_size_frac", "0.003",
        "--fusion_voxel_min_count", "2",
        "--fusion_voxel_max_per_cell", "1",
        "--fusion_voxel_mode", "select",
        "--fusion_voxel_color_mode", "average",
        "--fusion_voxel_representative", "score",
        "--fusion_voxel_scale_mult", "0.40",
        "--fusion_voxel_scale_floor_z_mult", "0.1",
        "--fusion_voxel_score_depth", "1",
        "--fusion_voxel_score_color", "1",
        "--fusion_voxel_score_conflict_weight", "0.0",
        "--fusion_voxel_low_support_opacity_decay", "1.0",
        "--fusion_sh_degree", "2",
        "--fusion_sh_mix", "0.1",
        "--condition_rgb_inpaint_px", "2",
        "--condition_mask_erode_px", "1",
        "--condition_unsharp_amount", "1.0",
        "--condition_contrast", "1.04",
        "--condition_saturation", "1.03",
        "--lr", f"{lr:g}",
        "--warmup", "1" if lr == 0 else "100",
        "--perceptual_weight", "0",
        "--num_workers", "0",
        "--max_train_objects", str(max_train_objects),
        "--n_train_eval", "8",
        "--n_heldout_eval", "16",
        "--eval_render_chunk", "1",
        "--eval_every", str(eval_every),
        "--log_every", "20",
        "--eval_at_step0", "1",
        "--save_eval_viz", "1",
        "--save_eval_viz_views", "2",
        "--save_every", str(eval_every),
        "--wandb_mode", "offline",
        "--min_free_vram_gb", "40",
        "--out_dir", f"{RUNS_DIR}/{name}",
    ]


def _quality_step0_args(dataset: str) -> list[str]:
    return _quality_base_args(
        dataset,
        1,
        f"{dataset}_quality_step0",
        eval_every=1,
        lr=0.0,
        max_train_objects=8,
    )


def _quality_colorcal_step0_args(dataset: str) -> list[str]:
    args = _quality_base_args(
        dataset,
        1,
        f"{dataset}_colorcal_step0",
        eval_every=1,
        lr=0.0,
        max_train_objects=0,
    )
    args.extend([
        "--condition_color_calibration", "train_affine",
        "--condition_color_calib_max_objects", "8",
        "--condition_color_calib_views", "9",
        "--fusion_voxel_low_support_opacity_decay", "2.0",
        "--fusion_voxel_score_conflict_weight", "0.5",
        "--fusion_voxel_score_opacity_norm", "2.0",
        "--fusion_voxel_score_opacity_floor", "0.2",
        "--fusion_voxel_score_opacity_power", "1.0",
        "--anchor_iblend_support_weight", "1.0",
        "--anchor_iblend_support_refs", "4",
        "--anchor_iblend_support_floor", "0.25",
        "--anchor_iblend_support_decay", "0.5",
        "--anchor_iblend_support_tol_frac", "0.025",
        "--fusion_depth_filter", "1",
        "--fusion_filter_all_views", "1",
        "--fusion_filter_mode", "opacity",
        "--fusion_filter_silhouette_weight", "1.0",
        "--fusion_filter_front_weight", "0.0",
        "--fusion_conflict_opacity_decay", "0.35",
        "--fusion_bg_margin_px", "3",
        "--output_alpha_cleanup_min", "0.08",
        "--output_alpha_cleanup_softness", "0.08",
    ])
    return args


def _set_arg(args: list[str], flag: str, value: str) -> None:
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend([flag, value])


def _quality_source_gtdepth_step0_args(dataset: str) -> list[str]:
    """Oracle RGBD ceiling using dataset RGB frames + GT depth as conditioning.

    This is not deployable; it is a cheap infrastructure/ceiling check for new
    datasets that do not yet have LTX-decoded RGB or DA3 depth caches.
    """
    args = _quality_colorcal_step0_args(dataset)
    _set_arg(args, "--cond_subdir", "")
    _set_arg(args, "--cond_depth_subdir", "depth")
    _set_arg(args, "--condition_color_calibration", "none")
    _set_arg(args, "--condition_rgb_inpaint_px", "0")
    _set_arg(args, "--condition_unsharp_amount", "0.0")
    _set_arg(args, "--condition_contrast", "1.0")
    _set_arg(args, "--condition_saturation", "1.0")
    _set_arg(args, "--n_train_eval", "0")
    _set_arg(args, "--n_heldout_eval", "12")
    _set_arg(args, "--save_eval_viz_views", "1")
    _set_arg(args, "--wandb_mode", "disabled")
    _set_arg(args, "--out_dir", f"{RUNS_DIR}/{dataset}_source_gtdepth_step0")
    return args


def _quality_sparse_voxel_args(dataset: str, steps: int, lr: float,
                               name: str | None = None,
                               eval_every: int = 250,
                               max_train_objects: int = 0) -> list[str]:
    """Quality config + SparseVoxelFusion module enabled.

    Same flags as quality_colorcal_step0, but with --use_sparse_voxel_fusion 1
    and a training-friendly schedule.  With steps=1, lr=0 this is the step-0
    invariant check (must equal the deterministic prior).
    """
    args = _quality_colorcal_step0_args(dataset)
    # Replace name + step/lr fields
    run_name = name or f"{dataset}_sparse_voxel_s{steps}_lr{lr:g}"
    out_idx = args.index("--out_dir")
    args[out_idx + 1] = f"{RUNS_DIR}/{run_name}"
    steps_idx = args.index("--steps")
    args[steps_idx + 1] = str(steps)
    lr_idx = args.index("--lr")
    args[lr_idx + 1] = f"{lr:g}"
    warmup_idx = args.index("--warmup")
    args[warmup_idx + 1] = "1" if lr == 0 else "100"
    max_obj_idx = args.index("--max_train_objects")
    args[max_obj_idx + 1] = str(max_train_objects)
    eval_idx = args.index("--eval_every")
    args[eval_idx + 1] = str(eval_every)
    save_idx = args.index("--save_every")
    args[save_idx + 1] = str(eval_every)
    args.extend([
        "--use_sparse_voxel_fusion", "1",
        "--sparse_voxel_hidden", "32",
        "--save_named_checkpoints", "1",
        "--freeze_decoder", "1",
        # v1.4: anti-deletion via three orthogonal levers:
        # 1) HIGH fg_weight: with default fg_weight=10, bg pixels (95%) still
        #    dominate L1 loss (95%*1=0.95) over fg (5%*10=0.50). So the loss
        #    rewards "clean bg spray" MORE than "accurate fg colors" -> bias
        #    toward deletion.  fg_weight=30 makes fg dominate 1.5x.
        # 2) Low mask_weight: silhouette-edge L1 prefers binary alpha; soft
        #    edges look like "shell to delete".
        # 3) Identity reg on full residual + bounded vis range.
        "--fg_weight", "30",                                 # NEW: was 10 default
        "--sparse_voxel_vis_delta", "0.2",
        "--sparse_voxel_identity_reg_weight", "1.0",
        "--sparse_voxel_opacity_res_scale", "0.05",
        "--mask_weight", "0.05",
    ])
    return args


def _novel_grid16_args(dataset: str) -> list[str]:
    args = _quality_colorcal_step0_args(dataset)
    args.extend([
        "--n_train_eval", "0",
        "--n_heldout_eval", "1",
        "--save_eval_viz_views", "16",
        "--save_eval_viz_novel_azimuths=0,90,180,270",
        "--save_eval_viz_novel_elevations=-35,-10,15,40",
        "--save_eval_viz_novel_radius_scale", "1.35",
        "--eval_render_chunk", "1",
        "--wandb_mode", "disabled",
    ])
    out_i = args.index("--out_dir") + 1
    args[out_i] = f"{RUNS_DIR}/{dataset}_colorcal_novelgrid16"
    return args


def _quality_depthcal_step0_args(dataset: str) -> list[str]:
    args = _quality_colorcal_step0_args(dataset)
    args.extend([
        "--condition_depth_calibration", "train_affine_frac",
        "--condition_depth_calib_max_objects", "64",
        "--condition_depth_calib_views", "9",
        "--condition_depth_calib_sample_px", "20000",
    ])
    out_i = args.index("--out_dir") + 1
    args[out_i] = f"{RUNS_DIR}/{dataset}_depthcal_step0"
    return args


def _quality_learned_fill_args(dataset: str, steps: int,
                               max_train_objects: int = 0) -> list[str]:
    eval_every = max(25, min(250, steps // 4 if steps >= 4 else 1))
    args = _quality_colorcal_step0_args(dataset)
    _set_arg(args, "--steps", str(steps))
    _set_arg(args, "--lr", "2e-4")
    _set_arg(args, "--warmup", "100")
    _set_arg(args, "--max_train_objects", str(max_train_objects))
    _set_arg(args, "--eval_every", str(eval_every))
    _set_arg(args, "--save_every", str(eval_every))
    _set_arg(args, "--out_dir", f"{RUNS_DIR}/{dataset}_learned_iblend_fill_{steps}")
    mode_i = args.index("--anchor_render_mode") + 1
    args[mode_i] = "learned_iblend_fill"
    args.extend([
        "--freeze_decoder", "1",
        "--anchor_learned_fill_hidden", "64",
        "--anchor_learned_fill_layers", "4",
        "--anchor_learned_fill_delta_scale", "1.0",
        "--anchor_learned_fill_candidate_delta_scale", "0.0",
        "--anchor_learned_fill_prior_weight", "0.001",
        "--anchor_learned_fill_tv_weight", "0.001",
        "--bg_alpha_weight", "0.05",
    ])
    return args


def _quality_unet_fill_args(dataset: str, steps: int,
                            max_train_objects: int = 0) -> list[str]:
    args = _quality_learned_fill_args(dataset, steps, max_train_objects=max_train_objects)
    args.extend([
        "--anchor_learned_fill_arch", "unet",
        "--anchor_learned_fill_hidden", "32",
        "--anchor_iblend_color_mode", "maxweight_st",
        "--anchor_learned_fill_delta_scale", "1.5",
        "--anchor_learned_fill_candidate_delta_scale", "1.5",
        "--anchor_learned_fill_prior_weight", "0.0005",
        "--anchor_learned_fill_tv_weight", "0.0005",
        "--lr", "1e-4",
    ])
    out_i = args.index("--out_dir") + 1
    args[out_i] = f"{RUNS_DIR}/{dataset}_learned_iblend_unet_{steps}"
    return args


def _quality_unet_oracle_fill_args(dataset: str, steps: int,
                                   max_train_objects: int = 0) -> list[str]:
    args = _quality_unet_fill_args(dataset, steps, max_train_objects=max_train_objects)
    args.extend([
        "--anchor_learned_fill_oracle_weight", "0.05",
        "--anchor_learned_fill_oracle_temp", "0.035",
        "--anchor_learned_fill_oracle_mask_weight", "0.25",
        "--anchor_learned_fill_delta_scale", "4",
        "--anchor_learned_fill_candidate_delta_scale", "4",
        "--anchor_learned_fill_prior_weight", "0.0002",
        "--anchor_learned_fill_tv_weight", "0.0002",
    ])
    out_i = args.index("--out_dir") + 1
    args[out_i] = f"{RUNS_DIR}/{dataset}_learned_iblend_unet_oracle_{steps}"
    return args


def _quality_depth_refine_args(dataset: str, steps: int,
                               max_train_objects: int = 0) -> list[str]:
    eval_every = max(25, min(250, steps // 4 if steps >= 4 else 1))
    args = _quality_base_args(
        dataset,
        steps,
        f"{dataset}_depth_refine_{steps}",
        eval_every=eval_every,
        lr=2e-4,
        max_train_objects=max_train_objects,
    )
    args.extend([
        "--freeze_decoder", "1",
        "--depth_refine_unet", "1",
        "--depth_refine_hidden", "24",
        "--depth_refine_delta_scale", "0.12",
        "--depth_refine_gt_weight", "1.0",
        "--depth_refine_prior_weight", "0.0005",
        "--depth_refine_tv_weight", "0.0005",
    ])
    return args


def _quality_support_gate_args(dataset: str, steps: int,
                               max_train_objects: int = 0) -> list[str]:
    args = _quality_base_args(
        dataset,
        steps,
        f"{dataset}_support_gate_mv_{steps}",
        eval_every=max(25, min(250, steps // 4 if steps >= 4 else 1)),
        lr=1e-4,
        max_train_objects=max_train_objects,
    )
    args.extend([
        "--freeze_decoder", "1",
        "--support_gate_unet", "1",
        "--support_gate_hidden", "24",
        "--support_gate_init", "0.99",
        "--support_gate_floor", "0.25",
        "--support_gate_delta_scale", "4.0",
        "--support_gate_gt_weight", "0.5",
        "--support_gate_prior_weight", "0.0005",
        "--support_gate_tv_weight", "0.0005",
        "--support_gate_depth_tol_frac", "0.02",
        "--support_gate_multiview_target", "1",
        "--support_gate_multiview_refs", "4",
        "--fusion_voxel_low_support_opacity_decay", "2.0",
        "--anchor_iblend_support_weight", "1.0",
        "--anchor_iblend_support_refs", "4",
        "--anchor_iblend_support_floor", "0.25",
        "--anchor_iblend_support_decay", "0.5",
        "--anchor_iblend_support_tol_frac", "0.025",
        "--output_alpha_cleanup_min", "0.08",
        "--output_alpha_cleanup_softness", "0.08",
    ])
    return args


def _quality_surface_confidence_args(dataset: str, steps: int,
                                     max_train_objects: int = 0) -> list[str]:
    eval_every = max(25, min(100, steps // 4 if steps >= 4 else 1))
    args = _quality_colorcal_step0_args(dataset)

    def set_arg(flag: str, value: str) -> None:
        args[args.index(flag) + 1] = value

    set_arg("--steps", str(steps))
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--lr", "1e-4")
    set_arg("--warmup", "50")
    set_arg("--max_train_objects", str(max_train_objects))
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_surface_confidence_{steps}")
    args.extend([
        "--freeze_decoder", "1",
        "--surface_confidence_unet", "1",
        "--surface_confidence_hidden", "32",
        "--surface_confidence_init", "0.90",
        "--surface_confidence_floor", "0.30",
        "--surface_confidence_opacity_max", "0.995",
        "--surface_confidence_gate_strength", "1.0",
        "--surface_confidence_delta_scale", "4.0",
        "--surface_confidence_gt_weight", "0.2",
        "--surface_confidence_prior_weight", "0.0002",
        "--surface_confidence_tv_weight", "0.0002",
        "--surface_confidence_positive_weight", "4.0",
        "--surface_confidence_negative_weight", "1.0",
        "--surface_confidence_target_pos_min", "0.75",
        "--surface_confidence_target_neg_max", "0.05",
        "--surface_confidence_target_min_pos_support", "1.0",
        "--surface_confidence_target_min_neg_conflicts", "2.0",
        "--surface_confidence_depth_tol_frac", "0.02",
        "--surface_confidence_multiview_refs", "4",
        "--surface_confidence_multiview_tol_frac", "0.02",
        "--surface_confidence_score_weight", "1.0",
        "--mask_weight", "1.0",
        "--bg_alpha_weight", "0.03",
    ])
    return args


def _quality_surface_depth_confidence_args(dataset: str, steps: int,
                                           max_train_objects: int = 0) -> list[str]:
    args = _quality_surface_confidence_args(dataset, steps, max_train_objects)
    args[args.index("--out_dir") + 1] = (
        f"{RUNS_DIR}/{dataset}_surface_depth_confidence_{steps}"
    )
    args.extend([
        "--depth_refine_unet", "1",
        "--depth_refine_hidden", "24",
        "--depth_refine_delta_scale", "0.04",
        "--depth_refine_gt_weight", "0.2",
        "--depth_refine_gt_outlier_weight", "1.0",
        "--depth_refine_gt_outlier_power", "1.0",
        "--depth_refine_prior_weight", "0.002",
        "--depth_refine_tv_weight", "0.002",
        "--depth_refine_multiview_features", "1",
        "--depth_refine_multiview_refs", "4",
        "--depth_refine_multiview_tol_frac", "0.02",
    ])
    return args


def _quality_output_alpha_refine_args(dataset: str, steps: int,
                                      max_train_objects: int = 0) -> list[str]:
    eval_every = max(25, min(100, steps // 4 if steps >= 4 else 1))
    args = _quality_colorcal_step0_args(dataset)

    def set_arg(flag: str, value: str) -> None:
        args[args.index(flag) + 1] = value

    set_arg("--steps", str(steps))
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--lr", "1e-4")
    set_arg("--warmup", "50")
    set_arg("--max_train_objects", str(max_train_objects))
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_output_alpha_refine_{steps}")
    args.extend([
        "--freeze_decoder", "1",
        "--output_alpha_refine_unet", "1",
        "--output_alpha_refine_hidden", "16",
        "--output_alpha_refine_init", "0.995",
        "--output_alpha_refine_delta_scale", "10",
        "--output_alpha_refine_prior_weight", "0.0002",
        "--output_alpha_refine_tv_weight", "0.0002",
        "--mask_weight", "1.0",
        "--bg_alpha_weight", "0.05",
    ])
    return args


def _quality_fusion_candidate_hot_args(dataset: str, steps: int,
                                       max_train_objects: int = 32) -> list[str]:
    """Small learned pre-voxel candidate gate.

    This is the first learned module that showed a real heldout improvement over
    the deterministic RGBD fusion prior on combined_v7_v8 eval-16. It freezes the
    88M decoder and trains only a tiny per-splat support/conflict gate before
    voxel selection, so the learned part changes the 3D candidate set instead of
    doing an image-space postprocess.

    v7+v8 eval-48 after the capped-camera fix showed a stronger deterministic
    prior with slightly wider splats and less RGB sharpening:
    scale_mult=0.45, unsharp=0.5.  Keep those defaults here so candidate-gate
    pilots build on the current best feed-forward prior.
    """
    eval_every = max(20, min(80, steps // 2 if steps >= 2 else 1))
    args = _quality_colorcal_step0_args(dataset)

    def set_arg(flag: str, value: str) -> None:
        _set_arg(args, flag, value)

    set_arg("--steps", str(steps))
    set_arg("--lr", "0.003")
    set_arg("--warmup", "10")
    set_arg("--max_train_objects", str(max_train_objects))
    set_arg("--n_train_eval", "4")
    set_arg("--n_heldout_eval", "16")
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--save_eval_viz_views", "1")
    set_arg("--save_eval_viz_heldout_count", "2")
    set_arg("--save_eval_viz_train_count", "1")
    set_arg("--fusion_voxel_scale_mult", "0.45")
    set_arg("--condition_unsharp_amount", "0.5")
    set_arg("--bg_alpha_weight", "0.03")
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_fusion_candidate_hot_{steps}")
    args.extend([
        "--freeze_decoder", "1",
        "--fusion_candidate_gate", "1",
        "--fusion_candidate_hidden", "32",
        "--fusion_candidate_layers", "3",
        "--fusion_candidate_score_delta_scale", "8.0",
        "--fusion_candidate_opacity_delta_scale", "8.0",
        "--fusion_candidate_opacity_init", "0.9",
        "--fusion_candidate_opacity_floor", "0.15",
        "--fusion_candidate_gt_weight", "1.0",
        "--fusion_candidate_prior_weight", "0.00005",
        "--fusion_candidate_positive_weight", "2.0",
        "--fusion_candidate_negative_weight", "4.0",
    ])
    return args


def _quality_fusion_candidate_scoreonly_args(dataset: str, steps: int,
                                             max_train_objects: int = 0) -> list[str]:
    """Current best candidate-gate tradeoff: score/ranking only, no opacity gate.

    Full combined_v7_v8 eval-138, 8 views/object:
      deterministic prior: 20.652 PSNR / 0.946 sharp / 0.958 IoU
      score-only h128x4:   20.707 PSNR / 0.942 sharp / 0.957 IoU

    This keeps the learned module from dimming/deleting surfaces while still
    letting it alter pre-voxel representative ranking from support labels.
    """
    args = _quality_fusion_candidate_hot_args(
        dataset, steps, max_train_objects=max_train_objects
    )
    _set_arg(args, "--out_dir", f"{RUNS_DIR}/{dataset}_fusion_candidate_scoreonly_{steps}")
    _set_arg(args, "--lr", "0.001")
    _set_arg(args, "--warmup", "10")
    _set_arg(args, "--fusion_candidate_hidden", "128")
    _set_arg(args, "--fusion_candidate_layers", "4")
    _set_arg(args, "--fusion_candidate_coord_features", "1")
    _set_arg(args, "--fusion_candidate_rich_features", "0")
    _set_arg(args, "--fusion_candidate_voxel_features", "0")
    _set_arg(args, "--fusion_candidate_opacity_delta_scale", "0")
    _set_arg(args, "--fusion_voxel_scale_mult", "0.40")
    _set_arg(args, "--condition_unsharp_amount", "1.0")
    _set_arg(args, "--eval_before_train", "1")
    _set_arg(args, "--eval_at_step0", "0")
    _set_arg(args, "--eval_views_per_object", "8")
    return args


def _quality_depth_confidence_hot_args(dataset: str, steps: int,
                                       max_train_objects: int = 0) -> list[str]:
    """Strong RGBD prior + learned conditioning-depth confidence.

    This keeps the current best deterministic/candidate configuration and adds
    only a small per-pixel confidence head over LTX RGB + DA3 depth + multiview
    support.  The head is trained from GT depth but receives no GT depth at
    inference; it can only reweight source-depth support before 3DGS fusion.
    """
    args = _quality_fusion_candidate_hot_args(
        dataset, steps, max_train_objects=max_train_objects
    )
    _set_arg(args, "--out_dir", f"{RUNS_DIR}/{dataset}_fusion_candidate_depthconf_hot_{steps}")
    _set_arg(args, "--n_heldout_eval", "48")
    _set_arg(args, "--eval_views_per_object", "8")
    _set_arg(args, "--eval_at_step0", "0")
    _set_arg(args, "--eval_before_train", "1")
    _set_arg(args, "--save_checkpoints", "0")
    _set_arg(args, "--save_named_checkpoints", "0")
    _set_arg(args, "--save_eval_viz_train_count", "0")
    _set_arg(args, "--lr", "0.0001")
    _set_arg(args, "--warmup", "5")
    args.extend([
        "--condition_depth_confidence_unet", "1",
        "--condition_depth_confidence_hidden", "32",
        "--condition_depth_confidence_multiview_features", "1",
        "--condition_depth_confidence_init", "0.995",
        "--condition_depth_confidence_floor", "0.70",
        "--condition_depth_confidence_delta_scale", "3.0",
        "--condition_depth_confidence_gt_weight", "0.5",
        "--condition_depth_confidence_prior_weight", "0.001",
        "--condition_depth_confidence_tv_weight", "0.0005",
        "--condition_depth_confidence_tol_frac", "0.02",
        "--condition_depth_confidence_neg_tol_frac", "0.06",
        "--condition_depth_confidence_positive_weight", "4.0",
        "--condition_depth_confidence_negative_weight", "1.0",
        "--fusion_voxel_score_confidence", "1",
        "--fusion_voxel_score_confidence_floor", "0.50",
    ])
    return args


def _quality_rgbd_depth_refine_args(dataset: str, steps: int,
                                    max_train_objects: int = 0) -> list[str]:
    """Learn bounded depth corrections before deterministic RGBD fusion.

    This targets the remaining non-oracle gap without using a deletion-style
    confidence head.  The RGB path is kept deterministic/color-calibrated
    because the RGBD refiner's direct RGB loss is applied before the affine
    color calibration in train_phase2.py; enabling RGB residuals here can fight
    the current best color prior.  The depth head is zero-initialized, so the
    step -1 eval is exactly the deterministic prior for the same heldout slice.
    """
    eval_every = max(20, min(80, steps // 2 if steps >= 2 else 1))
    args = _quality_colorcal_step0_args(dataset)

    def set_arg(flag: str, value: str) -> None:
        _set_arg(args, flag, value)

    set_arg("--steps", str(steps))
    set_arg("--lr", "0.0002")
    set_arg("--warmup", "10")
    set_arg("--max_train_objects", str(max_train_objects))
    set_arg("--n_train_eval", "4")
    set_arg("--n_heldout_eval", "16")
    set_arg("--eval_views_per_object", "8")
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--eval_at_step0", "0")
    set_arg("--eval_before_train", "1")
    set_arg("--save_eval_viz", "0")
    set_arg("--save_eval_viz_views", "1")
    set_arg("--save_checkpoints", "0")
    set_arg("--save_named_checkpoints", "0")
    set_arg("--fusion_voxel_scale_mult", "0.45")
    set_arg("--condition_unsharp_amount", "0.5")
    set_arg("--bg_alpha_weight", "0.03")
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_rgbd_depth_refine_{steps}")
    args.extend([
        "--freeze_decoder", "1",
        "--condition_rgbd_refine_unet", "1",
        "--condition_rgbd_refine_arch", "view",
        "--condition_rgbd_refine_hidden", "48",
        "--condition_rgbd_refine_context_layers", "2",
        "--condition_rgbd_refine_context_heads", "4",
        "--condition_rgbd_refine_multiview_features", "1",
        "--condition_rgbd_refine_multiview_refs", "4",
        "--condition_rgbd_refine_multiview_tol_frac", "0.02",
        "--condition_rgbd_refine_rgb_scale", "0.0",
        "--condition_rgbd_refine_depth_scale", "0.18",
        "--condition_rgbd_refine_apply_erode_px", "1",
        "--condition_rgbd_refine_prior_weight", "0.001",
        "--condition_rgbd_refine_tv_weight", "0.0005",
        "--condition_rgbd_refine_rgb_gt_weight", "0.0",
        "--condition_rgbd_refine_depth_gt_weight", "1.0",
        "--condition_rgbd_refine_gt_alpha_min", "0.5",
    ])
    return args


def _quality_depth_affine_hot_args(dataset: str, steps: int,
                                   max_train_objects: int = 0) -> list[str]:
    """Strong RGBD prior + learned per-view depth scale/shift correction.

    The depth oracle gap is much larger than the opacity-gate gap. This pilot
    keeps the best deterministic/candidate setup but adds a tiny zero-init MLP
    that predicts bounded affine corrections on normalized DA3 depth from
    inference-available RGB/depth/mask statistics.
    """
    args = _quality_fusion_candidate_hot_args(
        dataset, steps, max_train_objects=max_train_objects
    )
    _set_arg(args, "--out_dir", f"{RUNS_DIR}/{dataset}_fusion_candidate_depthaffine_hot_{steps}")
    _set_arg(args, "--n_heldout_eval", "16")
    _set_arg(args, "--eval_views_per_object", "8")
    _set_arg(args, "--eval_at_step0", "0")
    _set_arg(args, "--eval_before_train", "1")
    _set_arg(args, "--save_checkpoints", "0")
    _set_arg(args, "--save_named_checkpoints", "0")
    _set_arg(args, "--save_eval_viz", "0")
    _set_arg(args, "--save_eval_viz_train_count", "0")
    _set_arg(args, "--lr", "0.001")
    _set_arg(args, "--warmup", "5")
    args.extend([
        "--fusion_voxel_score_softmax_temp", "0.25",
        "--fusion_voxel_score_soft_opacity_mix", "0.2",
        "--condition_depth_affine_head", "1",
        "--condition_depth_affine_hidden", "64",
        "--condition_depth_affine_layers", "3",
        "--condition_depth_affine_scale_range", "0.20",
        "--condition_depth_affine_shift_range", "0.04",
        "--condition_depth_affine_gt_weight", "2.0",
        "--condition_depth_affine_prior_weight", "0.001",
    ])
    return args


def _quality_condition_rgb_refine_gt_args(dataset: str, steps: int,
                                          max_train_objects: int = 0) -> list[str]:
    """Learn an LTX-decoded RGB correction before RGBD fusion.

    This keeps the deterministic score-gated RGBD prior intact at step 0, freezes
    the base latent decoder, and gives the small condition RGB U-Net a direct
    masked source-view target in addition to the downstream render loss.
    """
    eval_every = max(20, min(80, steps // 3 if steps >= 4 else 1))
    args = _quality_colorcal_step0_args(dataset)

    def set_arg(flag: str, value: str) -> None:
        if flag in args:
            args[args.index(flag) + 1] = value
        else:
            args.extend([flag, value])

    set_arg("--steps", str(steps))
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--lr", "1e-4")
    set_arg("--warmup", "20")
    set_arg("--max_train_objects", str(max_train_objects))
    set_arg("--n_train_eval", "4")
    set_arg("--n_heldout_eval", "4")
    set_arg("--save_eval_viz_views", "1")
    set_arg("--fusion_voxel_scale_mult", "0.45")
    set_arg("--condition_unsharp_amount", "0.5")
    set_arg("--mask_weight", "1.0")
    set_arg("--bg_alpha_weight", "0.03")
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_condition_rgb_refine_gt_{steps}")
    args.extend([
        "--freeze_decoder", "1",
        "--condition_rgb_refine_unet", "1",
        "--condition_rgb_refine_hidden", "32",
        "--condition_rgb_refine_scale", "0.12",
        "--condition_rgb_refine_gt_weight", "2.0",
        "--condition_rgb_refine_gt_alpha_min", "0.5",
        "--fg_color_weight", "0.05",
        "--save_named_checkpoints", "1",
    ])
    return args


def _quality_surface_refine_args(dataset: str, steps: int,
                                 max_train_objects: int = 0) -> list[str]:
    """Learn high-res source-surface residuals before 3DGS voxel fusion.

    Unlike learned_iblend_fill, this edits the reusable 3DGS asset: opacity,
    scale, and color residuals are applied to each source-view Gaussian shell
    before static voxel fusion.  The head is zero-initialized, so the
    eval-before-train row is the deterministic DA3/LTX RGB prior.
    """
    eval_every = max(20, min(80, steps // 2 if steps >= 2 else 1))
    args = _quality_colorcal_step0_args(dataset)

    def set_arg(flag: str, value: str) -> None:
        _set_arg(args, flag, value)

    set_arg("--steps", str(steps))
    set_arg("--lr", "0.0003")
    set_arg("--warmup", "10")
    set_arg("--max_train_objects", str(max_train_objects))
    set_arg("--n_train_eval", "4")
    set_arg("--n_heldout_eval", "16")
    set_arg("--eval_views_per_object", "8")
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--eval_at_step0", "0")
    set_arg("--eval_before_train", "1")
    set_arg("--save_eval_viz", "0")
    set_arg("--save_eval_viz_views", "1")
    set_arg("--save_checkpoints", "0")
    set_arg("--save_named_checkpoints", "0")
    set_arg("--fusion_voxel_scale_mult", "0.45")
    set_arg("--condition_unsharp_amount", "0.5")
    set_arg("--bg_alpha_weight", "0.03")
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_surface_refine_{steps}")
    args.extend([
        "--freeze_decoder", "1",
        "--surface_refine_unet", "1",
        "--surface_refine_hidden", "32",
        "--surface_refine_init", "0.995",
        "--surface_refine_opacity_floor", "0.35",
        "--surface_refine_opacity_delta_scale", "2.0",
        "--surface_refine_scale_delta_scale", "0.20",
        "--surface_refine_scale_floor", "0.65",
        "--surface_refine_rgb_delta_scale", "0.08",
        "--surface_refine_prior_weight", "0.0005",
        "--surface_refine_tv_weight", "0.0002",
        "--surface_refine_rgb_gt_weight", "1.0",
        "--surface_refine_gt_alpha_min", "0.5",
    ])
    return args


def _quality_residual_decoder_args(dataset: str, steps: int,
                                   max_train_objects: int = 0) -> list[str]:
    args = _quality_base_args(
        dataset,
        steps,
        f"{dataset}_residual_decoder_{steps}",
        eval_every=max(25, min(250, steps // 4 if steps >= 4 else 1)),
        lr=5e-5,
        max_train_objects=max_train_objects,
    )
    args.extend([
        "--accum", "4",
        "--mean_offset_frac", "0.012",
        "--image_geom_residual_scale", "0.05",
        "--image_depth_residual_scale", "0.05",
        "--image_rgb_residual_scale", "0.05",
        "--image_opacity_residual_scale", "0.05",
        "--depth_weight", "0.02",
        "--depth_render_scale", "0.5",
        "--bg_alpha_weight", "0.02",
        "--grad_weight", "0.02",
        "--grad_start", "50",
        "--grad_ramp", "100",
        "--fusion_voxel_low_support_opacity_decay", "2.0",
        "--anchor_iblend_support_weight", "1.0",
        "--anchor_iblend_support_refs", "4",
        "--anchor_iblend_support_floor", "0.25",
        "--anchor_iblend_support_decay", "0.5",
        "--anchor_iblend_support_tol_frac", "0.025",
        "--output_alpha_cleanup_min", "0.08",
        "--output_alpha_cleanup_softness", "0.08",
    ])
    return args


def _quality_residual_decoder_aggressive_args(dataset: str, steps: int,
                                              max_train_objects: int = 0) -> list[str]:
    """Higher-movement residual decoder preset for learned 3DGS corrections.

    The conservative residual preset is stable but barely changes heldout
    renders. This variant preserves the deterministic RGBD prior at step 0 with
    a zero-initialized residual head, then gives the decoder enough residual
    range and schedule room to make measurable color/opacity/depth/geometry
    corrections.
    """
    args = _quality_residual_decoder_args(dataset, steps, max_train_objects)

    def set_arg(flag: str, value: str) -> None:
        found = False
        for i, item in enumerate(args):
            if item == flag:
                args[i + 1] = value
                found = True
        if not found:
            raise ValueError(f"missing expected argument {flag}")

    eval_every = max(50, min(150, steps // 2 if steps >= 4 else 1))
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_residual_decoder_aggressive_{steps}")
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--lr", "1e-4")
    set_arg("--warmup", "25")
    set_arg("--n_train_eval", "4")
    set_arg("--n_heldout_eval", "4")
    set_arg("--save_eval_viz_views", "1")
    set_arg("--mean_offset_frac", "0.02")
    set_arg("--image_geom_residual_scale", "0.12")
    set_arg("--image_depth_residual_scale", "0.20")
    set_arg("--image_rgb_residual_scale", "0.20")
    set_arg("--image_opacity_residual_scale", "0.20")
    set_arg("--grad_weight", "0.04")
    set_arg("--grad_start", "0")
    set_arg("--grad_ramp", "50")
    set_arg("--bg_alpha_weight", "0.03")
    args.extend([
        "--zero_init_head", "1",
        "--explicit_depth_head", "1",
        "--explicit_visibility_head", "1",
        "--depth_head_scale", "0.20",
        "--visibility_head_scale", "0.20",
        "--fg_color_weight", "0.05",
    ])
    return args


def _quality_residual_decoder_rgb_args(dataset: str, steps: int,
                                       max_train_objects: int = 0) -> list[str]:
    """Learn only RGB residuals on top of deterministic RGBD fusion.

    This is the first residual setting that improved full-v7 heldout PSNR while
    preserving alpha IoU. Geometry/depth/opacity residuals are kept disabled
    here because the current small-scale tests improve sharpness but hurt
    heldout mask/PSNR.
    """
    args = _quality_residual_decoder_aggressive_args(dataset, steps, max_train_objects)

    def set_arg(flag: str, value: str) -> None:
        found = False
        for i, item in enumerate(args):
            if item == flag:
                args[i + 1] = value
                found = True
        if not found:
            args.extend([flag, value])

    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_residual_decoder_rgb_{steps}")
    set_arg("--mean_offset_frac", "0")
    set_arg("--image_geom_residual_scale", "0")
    set_arg("--image_depth_residual_scale", "0")
    set_arg("--image_opacity_residual_scale", "0")
    set_arg("--explicit_depth_head", "0")
    set_arg("--explicit_visibility_head", "0")
    set_arg("--grad_weight", "0")
    set_arg("--mask_weight", "1.0")
    set_arg("--bg_alpha_weight", "0.05")
    set_arg("--fg_color_weight", "0.10")
    return args


def _quality_residual_decoder_staged_args(dataset: str, steps: int,
                                          max_train_objects: int = 0) -> list[str]:
    """Small non-RGB residual stage intended to resume from the RGB-only decoder.

    Keep the architecture checkpoint-compatible with the RGB-only run, but allow
    low-range depth/scale/quaternion/opacity residuals with direct L2 penalties.
    """
    args = _quality_residual_decoder_rgb_args(dataset, steps, max_train_objects)

    def set_arg(flag: str, value: str) -> None:
        found = False
        for i, item in enumerate(args):
            if item == flag:
                args[i + 1] = value
                found = True
        if not found:
            args.extend([flag, value])

    eval_every = max(40, min(120, steps // 3 if steps >= 4 else 1))
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_residual_decoder_staged_{steps}")
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--lr", "2e-5")
    set_arg("--warmup", "10")
    set_arg("--reset_optimizer_on_resume", "1")
    set_arg("--image_geom_residual_scale", "0.02")
    set_arg("--image_depth_residual_scale", "0.02")
    set_arg("--image_opacity_residual_scale", "0.02")
    set_arg("--residual_geom_weight", "0.01")
    set_arg("--residual_depth_weight", "0.02")
    set_arg("--residual_opacity_weight", "0.02")
    set_arg("--residual_rgb_weight", "0")
    set_arg("--residual_offset_weight", "0")
    return args


def _quality_bold_learned_decoder_args(dataset: str, steps: int,
                                       max_train_objects: int = 0) -> list[str]:
    """Large learned decoder preset for full v7+v8.

    This intentionally stops being a tiny diagnostic head. It trains the
    latent-transformer decoder itself and adds learned modules at each
    reconstruction stage:

    - high-resolution RGBD view refinement before lifting,
    - source-surface residuals before voxel fusion,
    - rich pre-fusion candidate scoring with score-soft geometry,
    - learned occupied-voxel message passing after fusion.

    The deterministic RGBD path is still present as an identity prior, but the
    trainable part now has enough capacity and gradient paths to own geometry,
    color, and visibility instead of only nudging scalar opacity.
    """
    args = _quality_residual_decoder_aggressive_args(
        dataset, steps, max_train_objects=max_train_objects
    )

    def set_arg(flag: str, value: str) -> None:
        found = False
        for i, item in enumerate(args):
            if item == flag:
                args[i + 1] = value
                found = True
        if not found:
            args.extend([flag, value])

    eval_every = max(25, min(100, steps // 4 if steps >= 4 else 1))
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_bold_learned_decoder_{steps}")
    set_arg("--lr", "5e-5")
    set_arg("--warmup", "50")
    set_arg("--accum", "2")
    set_arg("--anchor_views", "4")
    set_arg("--n_train_eval", "4")
    set_arg("--n_heldout_eval", "16")
    set_arg("--eval_views_per_object", "8")
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--eval_before_train", "1")
    set_arg("--eval_at_step0", "0")
    set_arg("--save_eval_viz", "1")
    set_arg("--save_eval_viz_views", "1")
    set_arg("--save_eval_viz_heldout_count", "4")
    set_arg("--save_eval_viz_train_count", "1")
    set_arg("--save_checkpoints", "1")
    set_arg("--save_named_checkpoints", "1")
    set_arg("--fusion_voxel_color_mode", "score_soft")
    set_arg("--fusion_voxel_score_softmax_temp", "0.25")
    set_arg("--fusion_voxel_score_soft_opacity_mix", "0.25")
    set_arg("--fusion_voxel_score_soft_geometry_mix", "0.5")
    set_arg("--mask_weight", "1.0")
    set_arg("--bg_alpha_weight", "0.05")
    set_arg("--fg_alpha_weight", "1.0")
    set_arg("--grad_weight", "0.03")
    set_arg("--grad_start", "0")
    set_arg("--grad_ramp", "100")
    args.extend([
        "--condition_rgbd_refine_unet", "1",
        "--condition_rgbd_refine_arch", "view",
        "--condition_rgbd_refine_hidden", "32",
        "--condition_rgbd_refine_context_layers", "2",
        "--condition_rgbd_refine_context_heads", "4",
        "--condition_rgbd_refine_multiview_features", "1",
        "--condition_rgbd_refine_multiview_refs", "4",
        "--condition_rgbd_refine_multiview_tol_frac", "0.02",
        "--condition_rgbd_refine_rgb_scale", "0.06",
        "--condition_rgbd_refine_depth_scale", "0.12",
        "--condition_rgbd_refine_apply_erode_px", "1",
        "--condition_rgbd_refine_prior_weight", "0.001",
        "--condition_rgbd_refine_tv_weight", "0.0005",
        "--condition_rgbd_refine_rgb_gt_weight", "0.5",
        "--condition_rgbd_refine_depth_gt_weight", "0.5",
        "--condition_rgbd_refine_gt_alpha_min", "0.5",
        "--surface_refine_unet", "1",
        "--surface_refine_hidden", "16",
        "--surface_refine_init", "0.995",
        "--surface_refine_opacity_floor", "0.35",
        "--surface_refine_opacity_delta_scale", "1.5",
        "--surface_refine_scale_delta_scale", "0.15",
        "--surface_refine_scale_floor", "0.65",
        "--surface_refine_rgb_delta_scale", "0.06",
        "--surface_refine_prior_weight", "0.0005",
        "--surface_refine_tv_weight", "0.0002",
        "--surface_refine_checkpoint", "1",
        "--surface_refine_rgb_gt_weight", "0.5",
        "--surface_refine_gt_alpha_min", "0.5",
        "--fusion_candidate_gate", "1",
        "--fusion_candidate_hidden", "160",
        "--fusion_candidate_layers", "3",
        "--fusion_candidate_coord_features", "1",
        "--fusion_candidate_rich_features", "1",
        "--fusion_candidate_voxel_features", "1",
        "--fusion_candidate_neighbor_features", "1",
        "--fusion_candidate_neighbor_radius", "1",
        "--fusion_candidate_checkpoint", "1",
        "--fusion_candidate_chunk_size", "131072",
        "--fusion_candidate_score_delta_scale", "3.0",
        "--fusion_candidate_opacity_delta_scale", "0.0",
        "--fusion_candidate_gt_weight", "0.25",
        "--fusion_candidate_gt_source", "target_depth",
        "--fusion_candidate_prior_weight", "0.0001",
        "--fusion_candidate_positive_weight", "2.0",
        "--fusion_candidate_negative_weight", "2.0",
        "--use_message_voxel_fusion", "1",
        "--sparse_voxel_hidden", "256",
        "--mlp_voxel_layers", "2",
        "--mlp_voxel_neighbor_radius", "1",
        "--mlp_voxel_message_radius", "1",
        "--sparse_voxel_depth_res_frac", "0.008",
        "--sparse_voxel_rgb_res_scale", "0.04",
        "--sparse_voxel_opacity_res_scale", "0.12",
        "--sparse_voxel_vis_delta", "0.30",
        "--sparse_voxel_identity_reg_weight", "0.02",
        "--sparse_voxel_support_reg_weight", "0.20",
        "--sparse_voxel_target_vis_weight", "0.50",
        "--sparse_voxel_target_vis_pos_min", "0.75",
        "--sparse_voxel_target_vis_neg_max", "0.25",
        "--sparse_voxel_target_vis_positive_weight", "3.0",
        "--sparse_voxel_target_vis_negative_weight", "1.0",
    ])
    return args


def _quality_bold_visibility_decoder_args(dataset: str, steps: int,
                                          max_train_objects: int = 0) -> list[str]:
    """Aggressive learned-visibility variant of the bold decoder.

    The first bold pilot made the model much larger, but it kept the most
    important pre-fusion visibility path disabled:
    ``--fusion_candidate_opacity_delta_scale 0``. That lets the candidate head
    learn target-depth labels without giving it enough render-connected control
    to remove DA3/LTX shell candidates before voxel fusion. This variant makes
    that path active while keeping foreground alpha and identity regularization
    strong enough to catch collapse quickly.
    """
    args = _quality_bold_learned_decoder_args(
        dataset, steps, max_train_objects=max_train_objects
    )

    def set_arg(flag: str, value: str) -> None:
        found = False
        for i, item in enumerate(args):
            if item == flag:
                args[i + 1] = value
                found = True
        if not found:
            args.extend([flag, value])

    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_bold_visibility_decoder_{steps}")
    set_arg("--fusion_candidate_opacity_delta_scale", "8.0")
    set_arg("--fusion_candidate_opacity_floor", "0.02")
    set_arg("--fusion_candidate_gt_weight", "1.0")
    set_arg("--fusion_candidate_prior_weight", "0.00005")
    set_arg("--fusion_candidate_positive_weight", "3.0")
    set_arg("--fusion_candidate_negative_weight", "2.0")
    set_arg("--fusion_voxel_score_soft_geometry_mix", "0.75")
    set_arg("--sparse_voxel_identity_reg_weight", "0.005")
    set_arg("--sparse_voxel_support_reg_weight", "0.10")
    set_arg("--sparse_voxel_target_vis_weight", "1.0")
    set_arg("--sparse_voxel_opacity_res_scale", "0.18")
    set_arg("--sparse_voxel_vis_delta", "0.45")
    set_arg("--fg_alpha_weight", "2.0")
    set_arg("--bg_alpha_weight", "0.08")
    set_arg("--mask_weight", "1.25")
    return args


def _quality_surface_token_5x_args(dataset: str, steps: int,
                                   max_train_objects: int = 0) -> list[str]:
    """Early learned surface-token decoder, 5x capacity.

    The previous bold decoder added many learned residual modules, but most of
    them acted after RGBD lifting and voxel/candidate decisions. This preset
    makes the trainable surface-token decoder the main geometry/color owner:
    RGBD points are only the initialization, and voxel fusion is disabled.

    Capacity target: the default surface-token decoder is ~5.0M params
    (h256/slots256/layers3). This preset is ~25.0M params
    (h448/slots448/layers5), roughly 5x learned capacity in the path that
    directly emits Gaussians.
    """
    args = _quality_base_args(
        dataset,
        steps,
        f"{dataset}_surface_token_5x_{steps}",
        eval_every=max(10, min(50, steps // 2 if steps >= 2 else 1)),
        lr=1e-4,
        max_train_objects=max_train_objects,
    )

    def set_arg(flag: str, value: str) -> None:
        _set_arg(args, flag, value)

    set_arg("--freeze_decoder", "1")
    set_arg("--use_surface_token_decoder", "1")
    set_arg("--surface_token_hidden", "448")
    set_arg("--surface_token_slots", "448")
    set_arg("--surface_token_layers", "5")
    set_arg("--surface_token_heads", "8")
    set_arg("--surface_token_grid_h", "64")
    set_arg("--surface_token_grid_w", "96")
    set_arg("--surface_token_mean_res_frac", "0.08")
    set_arg("--surface_token_rgb_res_scale", "0.50")
    set_arg("--surface_token_scale_frac", "0.0040")
    set_arg("--surface_token_normal_scale_frac", "0.00035")
    set_arg("--surface_token_scale_res_scale", "1.5")
    set_arg("--surface_token_quat_res_scale", "0.35")
    set_arg("--surface_token_opacity_init", "0.85")
    set_arg("--anchor_views", "4")
    set_arg("--anchor_render_mode", "concat")
    set_arg("--anchor_blend_topk", "1")
    set_arg("--fusion_voxel_size_frac", "0.0")
    set_arg("--fusion_sh_degree", "0")
    set_arg("--image_condition", "0")
    set_arg("--image_head_skip", "0")
    set_arg("--image_depth_condition", "0")
    set_arg("--image_depth_skip", "0")
    set_arg("--image_normal_condition", "0")
    set_arg("--condition_color_calibration", "train_affine")
    set_arg("--condition_color_calib_max_objects", "8")
    set_arg("--condition_color_calib_views", "9")
    set_arg("--condition_rgb_inpaint_px", "2")
    set_arg("--condition_mask_erode_px", "1")
    set_arg("--condition_unsharp_amount", "0.5")
    set_arg("--condition_contrast", "1.02")
    set_arg("--condition_saturation", "1.02")
    set_arg("--accum", "1")
    set_arg("--lr", "1e-4")
    set_arg("--warmup", "20")
    set_arg("--fg_weight", "20")
    set_arg("--mask_weight", "1.25")
    set_arg("--fg_alpha_weight", "2.0")
    set_arg("--bg_alpha_weight", "0.08")
    set_arg("--grad_weight", "0.03")
    set_arg("--grad_start", "0")
    set_arg("--grad_ramp", "50")
    set_arg("--depth_weight", "0.03")
    set_arg("--depth_render_scale", "0.5")
    set_arg("--n_train_eval", "4")
    set_arg("--n_heldout_eval", "16")
    set_arg("--eval_views_per_object", "8")
    set_arg("--save_eval_viz", "1")
    set_arg("--save_eval_viz_views", "2")
    set_arg("--save_eval_viz_heldout_count", "4")
    set_arg("--save_eval_viz_train_count", "1")
    set_arg("--eval_before_train", "1")
    set_arg("--eval_at_step0", "0")
    set_arg("--save_checkpoints", "1")
    set_arg("--save_named_checkpoints", "1")
    set_arg("--min_free_vram_gb", "24")
    return args


def _quality_surface_token_sharp_args(dataset: str, steps: int,
                                      max_train_objects: int = 0) -> list[str]:
    """Surface-token decoder with anti-fog/sharp-surface priors.

    The first 5x surface-token run learned heldout average color quickly, but it
    did so by expanding low-frequency alpha support. This preset keeps the
    structural decoder and capacity, then changes the prior/objective:

    - moderately smaller tangential/normal splats,
    - lower but still render-visible opacity initialization,
    - stronger background alpha, alpha-edge, and image-gradient penalties,
    - lower LR so opacity/scale do not balloon before geometry settles.
    """
    args = _quality_surface_token_5x_args(
        dataset, steps, max_train_objects=max_train_objects
    )

    def set_arg(flag: str, value: str) -> None:
        _set_arg(args, flag, value)

    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_surface_token_sharp_{steps}")
    set_arg("--lr", "5e-5")
    set_arg("--warmup", "30")
    set_arg("--surface_token_scale_frac", "0.0032")
    set_arg("--surface_token_normal_scale_frac", "0.00030")
    set_arg("--surface_token_opacity_init", "0.82")
    set_arg("--surface_token_mean_res_frac", "0.06")
    set_arg("--surface_token_scale_res_scale", "1.25")
    set_arg("--surface_token_quat_res_scale", "0.20")
    set_arg("--mask_weight", "1.5")
    set_arg("--fg_alpha_weight", "2.0")
    set_arg("--bg_alpha_weight", "0.12")
    set_arg("--grad_weight", "0.06")
    set_arg("--grad_start", "0")
    set_arg("--grad_ramp", "25")
    set_arg("--alpha_grad_weight", "0.25")
    set_arg("--alpha_grad_start", "0")
    set_arg("--alpha_grad_ramp", "25")
    set_arg("--alpha_grad_band_px", "2")
    set_arg("--depth_weight", "0.05")
    return args


def _quality_surface_token_balanced_args(dataset: str, steps: int,
                                         max_train_objects: int = 0) -> list[str]:
    """5x surfel decoder with regularized scale growth.

    The sharp preset under-covers because it shrinks the initial surfels too
    aggressively. This keeps the successful 5x surfel initialization close to
    the step-0 prior, but adds explicit scale/mean regularization so training
    cannot improve PSNR merely by swelling splats into low-frequency blobs.
    """
    args = _quality_surface_token_5x_args(
        dataset, steps, max_train_objects=max_train_objects
    )

    def set_arg(flag: str, value: str) -> None:
        _set_arg(args, flag, value)

    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_surface_token_balanced_{steps}")
    set_arg("--lr", "7.5e-5")
    set_arg("--warmup", "25")
    set_arg("--surface_token_scale_frac", "0.0032")
    set_arg("--surface_token_normal_scale_frac", "0.00028")
    set_arg("--surface_token_mean_res_frac", "0.06")
    set_arg("--surface_token_scale_res_scale", "1.0")
    set_arg("--surface_token_quat_res_scale", "0.30")
    set_arg("--surface_token_opacity_init", "0.85")
    set_arg("--surface_token_scale_reg_weight", "0.10")
    set_arg("--surface_token_tangent_scale_max_frac", "0.0032")
    set_arg("--surface_token_normal_scale_max_frac", "0.00028")
    set_arg("--surface_token_mean_reg_weight", "0.05")
    set_arg("--fg_alpha_weight", "1.2")
    set_arg("--bg_alpha_weight", "0.16")
    set_arg("--mask_weight", "1.25")
    set_arg("--grad_weight", "0.05")
    set_arg("--grad_start", "0")
    set_arg("--grad_ramp", "25")
    set_arg("--alpha_grad_weight", "0.12")
    set_arg("--alpha_grad_start", "0")
    set_arg("--alpha_grad_ramp", "25")
    set_arg("--alpha_grad_band_px", "2")
    set_arg("--depth_weight", "0.04")
    return args


def _quality_canonical_voxel_args(dataset: str, steps: int,
                                  max_train_objects: int = 0) -> list[str]:
    """Bold learned canonical-voxel decoder.

    This is the next step after the surface-token runs: instead of rendering
    one Gaussian for every sampled source pixel, it first consolidates RGBD
    observations into canonical occupied voxels and then learns 3D message
    passing plus full latent-grid cross-attention before emitting Gaussians.
    """
    args = _quality_base_args(
        dataset,
        steps,
        f"{dataset}_canonical_voxel_{steps}",
        eval_every=max(10, min(50, steps // 2 if steps >= 2 else 1)),
        lr=7.5e-5,
        max_train_objects=max_train_objects,
    )

    def set_arg(flag: str, value: str) -> None:
        _set_arg(args, flag, value)

    set_arg("--freeze_decoder", "1")
    set_arg("--use_canonical_voxel_decoder", "1")
    set_arg("--canonical_voxel_hidden", "384")
    set_arg("--canonical_voxel_layers", "5")
    set_arg("--canonical_voxel_heads", "8")
    set_arg("--canonical_voxel_grid_h", "72")
    set_arg("--canonical_voxel_grid_w", "108")
    set_arg("--canonical_voxel_latent_pool", "2")
    set_arg("--canonical_voxel_message_radius", "1")
    set_arg("--canonical_voxel_size_frac", "0.0025")
    set_arg("--canonical_voxel_max_voxels", "45000")
    set_arg("--canonical_voxel_gaussians_per_voxel", "4")
    set_arg("--canonical_voxel_child_offset_mult", "0.38")
    set_arg("--canonical_voxel_mean_res_voxels", "0.75")
    set_arg("--canonical_voxel_rgb_res_scale", "0.30")
    set_arg("--canonical_voxel_tangent_scale_mult", "0.85")
    set_arg("--canonical_voxel_normal_scale_mult", "0.08")
    set_arg("--canonical_voxel_scale_res_scale", "0.60")
    set_arg("--canonical_voxel_quat_res_scale", "0.18")
    set_arg("--canonical_voxel_opacity_init", "0.88")
    set_arg("--canonical_voxel_opacity_support_floor", "0.45")
    set_arg("--canonical_voxel_opacity_support_target", "2.0")
    set_arg("--canonical_voxel_detail_sampling", "1")
    set_arg("--canonical_voxel_detail_color_mix", "0.85")
    set_arg("--canonical_voxel_detail_depth_tol_frac", "0.012")
    set_arg("--canonical_voxel_detail_score_temp", "0.55")
    set_arg("--canonical_voxel_detail_chunk", "8192")
    set_arg("--anchor_views", "5")
    set_arg("--anchor_render_mode", "concat")
    set_arg("--fusion_voxel_size_frac", "0.0")
    set_arg("--fusion_sh_degree", "0")
    set_arg("--image_condition", "0")
    set_arg("--image_head_skip", "0")
    set_arg("--image_depth_condition", "0")
    set_arg("--image_depth_skip", "0")
    set_arg("--image_normal_condition", "0")
    set_arg("--scale_cap_frac", "0.002")
    set_arg("--condition_color_calibration", "train_affine")
    set_arg("--condition_color_calib_max_objects", "8")
    set_arg("--condition_color_calib_views", "9")
    set_arg("--condition_rgb_inpaint_px", "2")
    set_arg("--condition_mask_erode_px", "1")
    set_arg("--condition_unsharp_amount", "0.75")
    set_arg("--condition_contrast", "1.03")
    set_arg("--condition_saturation", "1.02")
    set_arg("--accum", "1")
    set_arg("--lr", "7.5e-5")
    set_arg("--warmup", "25")
    set_arg("--fg_weight", "18")
    set_arg("--mask_weight", "1.2")
    set_arg("--fg_alpha_weight", "1.6")
    set_arg("--bg_alpha_weight", "0.12")
    set_arg("--grad_weight", "0.06")
    set_arg("--grad_start", "0")
    set_arg("--grad_ramp", "25")
    set_arg("--alpha_grad_weight", "0.18")
    set_arg("--alpha_grad_start", "0")
    set_arg("--alpha_grad_ramp", "25")
    set_arg("--alpha_grad_band_px", "2")
    set_arg("--depth_weight", "0.04")
    set_arg("--depth_render_scale", "0.5")
    set_arg("--n_train_eval", "4")
    set_arg("--n_heldout_eval", "16")
    set_arg("--eval_views_per_object", "8")
    set_arg("--save_eval_viz", "1")
    set_arg("--save_eval_viz_views", "2")
    set_arg("--save_eval_viz_heldout_count", "4")
    set_arg("--save_eval_viz_train_count", "1")
    set_arg("--eval_before_train", "1")
    set_arg("--eval_at_step0", "0")
    set_arg("--save_checkpoints", "1")
    set_arg("--save_named_checkpoints", "1")
    set_arg("--min_free_vram_gb", "24")
    return args


def _quality_max_learned_canonical_args(dataset: str, steps: int,
                                        max_train_objects: int = 0) -> list[str]:
    """Maximum-learned canonical voxel decoder.

    This is the aggressive path after the first canonical voxel pilots. It keeps
    RGBD lifting only as the feed-forward proposal mechanism, then gives the
    learned decoder substantially more control:
    - Transformer context over the LTX latent tokens.
    - Global learned scene-memory slots inside every voxel block.
    - Shared per-view CNN features sampled at projected voxel locations.
    - Weaker opacity prior so visibility is learned from target-depth labels.
    """
    args = _quality_canonical_voxel_args(
        dataset, steps, max_train_objects=max_train_objects
    )

    def set_arg(flag: str, value: str) -> None:
        _set_arg(args, flag, value)

    eval_every = max(10, min(40, steps // 3 if steps >= 3 else 1))
    set_arg("--out_dir", f"{RUNS_DIR}/{dataset}_max_learned_canonical_{steps}")
    set_arg("--eval_every", str(eval_every))
    set_arg("--save_every", str(eval_every))
    set_arg("--canonical_voxel_hidden", "448")
    set_arg("--canonical_voxel_layers", "6")
    set_arg("--canonical_voxel_heads", "8")
    set_arg("--canonical_voxel_latent_layers", "2")
    set_arg("--canonical_voxel_scene_slots", "64")
    set_arg("--canonical_voxel_grid_h", "80")
    set_arg("--canonical_voxel_grid_w", "120")
    set_arg("--canonical_voxel_latent_pool", "1")
    set_arg("--canonical_voxel_size_frac", "0.0024")
    set_arg("--canonical_voxel_max_voxels", "50000")
    set_arg("--canonical_voxel_gaussians_per_voxel", "6")
    set_arg("--canonical_voxel_child_offset_mult", "0.44")
    set_arg("--canonical_voxel_mean_res_voxels", "1.10")
    set_arg("--canonical_voxel_rgb_res_scale", "0.65")
    set_arg("--canonical_voxel_tangent_scale_mult", "0.88")
    set_arg("--canonical_voxel_normal_scale_mult", "0.075")
    set_arg("--canonical_voxel_scale_res_scale", "0.90")
    set_arg("--canonical_voxel_quat_res_scale", "0.24")
    set_arg("--canonical_voxel_opacity_init", "0.78")
    set_arg("--canonical_voxel_opacity_support_floor", "0.22")
    set_arg("--canonical_voxel_opacity_support_target", "2.5")
    set_arg("--canonical_voxel_opacity_prior_weight", "0.45")
    set_arg("--canonical_voxel_view_feature_channels", "48")
    set_arg("--canonical_voxel_view_feature_scale", "0.5")
    set_arg("--canonical_voxel_detail_color_mix", "0.62")
    set_arg("--canonical_voxel_detail_depth_tol_frac", "0.010")
    set_arg("--canonical_voxel_detail_score_temp", "0.45")
    set_arg("--canonical_voxel_detail_chunk", "4096")
    set_arg("--anchor_views", "7")
    set_arg("--lr", "3e-5")
    set_arg("--warmup", "20")
    set_arg("--fg_weight", "18")
    set_arg("--mask_weight", "1.15")
    set_arg("--fg_alpha_weight", "1.35")
    set_arg("--bg_alpha_weight", "0.14")
    set_arg("--grad_weight", "0.055")
    set_arg("--grad_start", "15")
    set_arg("--grad_ramp", "15")
    set_arg("--alpha_grad_weight", "0.16")
    set_arg("--alpha_grad_start", "15")
    set_arg("--alpha_grad_ramp", "15")
    set_arg("--depth_weight", "0.035")
    set_arg("--scale_reg", "0.0012")
    set_arg("--detail_teacher_weight", "0.10")
    set_arg("--detail_teacher_start", "35")
    set_arg("--detail_teacher_ramp", "10")
    set_arg("--detail_teacher_alpha_min", "0.35")
    set_arg("--detail_teacher_edge_thresh", "0.025")
    set_arg("--detail_teacher_artifact_weight", "0.08")
    set_arg("--canonical_target_vis_weight", "0.06")
    set_arg("--canonical_target_vis_pos_min", "0.70")
    set_arg("--canonical_target_vis_neg_max", "0.30")
    set_arg("--canonical_target_vis_positive_weight", "2.0")
    set_arg("--canonical_target_vis_negative_weight", "1.6")
    set_arg("--canonical_source_vis_learned_refine", "1")
    set_arg("--canonical_source_vis_refine_hidden", "192")
    set_arg("--canonical_source_vis_refine_opacity_strength", "1.15")
    set_arg("--canonical_source_vis_refine_rgb_scale", "0.08")
    set_arg("--canonical_source_vis_refine_scale_res_scale", "0.18")
    set_arg("--canonical_source_vis_distill_weight", "0.15")
    set_arg("--canonical_source_vis_min_support", "1.5")
    set_arg("--canonical_source_vis_conflict_weight", "0.8")
    set_arg("--canonical_source_vis_softness", "0.75")
    set_arg("--canonical_source_vis_floor", "0.25")
    set_arg("--n_train_eval", "4")
    set_arg("--n_heldout_eval", "16")
    set_arg("--eval_views_per_object", "8")
    set_arg("--save_eval_viz_views", "2")
    set_arg("--save_eval_viz_heldout_count", "4")
    set_arg("--save_eval_viz_train_count", "1")
    set_arg("--min_free_vram_gb", "18")
    return args


@app.local_entrypoint()
def main(action: str = "info", steps: int = 1000, extra: str = "",
         force: bool = False, limit: int = 0, dataset: str = DATASET,
         manifest: str = MANIFEST, archive: str = "",
         depth_archive: str = "", split: str = "all") -> None:
    extra_args = shlex.split(extra) if extra else []

    def with_extra(args: list[str]) -> list[str]:
        return [*args, *extra_args]

    if action == "upload":
        for p in (LOCAL_RGB_ZIP, LOCAL_DEPTH_ZIP, LOCAL_MANIFEST):
            if not p.exists():
                raise FileNotFoundError(p)
        print("[upload] uploading RGB/depth archives + manifest to Modal volume", flush=True)
        with data_volume.batch_upload(force=True) as batch:
            batch.put_file(str(LOCAL_RGB_ZIP), "uploads/objaverse_v5_final.zip")
            batch.put_file(str(LOCAL_DEPTH_ZIP), "uploads/objaverse_v5_depth.zip")
            batch.put_file(str(LOCAL_MANIFEST), f"uploads/{MANIFEST}")
        print("[upload] done", flush=True)
    elif action == "upload_archive":
        archive_path = Path(archive)
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        print(f"[upload_archive] uploading {archive_path} to Modal data volume", flush=True)
        with data_volume.batch_upload(force=True) as batch:
            batch.put_file(str(archive_path), f"uploads/{archive_path.name}")
        print("[upload_archive] done", flush=True)
    elif action == "upload_ltx_vae":
        if not LOCAL_LTX_RAW_VAE.exists():
            raise FileNotFoundError(LOCAL_LTX_RAW_VAE)
        print("[upload_ltx_vae] uploading VAE-only checkpoint to Modal weights volume", flush=True)
        with weights_volume.batch_upload(force=True) as batch:
            batch.put_file(str(LOCAL_LTX_RAW_VAE), "ltx23/plain_int8-vae-bf16-backup.safetensors")
        print("[upload_ltx_vae] done", flush=True)
    elif action == "prepare":
        archive_name = Path(archive).name if archive else ""
        depth_archive_name = Path(depth_archive).name if depth_archive else ""
        print(prepare_dataset.remote(
            force=force,
            dataset=dataset,
            archive_name=archive_name,
            depth_archive_name=depth_archive_name,
            manifest_name=manifest,
        ))
    elif action == "smoke":
        smoke.remote(dataset=dataset, manifest=manifest)
    elif action == "download_ltx":
        print(download_ltx_checkpoint.remote())
    elif action == "compile_gsplat":
        print(compile_gsplat.remote())
    elif action == "spconv_smoke":
        print(spconv_smoke.remote())
    elif action == "sparse_voxel_fusion_smoke":
        print(sparse_voxel_fusion_smoke.remote())
    elif action == "quality_sparse_voxel_step0_check":
        # Step-0 invariant on REAL data: must reproduce prior to within float noise.
        train_sparse_voxel.remote(
            with_extra(_quality_sparse_voxel_args(
                dataset, steps=1, lr=0.0,
                name=f"{dataset}_sparse_voxel_step0_check",
                eval_every=1, max_train_objects=0,
            )),
            dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_smoke":
        train_sparse_voxel.remote(
            with_extra(_quality_sparse_voxel_args(
                dataset, steps=200, lr=2e-4,
                name=f"{dataset}_sparse_voxel_smoke200",
                eval_every=50, max_train_objects=32,
            )),
            dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_pilot":
        # Unique suffix per fire so parallel pilots don't collide on output dir.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        train_sparse_voxel.remote(
            with_extra(_quality_sparse_voxel_args(
                dataset, steps=steps, lr=2e-5,
                name=f"{dataset}_sparse_voxel_pilot{steps}_v14_{suffix}",
                eval_every=250, max_train_objects=0,
            )),
            dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_oracle_pilot":
        # v1.8: train with GT-depth oracle splat placement.  Prior at ~20 dB.
        # Can the head improve a STRONG prior?  Tests architecture's predictive
        # power independent of depth quality.  Note: at inference (without GT
        # depth) this won't deploy directly, but it teaches us whether the head
        # CAN learn helpful patterns when input is high quality.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        v18_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=5e-5,
                                              name=f"{dataset}_sparse_voxel_pilot{steps}_v18_{suffix}",
                                              eval_every=250, max_train_objects=0)
        v18_args.extend([
            "--oracle_anchor_depth", "1",       # NEW: GT depth for splat placement
            "--sparse_voxel_vis_delta", "0.3",
            "--sparse_voxel_identity_reg_weight", "1.0",
            "--sparse_voxel_opacity_res_scale", "0.05",
            "--mask_weight", "0.05",
            "--fg_weight", "30",
        ])
        train_sparse_voxel.remote(
            with_extra(v18_args), dataset=dataset, manifest=manifest,
        )
    elif action == "quality_depth_refine_clean_pilot":
        # Option C done properly: colorcal_step0 fusion config (spray-controlled)
        # + DepthRefineUNet trained on DA3 -> GT depth supervision.
        # Apples-to-apples vs 18.33 baseline; aims at GT-depth ceiling 19.96.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        oc_args = _quality_colorcal_step0_args(dataset)
        # Override base step/lr/save fields
        def _set(flag, val):
            if flag in oc_args:
                oc_args[oc_args.index(flag) + 1] = str(val)
            else:
                oc_args.extend([flag, str(val)])
        _set("--steps", str(steps))
        _set("--lr", "2e-4")
        _set("--warmup", "100")
        _set("--max_train_objects", "0")
        _set("--eval_every", "250")
        _set("--save_every", "250")
        _set("--out_dir", f"{RUNS_DIR}/{dataset}_depth_refine_clean{steps}_{suffix}")
        oc_args.extend([
            "--freeze_decoder", "1",                                  # only refine head trains
            "--depth_refine_unet", "1",
            "--depth_refine_hidden", "32",
            "--depth_refine_delta_scale", "0.12",
            "--depth_refine_gt_weight", "1.0",                        # direct GT supervision
            "--depth_refine_gt_outlier_weight", "1.0",                # focus on biggest errors
            "--depth_refine_multiview_features", "1",                 # cross-view consistency
            "--depth_refine_prior_weight", "0.0005",
            "--depth_refine_tv_weight", "0.0005",
            "--save_named_checkpoints", "1",
        ])
        train.remote(with_extra(oc_args), dataset=dataset, manifest=manifest)
    elif action == "quality_sparse_voxel_gtdepth":
        # v3.1: --cond_depth_subdir depth → loads from {obj_dir}/depth/depth_NNN.npy
        # which is the GT depth directly (where load_depth_view also looks).
        # Replaces DA3 (da3_ltx) with GT depth at the model input.  Prior at
        # this config achieves the documented ~20 dB oracle.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        v31_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=1e-5,
                                              name=f"{dataset}_sparse_voxel_gtdepth{steps}_{suffix}",
                                              eval_every=250, max_train_objects=0)
        # Override cond_depth_subdir from da3_ltx → depth (GT)
        if "--cond_depth_subdir" in v31_args:
            idx = v31_args.index("--cond_depth_subdir")
            v31_args[idx + 1] = "depth"
        v31_args.extend([
            "--sparse_voxel_enhance_only", "1",
            "--sparse_voxel_vis_delta", "0.3",
            "--sparse_voxel_opacity_res_scale", "0.1",
            "--sparse_voxel_depth_res_frac", "0.0",
            "--sparse_voxel_rgb_res_scale", "0.02",
            "--sparse_voxel_identity_reg_weight", "1.0",
            "--mask_weight", "0.05",
            "--fg_weight", "30",
        ])
        train_sparse_voxel.remote(
            with_extra(v31_args), dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_great_quality":
        # v3.0: GT depth at conditioning views + sparse-voxel head.
        # This is the bold "actually great visual quality" path: the prior
        # at this input config reaches ~20 dB (vs 18.33 with DA3 depth).
        # We use v1.7's safe sparse-voxel config so the head doesn't break
        # the great prior either.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        v30_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=1e-5,
                                              name=f"{dataset}_sparse_voxel_great{steps}_{suffix}",
                                              eval_every=250, max_train_objects=0)
        v30_args.extend([
            "--cond_use_target_depth", "1",          # GT DEPTH AT INPUT (the key win)
            "--sparse_voxel_enhance_only", "1",
            "--sparse_voxel_vis_delta", "0.3",
            "--sparse_voxel_opacity_res_scale", "0.1",
            "--sparse_voxel_depth_res_frac", "0.0",
            "--sparse_voxel_rgb_res_scale", "0.02",
            "--sparse_voxel_identity_reg_weight", "1.0",
            "--mask_weight", "0.05",
            "--fg_weight", "30",
        ])
        train_sparse_voxel.remote(
            with_extra(v30_args), dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_fg500_pilot":
        # v2.0: same as v1.9 but EXTREME fg_weight=500.
        # At fg_weight=500: bg loss contribution is ~4% of total, fg ~96%.
        # The optimizer essentially can't see bg pixels - removes deletion
        # gradient entirely.  If this fails, the deletion path isn't bg-loss-
        # driven and the issue is elsewhere.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        v20_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=5e-5,
                                              name=f"{dataset}_sparse_voxel_pilot{steps}_v20_{suffix}",
                                              eval_every=250, max_train_objects=0)
        v20_args.extend([
            "--sparse_voxel_enhance_only", "1",
            "--sparse_voxel_vis_delta", "0.5",
            "--sparse_voxel_opacity_res_scale", "0.15",
            "--sparse_voxel_depth_res_frac", "0.0",
            "--sparse_voxel_rgb_res_scale", "0.05",
            "--sparse_voxel_identity_reg_weight", "0.3",
            "--mask_weight", "0.0",      # NO mask loss
            "--fg_weight", "500",        # EXTREME fg dominance
        ])
        train_sparse_voxel.remote(
            with_extra(v20_args), dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_enhance_only_loose_pilot":
        # v1.9: enhance-only + no depth, but MORE freedom than v1.7 (which
        # was too tight to learn anything — PSNR flat, FN unchanged).
        # Higher lr, lower ID reg, wider vis_delta.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        v19_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=5e-5,
                                              name=f"{dataset}_sparse_voxel_pilot{steps}_v19_{suffix}",
                                              eval_every=250, max_train_objects=0)
        v19_args.extend([
            "--sparse_voxel_enhance_only", "1",
            "--sparse_voxel_vis_delta", "0.5",         # wider boost range
            "--sparse_voxel_opacity_res_scale", "0.15",
            "--sparse_voxel_depth_res_frac", "0.0",     # NO depth shifts
            "--sparse_voxel_rgb_res_scale", "0.05",     # small color shifts
            "--sparse_voxel_identity_reg_weight", "0.3",  # 3x less than v1.7
            "--mask_weight", "0.05",
            "--fg_weight", "30",
        ])
        train_sparse_voxel.remote(
            with_extra(v19_args), dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_enhance_only_loose_spawn":
        # Detached version for longer spconv runs.  Blocking Modal CLI sessions
        # can interrupt the remote function if the local process is stopped;
        # spawn lets us poll /runs for metrics instead.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        v19_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=5e-5,
                                              name=f"{dataset}_sparse_voxel_pilot{steps}_v19_{suffix}",
                                              eval_every=250, max_train_objects=0)
        v19_args.extend([
            "--sparse_voxel_enhance_only", "1",
            "--sparse_voxel_vis_delta", "0.5",
            "--sparse_voxel_opacity_res_scale", "0.15",
            "--sparse_voxel_depth_res_frac", "0.0",
            "--sparse_voxel_rgb_res_scale", "0.05",
            "--sparse_voxel_identity_reg_weight", "0.3",
            "--mask_weight", "0.05",
            "--fg_weight", "30",
        ])
        call = train_sparse_voxel.spawn(
            with_extra(v19_args), dataset=dataset, manifest=manifest,
        )
        print(f"[spawn] sparse train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_sparse_voxel_enhance_only_safe_pilot":
        # v1.7: enhance-only + NO depth_res + tiny rgb_res.
        # v1.6 found that depth_res can shift splats off their original pixel
        # (effective deletion via position).  Disable depth entirely.
        # rgb_res tiny since color-only (v1.3) showed even rgb drifts down.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        v17_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=1e-5,
                                              name=f"{dataset}_sparse_voxel_pilot{steps}_v17_{suffix}",
                                              eval_every=250, max_train_objects=0)
        v17_args.extend([
            "--sparse_voxel_enhance_only", "1",
            "--sparse_voxel_vis_delta", "0.3",
            "--sparse_voxel_opacity_res_scale", "0.1",
            "--sparse_voxel_depth_res_frac", "0.0",     # NO depth shifts
            "--sparse_voxel_rgb_res_scale", "0.02",     # tiny color shifts
            "--sparse_voxel_identity_reg_weight", "1.0",
            "--mask_weight", "0.05",
            "--fg_weight", "30",
        ])
        train_sparse_voxel.remote(
            with_extra(v17_args), dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_enhance_only_pilot":
        # v1.6: enhance-only architecture.  vis ∈ [1, 1+2δ], op_res ∈ [0, ...].
        # Architecturally impossible to delete splats.  Only color shifts are
        # bidirectional (colors need both directions).
        import time as _t
        suffix = _t.strftime("%H%M%S")
        v16_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=5e-5,
                                              name=f"{dataset}_sparse_voxel_pilot{steps}_v16_{suffix}",
                                              eval_every=250, max_train_objects=0)
        v16_args.extend([
            "--sparse_voxel_enhance_only", "1",
            "--sparse_voxel_vis_delta", "0.3",   # allows vis up to 1.6 (60% boost)
            "--sparse_voxel_opacity_res_scale", "0.1",  # allows up to +0.2 opacity
            # mask_weight low since enhance-only can't fix bg-spray anyway
            "--mask_weight", "0.05",
        ])
        train_sparse_voxel.remote(
            with_extra(v16_args), dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_l2_pilot":
        # v1.5: L2 (MSE) photometric loss instead of L1.  Aligns optimization
        # with PSNR exactly (PSNR ≡ -10·log10(MSE)).  Otherwise same as v1.4.
        import time as _t
        suffix = _t.strftime("%H%M%S")
        v15_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=2e-5,
                                              name=f"{dataset}_sparse_voxel_pilot{steps}_v15_{suffix}",
                                              eval_every=250, max_train_objects=0)
        v15_args.extend(["--photometric_loss_type", "l2"])
        train_sparse_voxel.remote(
            with_extra(v15_args), dataset=dataset, manifest=manifest,
        )
    elif action == "quality_sparse_voxel_color_only_pilot":
        # v1.3: color residuals ONLY (vis=1, op_res=0, depth_res=0).
        # Color shifts cannot cause deletion → safe regime to prove the
        # architecture trains at all without destabilizing the prior.
        # If this beats prior, add geometry/opacity channels back step by step.
        v13_args = _quality_sparse_voxel_args(dataset, steps=steps, lr=5e-5,
                                              name=f"{dataset}_sparse_voxel_color_only{steps}",
                                              eval_every=250, max_train_objects=0)
        # Override the v1.2 aggressive defaults with the color-only config:
        def _set(flag, val):
            if flag in v13_args:
                v13_args[v13_args.index(flag) + 1] = str(val)
            else:
                v13_args.extend([flag, str(val)])
        _set("--sparse_voxel_vis_delta", "0.0")
        _set("--sparse_voxel_opacity_res_scale", "0.0")
        _set("--sparse_voxel_depth_res_frac", "0.0")
        _set("--sparse_voxel_rgb_res_scale", "0.1")
        _set("--sparse_voxel_identity_reg_weight", "0.0")  # no need: no deletion possible
        _set("--mask_weight", "0.5")  # restore default
        train_sparse_voxel.remote(
            with_extra(v13_args), dataset=dataset, manifest=manifest,
        )
    elif action == "phase1_depth_probe":
        train.remote(
            _phase1_depth_args(steps, f"phase1_depth_probe_{steps}", eval_every=max(10, steps // 2)),
            dataset=dataset,
            manifest=manifest,
        )
    elif action == "phase3_offset_probe":
        train.remote(
            _phase3_offset_args(steps, f"phase3_offset_probe_{steps}", eval_every=max(10, steps // 2)),
            dataset=dataset,
            manifest=manifest,
        )
    elif action == "phase1_depth_pilot":
        train.remote(_phase1_depth_args(steps), dataset=dataset, manifest=manifest)
    elif action == "phase3_offset_pilot":
        train.remote(_phase3_offset_args(steps), dataset=dataset, manifest=manifest)
    elif action == "densified_smoke":
        densified.remote(
            [
                "--steps", "20",
                "--init_gaussians", "1024",
                "--train_views", "2",
                "--eval_views", "2",
                "--eval_every", "10",
                "--eval_render_chunk", "1",
                "--refine_start", "100",
                "--refine_stop", "100",
                "--max_gaussians", "4096",
                "--min_free_vram_gb", "40",
                "--out_dir", f"{RUNS_DIR}/densified_smoke",
            ],
            dataset=dataset,
            manifest=manifest,
        )
    elif action == "densified":
        densified.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "decode_ltx":
        decode_ltx.remote(
            split=split, limit=limit, overwrite=force, extra_args=extra.split(),
            dataset=dataset, manifest=manifest,
        )
    elif action == "decode_ltx_spawn":
        call = decode_ltx.spawn(
            split=split, limit=limit, overwrite=force, extra_args=extra.split(),
            dataset=dataset, manifest=manifest,
        )
        print(f"[spawn] decode_ltx call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "decode_ltx_shards_spawn":
        shard_size = max(int(steps), 1)
        n_shards = max(int(limit), 1)
        for shard_i in range(n_shards):
            offset = shard_i * shard_size
            shard_extra = [*extra_args, "--offset", str(offset)]
            call = decode_ltx.spawn(
                split=split,
                limit=shard_size,
                overwrite=force,
                extra_args=shard_extra,
                dataset=dataset,
                manifest=manifest,
            )
            print(
                f"[spawn] decode_ltx shard={shard_i} "
                f"offset={offset} limit={shard_size} "
                f"call id: {getattr(call, 'object_id', call)}",
                flush=True,
            )
    elif action == "predict_depth":
        predict_depth_anything.remote(split=split, limit=limit, overwrite=force,
                                      extra_args=extra.split(), dataset=dataset,
                                      manifest=manifest)
    elif action == "predict_vggt_depth":
        predict_depth_vggt.remote(split=split, limit=limit, overwrite=force,
                                  extra_args=extra.split(), dataset=dataset,
                                  manifest=manifest)
    elif action == "predict_da3_depth":
        predict_depth_da3.remote(split=split, limit=limit, overwrite=force,
                                 extra_args=extra.split(), dataset=dataset,
                                 manifest=manifest)
    elif action == "predict_da3_depth_spawn":
        call = predict_depth_da3.spawn(
            split=split, limit=limit, overwrite=force, extra_args=extra.split(),
            dataset=dataset, manifest=manifest,
        )
        print(f"[spawn] predict_da3_depth call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "predict_da3_depth_shards_spawn":
        shard_size = max(int(steps), 1)
        n_shards = max(int(limit), 1)
        for shard_i in range(n_shards):
            offset = shard_i * shard_size
            shard_extra = [*extra_args, "--offset", str(offset)]
            call = predict_depth_da3.spawn(
                split=split,
                limit=shard_size,
                overwrite=force,
                extra_args=shard_extra,
                dataset=dataset,
                manifest=manifest,
            )
            print(
                f"[spawn] predict_da3_depth shard={shard_i} "
                f"offset={offset} limit={shard_size} "
                f"call id: {getattr(call, 'object_id', call)}",
                flush=True,
            )
    elif action == "train":
        train.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "quality_step0":
        train.remote([*_quality_step0_args(dataset), *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_colorcal_step0":
        train.remote([*_quality_colorcal_step0_args(dataset), *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_source_gtdepth_step0":
        train.remote([*_quality_source_gtdepth_step0_args(dataset), *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_colorcal_novelgrid16":
        train.remote([*_novel_grid16_args(dataset), *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_depthcal_step0":
        train.remote([*_quality_depthcal_step0_args(dataset), *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_learned_fill_smoke":
        train.remote([*_quality_learned_fill_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_learned_fill_pilot":
        train.remote([*_quality_learned_fill_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_unet_fill_smoke":
        train.remote([*_quality_unet_fill_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_unet_fill_pilot":
        train.remote([*_quality_unet_fill_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_unet_oracle_fill_smoke":
        train.remote([*_quality_unet_oracle_fill_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_unet_oracle_fill_pilot":
        train.remote([*_quality_unet_oracle_fill_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_depth_refine_smoke":
        train.remote([*_quality_depth_refine_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_depth_refine_pilot":
        train.remote([*_quality_depth_refine_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_support_gate_smoke":
        train.remote([*_quality_support_gate_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_support_gate_pilot":
        train.remote([*_quality_support_gate_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_confidence_smoke":
        train.remote([*_quality_surface_confidence_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_confidence_pilot":
        train.remote([*_quality_surface_confidence_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_confidence_pilot_spawn":
        call = train.spawn(
            [*_quality_surface_confidence_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_surface_depth_confidence_smoke":
        train.remote([*_quality_surface_depth_confidence_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_depth_confidence_pilot":
        train.remote([*_quality_surface_depth_confidence_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_depth_confidence_pilot_spawn":
        call = train.spawn(
            [*_quality_surface_depth_confidence_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_output_alpha_refine_smoke":
        train.remote([*_quality_output_alpha_refine_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_output_alpha_refine_pilot":
        train.remote([*_quality_output_alpha_refine_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_fusion_candidate_hot_smoke":
        train.remote([*_quality_fusion_candidate_hot_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_fusion_candidate_hot_pilot":
        train.remote([*_quality_fusion_candidate_hot_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_fusion_candidate_scoreonly_pilot":
        train.remote([*_quality_fusion_candidate_scoreonly_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_depth_confidence_hot_pilot":
        train.remote([*_quality_depth_confidence_hot_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_depth_confidence_hot_pilot_spawn":
        call = train.spawn(
            [*_quality_depth_confidence_hot_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_depth_affine_hot_pilot":
        train.remote([*_quality_depth_affine_hot_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_depth_affine_hot_pilot_spawn":
        call = train.spawn(
            [*_quality_depth_affine_hot_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_rgbd_depth_refine_smoke":
        train.remote([*_quality_rgbd_depth_refine_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_rgbd_depth_refine_pilot":
        train.remote([*_quality_rgbd_depth_refine_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_rgbd_depth_refine_pilot_spawn":
        call = train.spawn(
            [*_quality_rgbd_depth_refine_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_surface_refine_smoke":
        train.remote([*_quality_surface_refine_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_refine_pilot":
        train.remote([*_quality_surface_refine_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_refine_pilot_spawn":
        call = train.spawn(
            [*_quality_surface_refine_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_condition_rgb_refine_gt_smoke":
        train.remote([*_quality_condition_rgb_refine_gt_args(dataset, steps, max_train_objects=64),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_condition_rgb_refine_gt_pilot":
        train.remote([*_quality_condition_rgb_refine_gt_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_residual_decoder_smoke":
        train.remote([*_quality_residual_decoder_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_residual_decoder_pilot":
        train.remote([*_quality_residual_decoder_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_residual_decoder_pilot_spawn":
        call = train.spawn(
            [*_quality_residual_decoder_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_residual_decoder_aggressive_smoke":
        train.remote([*_quality_residual_decoder_aggressive_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_residual_decoder_aggressive_pilot":
        train.remote([*_quality_residual_decoder_aggressive_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_residual_decoder_aggressive_pilot_spawn":
        call = train.spawn(
            [*_quality_residual_decoder_aggressive_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_residual_decoder_rgb_smoke":
        train.remote([*_quality_residual_decoder_rgb_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_residual_decoder_rgb_pilot":
        train.remote([*_quality_residual_decoder_rgb_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_residual_decoder_rgb_pilot_spawn":
        call = train.spawn(
            [*_quality_residual_decoder_rgb_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_residual_decoder_staged_smoke":
        train.remote([*_quality_residual_decoder_staged_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_residual_decoder_staged_pilot":
        train.remote([*_quality_residual_decoder_staged_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_residual_decoder_staged_pilot_spawn":
        call = train.spawn(
            [*_quality_residual_decoder_staged_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_bold_learned_decoder_smoke":
        train.remote([*_quality_bold_learned_decoder_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_bold_learned_decoder_pilot":
        train.remote([*_quality_bold_learned_decoder_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_bold_learned_decoder_pilot_spawn":
        call = train.spawn(
            [*_quality_bold_learned_decoder_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_bold_visibility_decoder_smoke":
        train.remote([*_quality_bold_visibility_decoder_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_bold_visibility_decoder_pilot":
        train.remote([*_quality_bold_visibility_decoder_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_bold_visibility_decoder_pilot_spawn":
        call = train.spawn(
            [*_quality_bold_visibility_decoder_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_surface_token_5x_smoke":
        train.remote([*_quality_surface_token_5x_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_token_5x_pilot":
        train.remote([*_quality_surface_token_5x_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_token_5x_pilot_spawn":
        call = train.spawn(
            [*_quality_surface_token_5x_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_surface_token_sharp_smoke":
        train.remote([*_quality_surface_token_sharp_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_token_sharp_pilot":
        train.remote([*_quality_surface_token_sharp_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_token_sharp_pilot_spawn":
        call = train.spawn(
            [*_quality_surface_token_sharp_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_surface_token_balanced_smoke":
        train.remote([*_quality_surface_token_balanced_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_token_balanced_pilot":
        train.remote([*_quality_surface_token_balanced_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_surface_token_balanced_pilot_spawn":
        call = train.spawn(
            [*_quality_surface_token_balanced_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_canonical_voxel_smoke":
        train.remote([*_quality_canonical_voxel_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_canonical_voxel_pilot":
        train.remote([*_quality_canonical_voxel_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_canonical_voxel_pilot_spawn":
        call = train.spawn(
            [*_quality_canonical_voxel_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "quality_max_learned_canonical_smoke":
        train.remote([*_quality_max_learned_canonical_args(dataset, steps, max_train_objects=32),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_max_learned_canonical_pilot":
        train.remote([*_quality_max_learned_canonical_args(dataset, steps, max_train_objects=0),
                      *extra.split()],
                     dataset=dataset, manifest=manifest)
    elif action == "quality_max_learned_canonical_pilot_spawn":
        call = train.spawn(
            [*_quality_max_learned_canonical_args(dataset, steps, max_train_objects=0),
             *extra.split()],
            dataset=dataset,
            manifest=manifest,
        )
        print(f"[spawn] train call id: {getattr(call, 'object_id', call)}", flush=True)
    elif action == "condition_oracle":
        condition_oracle.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "pretrain_depth_refine":
        pretrain_depth_refine.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "train_depth_anchor":
        train_depth_anchor.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "bake_depth_anchor":
        bake_depth_anchor.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "bake_depth_refine":
        bake_depth_refine.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "pretrain_condition_rgb":
        pretrain_condition_rgb.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "pretrain_condition_rgbd":
        pretrain_condition_rgbd.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "pretrain_condition_mask":
        pretrain_condition_mask.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "pretrain_surface_confidence":
        pretrain_surface_confidence.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "build_combined_manifest":
        build_combined_manifest.remote(extra.split())
    elif action == "inspect_dataset":
        inspect_dataset_remote.remote(extra.split(), dataset=dataset, manifest=manifest)
    elif action == "condition_coverage":
        inspect_dataset_remote.remote(
            [
                "--condition_coverage", "1",
                "--cond_subdir", "@ltx_decoded" if _is_combined_dataset(dataset) else "ltx_decoded",
                "--cond_depth_subdir", "@da3_ltx" if _is_combined_dataset(dataset) else "da3_ltx",
                "--condition_coverage_min_views", "9",
                "--splits", "train,eval,test",
                *extra.split(),
            ],
            dataset=dataset,
            manifest=manifest,
        )
    else:
        print("actions: upload | upload_archive | upload_ltx_vae | prepare | smoke | phase1_depth_probe | "
              "phase3_offset_probe | phase1_depth_pilot | phase3_offset_pilot | "
              "densified_smoke | download_ltx | compile_gsplat | decode_ltx | "
              "decode_ltx_spawn | decode_ltx_shards_spawn | predict_depth | "
              "predict_vggt_depth | predict_da3_depth | predict_da3_depth_spawn | "
              "predict_da3_depth_shards_spawn | train | densified | "
              "quality_step0 | quality_learned_fill_smoke | "
              "quality_learned_fill_pilot | quality_unet_fill_smoke | "
              "quality_unet_fill_pilot | quality_unet_oracle_fill_smoke | "
              "quality_unet_oracle_fill_pilot | quality_depth_refine_smoke | "
              "quality_depth_refine_pilot | quality_support_gate_smoke | "
              "quality_support_gate_pilot | quality_surface_confidence_smoke | "
              "quality_surface_confidence_pilot | "
              "quality_surface_depth_confidence_smoke | "
              "quality_surface_depth_confidence_pilot | "
              "quality_surface_depth_confidence_pilot_spawn | "
              "quality_residual_decoder_smoke | "
              "quality_residual_decoder_pilot | quality_residual_decoder_pilot_spawn | "
              "quality_residual_decoder_aggressive_smoke | "
              "quality_residual_decoder_aggressive_pilot | "
              "quality_residual_decoder_aggressive_pilot_spawn | "
              "quality_residual_decoder_rgb_smoke | "
              "quality_residual_decoder_rgb_pilot | "
              "quality_residual_decoder_rgb_pilot_spawn | "
              "quality_residual_decoder_staged_smoke | "
              "quality_residual_decoder_staged_pilot | "
              "quality_residual_decoder_staged_pilot_spawn | "
              "quality_bold_learned_decoder_smoke | "
              "quality_bold_learned_decoder_pilot | "
              "quality_bold_learned_decoder_pilot_spawn | "
              "quality_bold_visibility_decoder_smoke | "
              "quality_bold_visibility_decoder_pilot | "
              "quality_bold_visibility_decoder_pilot_spawn | "
              "quality_surface_token_5x_smoke | "
              "quality_surface_token_5x_pilot | "
              "quality_surface_token_5x_pilot_spawn | "
              "quality_surface_token_sharp_smoke | "
              "quality_surface_token_sharp_pilot | "
              "quality_surface_token_sharp_pilot_spawn | "
              "quality_surface_token_balanced_smoke | "
              "quality_surface_token_balanced_pilot | "
              "quality_surface_token_balanced_pilot_spawn | "
              "quality_canonical_voxel_smoke | "
              "quality_canonical_voxel_pilot | "
              "quality_canonical_voxel_pilot_spawn | "
              "quality_max_learned_canonical_smoke | "
              "quality_max_learned_canonical_pilot | "
              "quality_max_learned_canonical_pilot_spawn | "
              "quality_colorcal_step0 | quality_colorcal_novelgrid16 | "
              "quality_depthcal_step0 | quality_output_alpha_refine_smoke | "
              "quality_output_alpha_refine_pilot | "
              "quality_fusion_candidate_hot_smoke | "
              "quality_fusion_candidate_hot_pilot | "
              "quality_fusion_candidate_scoreonly_pilot | "
              "quality_depth_confidence_hot_pilot | "
              "quality_depth_confidence_hot_pilot_spawn | "
              "quality_depth_affine_hot_pilot | "
              "quality_depth_affine_hot_pilot_spawn | "
              "quality_rgbd_depth_refine_smoke | "
              "quality_rgbd_depth_refine_pilot | "
              "quality_rgbd_depth_refine_pilot_spawn | "
              "quality_surface_refine_smoke | "
              "quality_surface_refine_pilot | "
              "quality_surface_refine_pilot_spawn | "
              "quality_condition_rgb_refine_gt_smoke | "
              "quality_condition_rgb_refine_gt_pilot | "
              "pretrain_depth_refine | pretrain_condition_rgb | pretrain_condition_rgbd | pretrain_condition_mask | "
              "pretrain_surface_confidence | "
              "condition_oracle | inspect_dataset | condition_coverage")
