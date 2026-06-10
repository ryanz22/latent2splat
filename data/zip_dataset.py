"""Zip a latent2splat dataset directory from a Modal Volume.

This is the handoff utility for datasets produced into a Modal Volume such as
`latent2splat-renders`. It intentionally writes a plain ZIP archive with the
same rootless layout consumed by `modal_phase123.py --action prepare`:

    manifest.json
    <uid>/cameras.json
    <uid>/latent.npy
    <uid>/frame_000.png
    <uid>/masks/mask_000.png
    ...

Default usage for the full v7 handoff:

    modal run data/zip_dataset.py::full \
        --volume-path objaverse_v7 \
        --manifest data/manifests/objaverse_v7_combined.json

The command above creates `/objaverse_v7.zip` inside the source volume. It can
then be copied with:

    modal volume get latent2splat-renders objaverse_v7.zip ./
"""
from __future__ import annotations

import json
import os
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import modal


APP_NAME = "latent2splat-zip-dataset"
VOLUME_NAME = os.environ.get("L2S_RENDERS_VOLUME", "latent2splat-renders")
VOLUME_DIR = "/renders"
PHASE_DATA_VOLUME_NAME = os.environ.get("L2S_PHASE_DATA_VOLUME", "latent2splat-phase123-data")
PHASE_DATA_DIR = "/phase-data"

renders_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
phase_data_volume = modal.Volume.from_name(PHASE_DATA_VOLUME_NAME, create_if_missing=True)
utility_image = modal.Image.debian_slim(python_version="3.12")
inspect_image = utility_image.pip_install("numpy==1.26.4", "Pillow")
app = modal.App(APP_NAME, image=utility_image)


def _entry_uid(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("uid", "object_id", "id"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _split_entries(manifest: dict[str, Any],
                   split_names: set[str] | None,
                   limit_per_split: int) -> tuple[dict[str, Any], list[str]]:
    """Return a manifest subset plus the object UIDs referenced by it."""
    out: dict[str, Any] = {}
    uids: list[str] = []
    seen: set[str] = set()

    for split, entries in manifest.items():
        if not isinstance(entries, list):
            out[split] = entries
            continue
        if split == "failed":
            out[split] = entries
            continue
        if split_names is not None and split not in split_names:
            continue

        selected = entries[:limit_per_split] if limit_per_split > 0 else entries
        out[split] = selected
        for entry in selected:
            uid = _entry_uid(entry)
            if uid and uid not in seen:
                seen.add(uid)
                uids.append(uid)

    return out, uids


def _load_manifest(source_root: Path,
                   manifest_text: str,
                   manifest_hint: str) -> tuple[dict[str, Any], str]:
    if manifest_text:
        return json.loads(manifest_text), "local"

    hint = Path(manifest_hint).name if manifest_hint else "manifest.json"
    candidates = [
        source_root / "manifest.json",
        source_root / hint,
        source_root.parent / hint,
        source_root.parent / "data" / "manifests" / hint,
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text()), str(path)
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"manifest was not provided locally and was not found in the source volume. "
        f"Checked: {checked}"
    )


def _iter_files(root: Path, exclude_subdirs: set[str]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".zip"):
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in exclude_subdirs:
            continue
        if "__pycache__" in rel.parts:
            continue
        yield path


def _object_dirs(root: Path) -> list[Path]:
    return [
        path for path in sorted(root.iterdir())
        if path.is_dir() and (path / "latent.npy").exists() and (path / "cameras.json").exists()
    ]


def _generated_split_manifest(root: Path,
                              eval_count: int,
                              test_count: int,
                              split_seed: int) -> dict[str, list[str]]:
    dirs = _object_dirs(root)
    if not dirs:
        raise ValueError(f"no object directories found under {root}")
    uids = [p.name for p in dirs]
    random.Random(int(split_seed)).shuffle(uids)
    n = len(uids)
    if n < 3:
        return {"train": uids, "eval": [], "test": []}
    if n < 100:
        eval_n = max(1, n // 10)
        test_n = max(1, n // 10)
    else:
        eval_n = max(int(eval_count), 0)
        test_n = max(int(test_count), 0)
    eval_n = min(eval_n, max(n - 1, 0))
    test_n = min(test_n, max(n - eval_n - 1, 0))
    eval_uids = uids[:eval_n]
    test_uids = uids[eval_n:eval_n + test_n]
    train_uids = uids[eval_n + test_n:]
    return {
        "train": [{"uid": uid} for uid in train_uids],
        "eval": [{"uid": uid} for uid in eval_uids],
        "test": [{"uid": uid} for uid in test_uids],
    }


def _camera_summary(cameras: Any) -> dict[str, Any]:
    if not isinstance(cameras, dict):
        return {"schema": type(cameras).__name__}
    width = cameras.get("width")
    height = cameras.get("height")
    if width is None and isinstance(cameras.get("intrinsics"), dict):
        width = cameras["intrinsics"].get("width")
        height = cameras["intrinsics"].get("height")
    out: dict[str, Any] = {"width": width, "height": height}
    for key in ("frames", "orbit", "orbit_cameras", "canonical", "canonical_cameras", "cameras"):
        value = cameras.get(key)
        if isinstance(value, list):
            out[f"{key}_count"] = len(value)
    return out


@app.function(
    volumes={VOLUME_DIR: renders_volume},
    timeout=24 * 60 * 60,
    cpu=4,
    memory=16 * 1024,
)
def _zip_dataset(volume_path: str,
                 manifest_text: str,
                 manifest_hint: str,
                 out_name: str,
                 splits: str,
                 limit_per_split: int,
                 compression: str,
                 exclude_subdirs: str,
                 dry_run: int,
                 strict: int) -> dict[str, Any]:
    source_root = Path(VOLUME_DIR) / volume_path.strip("/")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source dataset directory missing: {source_root}")

    split_names = {s.strip() for s in splits.split(",") if s.strip()} if splits else None
    manifest, manifest_source = _load_manifest(source_root, manifest_text, manifest_hint)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object keyed by split")

    manifest_out, uids = _split_entries(manifest, split_names, limit_per_split)
    if not uids:
        raise ValueError("manifest did not reference any object UIDs")

    exclude = {s.strip() for s in exclude_subdirs.split(",") if s.strip()}
    missing: list[str] = []
    object_dirs: list[Path] = []
    file_count = 0
    byte_count = 0
    for uid in uids:
        obj_dir = source_root / uid
        if not obj_dir.is_dir():
            missing.append(uid)
            continue
        object_dirs.append(obj_dir)
        for path in _iter_files(obj_dir, exclude):
            file_count += 1
            byte_count += path.stat().st_size

    if missing and strict:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} manifest objects missing under {source_root}: {preview}")

    out_path = Path(VOLUME_DIR) / (out_name or f"{source_root.name}.zip")
    if out_path.suffix != ".zip":
        out_path = out_path.with_suffix(".zip")

    summary = {
        "source_root": str(source_root),
        "manifest_source": manifest_source,
        "out_path": str(out_path),
        "splits": {k: len(v) for k, v in manifest_out.items() if isinstance(v, list)},
        "objects": len(object_dirs),
        "missing_objects": len(missing),
        "files": file_count,
        "bytes_gb": round(byte_count / 1e9, 3),
        "compression": compression,
        "exclude_subdirs": sorted(exclude),
        "dry_run": bool(dry_run),
    }
    print("[zip_dataset] summary before write:", json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return summary

    compression_mode = zipfile.ZIP_STORED if compression == "stored" else zipfile.ZIP_DEFLATED
    compresslevel = None if compression == "stored" else 1
    started = time.time()
    written = 0
    with zipfile.ZipFile(out_path, "w", compression=compression_mode, compresslevel=compresslevel) as zf:
        zf.writestr("manifest.json", json.dumps(manifest_out, indent=2, sort_keys=True))
        zf.writestr(
            "dataset_zip_info.json",
            json.dumps({**summary, "created_at_unix": started}, indent=2, sort_keys=True),
        )
        for obj_dir in object_dirs:
            uid = obj_dir.name
            for path in _iter_files(obj_dir, exclude):
                zf.write(path, arcname=str(Path(uid) / path.relative_to(obj_dir)))
                written += 1
                if written % 2000 == 0:
                    print(f"[zip_dataset] wrote {written}/{file_count} files", flush=True)

    size_gb = out_path.stat().st_size / 1e9
    summary.update({
        "written_files": written,
        "zip_size_gb": round(size_gb, 3),
        "elapsed_sec": round(time.time() - started, 1),
    })
    renders_volume.commit()
    print("[zip_dataset] done:", json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(
    volumes={VOLUME_DIR: renders_volume, PHASE_DATA_DIR: phase_data_volume},
    timeout=12 * 60 * 60,
    cpu=4,
    memory=16 * 1024,
)
def _install_phase123(volume_path: str,
                      archive_name: str,
                      dataset: str,
                      manifest_name: str,
                      force: int,
                      generate_manifest: int,
                      eval_count: int,
                      test_count: int,
                      split_seed: int) -> dict[str, Any]:
    """Extract a source-volume dataset archive into the Phase-123 data volume."""
    import shutil

    def _extract_archive(zip_path: Path, dst_root: Path) -> None:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n and not n.endswith("/")]
            tops = {n.split("/", 1)[0] for n in names if "/" in n}
            nested = len(tops) == 1 and any(n.endswith(f"/{manifest_name}") for n in names)
            if nested:
                tmp = Path(PHASE_DATA_DIR) / f"_extract_{dataset}"
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True)
                zf.extractall(tmp)
                only = tmp / next(iter(tops))
                for p in only.iterdir():
                    shutil.move(str(p), dst_root / p.name)
                shutil.rmtree(tmp)
            else:
                zf.extractall(dst_root)

    archive = archive_name or f"{volume_path.strip('/').rstrip('/')}.zip"
    zip_path = Path(VOLUME_DIR) / archive
    if not zip_path.exists():
        raise FileNotFoundError(f"source archive missing: {zip_path}")
    root = Path(PHASE_DATA_DIR) / dataset
    if force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    if force or not (root / manifest_name).exists():
        print(f"[install_phase123] extracting {zip_path} -> {root}", flush=True)
        _extract_archive(zip_path, root)
    if not (root / manifest_name).exists():
        if not generate_manifest:
            raise FileNotFoundError(f"manifest missing after extraction: {root / manifest_name}")
        manifest = _generated_split_manifest(root, eval_count, test_count, split_seed)
        (root / manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    manifest = json.loads((root / manifest_name).read_text())
    objects = sum(len(v) for v in manifest.values() if isinstance(v, list))
    out = {
        "root": str(root),
        "archive": str(zip_path),
        "objects": objects,
        "splits": {k: len(v) for k, v in manifest.items() if isinstance(v, list)},
        "depth_files": len(list(root.glob("*/depth/depth_*.npy"))),
        "ltx_decoded_frames": len(list(root.glob("*/ltx_decoded/frame_*.png"))),
        "da3_depth_files": len(list(root.glob("*/da3_ltx/depth_*.npy"))),
    }
    phase_data_volume.commit()
    print("[install_phase123]", json.dumps(out, indent=2), flush=True)
    return out


@app.function(
    image=inspect_image,
    volumes={VOLUME_DIR: renders_volume},
    timeout=30 * 60,
    cpu=2,
    memory=6 * 1024,
)
def _inspect_source(volume_path: str,
                    manifest_name: str,
                    limit_per_split: int,
                    cond_subdir: str,
                    cond_depth_subdir: str,
                    warn_only: int) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    root = Path(VOLUME_DIR) / volume_path.strip("/")
    if not root.is_dir():
        raise FileNotFoundError(f"source dataset directory missing: {root}")

    manifest_path = root / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    split_counts = {k: len(v) for k, v in manifest.items() if isinstance(v, list)}

    sample_uids: list[str] = []
    seen: set[str] = set()
    for split, entries in manifest.items():
        if not isinstance(entries, list) or split == "failed":
            continue
        selected = entries[:limit_per_split] if limit_per_split > 0 else entries[:1]
        for entry in selected:
            uid = _entry_uid(entry)
            if uid and uid not in seen:
                seen.add(uid)
                sample_uids.append(uid)

    errors: list[str] = []
    samples: list[dict[str, Any]] = []
    for uid in sample_uids:
        obj = root / uid
        info: dict[str, Any] = {"uid": uid}
        if not obj.is_dir():
            errors.append(f"{uid}: object directory missing")
            samples.append(info)
            continue

        latent_path = obj / "latent.npy"
        cameras_path = obj / "cameras.json"
        frame_paths = sorted(obj.glob("frame_*.png"))
        mask_paths = sorted((obj / "masks").glob("mask_*.png"))
        depth_paths = sorted((obj / "depth").glob("depth_*.npy"))
        info.update({
            "frames": len(frame_paths),
            "masks": len(mask_paths),
            "depth": len(depth_paths),
            "ltx_decoded": len(list((obj / cond_subdir).glob("frame_*.png"))) if cond_subdir else 0,
            "cond_depth": len(list((obj / cond_depth_subdir).glob("depth_*.npy"))) if cond_depth_subdir else 0,
        })

        if latent_path.exists():
            latent = np.load(latent_path, mmap_mode="r")
            info["latent_shape"] = list(latent.shape)
            info["latent_dtype"] = str(latent.dtype)
        else:
            errors.append(f"{uid}: latent.npy missing")

        if cameras_path.exists():
            cameras = json.loads(cameras_path.read_text())
            info["cameras"] = _camera_summary(cameras)
        else:
            errors.append(f"{uid}: cameras.json missing")

        if frame_paths:
            with Image.open(frame_paths[0]) as image:
                info["frame0_size"] = list(image.size)
                info["frame0_mode"] = image.mode
        else:
            errors.append(f"{uid}: no frame_*.png files")
        if mask_paths:
            with Image.open(mask_paths[0]) as image:
                info["mask0_size"] = list(image.size)
                info["mask0_mode"] = image.mode
        else:
            errors.append(f"{uid}: no masks/mask_*.png files")
        samples.append(info)

    totals = {
        "object_dirs": len(_object_dirs(root)),
        "latents": len(list(root.glob("*/latent.npy"))),
        "cameras": len(list(root.glob("*/cameras.json"))),
        "frames": len(list(root.glob("*/frame_*.png"))),
        "masks": len(list(root.glob("*/masks/mask_*.png"))),
        "depth": len(list(root.glob("*/depth/depth_*.npy"))),
        "ltx_decoded_frames": len(list(root.glob(f"*/{cond_subdir}/frame_*.png"))) if cond_subdir else 0,
        "cond_depth_files": len(list(root.glob(f"*/{cond_depth_subdir}/depth_*.npy"))) if cond_depth_subdir else 0,
    }
    out = {
        "root": str(root),
        "manifest": str(manifest_path),
        "splits": split_counts,
        "totals": totals,
        "samples": samples,
        "errors": errors,
        "ok": not errors,
    }
    print("[inspect_source]", json.dumps(out, indent=2), flush=True)
    if errors and not warn_only:
        preview = "; ".join(errors[:10])
        raise RuntimeError(f"dataset inspection failed with {len(errors)} errors: {preview}")
    return out


@app.function(
    volumes={VOLUME_DIR: renders_volume},
    timeout=30 * 60,
    cpu=2,
    memory=4 * 1024,
)
def _write_generated_manifest(volume_path: str,
                              manifest_name: str,
                              eval_count: int,
                              test_count: int,
                              split_seed: int,
                              force: int) -> dict[str, Any]:
    root = Path(VOLUME_DIR) / volume_path.strip("/")
    if not root.is_dir():
        raise FileNotFoundError(f"source dataset directory missing: {root}")
    manifest_path = root / manifest_name
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = _generated_split_manifest(root, eval_count, test_count, split_seed)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        renders_volume.commit()
    out = {
        "root": str(root),
        "manifest": str(manifest_path),
        "splits": {k: len(v) for k, v in manifest.items() if isinstance(v, list)},
        "objects": sum(len(v) for v in manifest.values() if isinstance(v, list)),
    }
    print("[write_manifest]", json.dumps(out, indent=2), flush=True)
    return out


def _read_manifest_text(manifest: str) -> str:
    if not manifest:
        return ""
    path = Path(manifest)
    if not path.exists():
        print(f"[zip_dataset] local manifest not found, will look inside volume: {path}", flush=True)
        return ""
    return path.read_text()


@app.local_entrypoint()
def full(volume_path: str = "objaverse_v7",
         manifest: str = "data/manifests/objaverse_v7_combined.json",
         out_name: str = "",
         splits: str = "",
         limit_per_split: int = 0,
         compression: str = "stored",
         exclude_subdirs: str = "",
         dry_run: int = 0,
         strict: int = 1) -> None:
    """Create a full dataset archive in the source Modal Volume."""
    if compression not in {"stored", "deflated"}:
        raise ValueError("--compression must be stored or deflated")
    result = _zip_dataset.remote(
        volume_path=volume_path,
        manifest_text=_read_manifest_text(manifest),
        manifest_hint=manifest,
        out_name=out_name,
        splits=splits,
        limit_per_split=limit_per_split,
        compression=compression,
        exclude_subdirs=exclude_subdirs,
        dry_run=dry_run,
        strict=strict,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def smoke(volume_path: str = "objaverse_v7",
          manifest: str = "data/manifests/objaverse_v7_combined.json",
          out_name: str = "objaverse_v7_smoke.zip",
          limit_per_split: int = 8,
          compression: str = "stored",
          dry_run: int = 0) -> None:
    """Create a small train/eval/test subset archive for cheap harness checks."""
    result = _zip_dataset.remote(
        volume_path=volume_path,
        manifest_text=_read_manifest_text(manifest),
        manifest_hint=manifest,
        out_name=out_name,
        splits="train,eval,test,heldout",
        limit_per_split=limit_per_split,
        compression=compression,
        exclude_subdirs="",
        dry_run=dry_run,
        strict=1,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def install_phase123(volume_path: str = "objaverse_v7",
                     archive_name: str = "",
                     dataset: str = "objaverse_v7",
                     manifest_name: str = "manifest.json",
                     force: int = 0,
                     generate_manifest: int = 1,
                     eval_count: int = 10,
                     test_count: int = 10,
                     split_seed: int = 0) -> None:
    """Install a zipped source-volume dataset into the Phase-123 data volume."""
    result = _install_phase123.remote(
        volume_path=volume_path,
        archive_name=archive_name,
        dataset=dataset,
        manifest_name=manifest_name,
        force=force,
        generate_manifest=generate_manifest,
        eval_count=eval_count,
        test_count=test_count,
        split_seed=split_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def write_manifest(volume_path: str = "objaverse_v7",
                   manifest_name: str = "manifest.json",
                   eval_count: int = 10,
                   test_count: int = 10,
                   split_seed: int = 0,
                   force: int = 0) -> None:
    """Create a deterministic manifest in a source-volume dataset directory."""
    result = _write_generated_manifest.remote(
        volume_path=volume_path,
        manifest_name=manifest_name,
        eval_count=eval_count,
        test_count=test_count,
        split_seed=split_seed,
        force=force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def inspect_source(volume_path: str = "objaverse_v7",
                   manifest_name: str = "manifest.json",
                   limit_per_split: int = 2,
                   cond_subdir: str = "ltx_decoded",
                   cond_depth_subdir: str = "da3_ltx",
                   warn_only: int = 0) -> None:
    """Inspect a source-volume dataset without building decoder CUDA images."""
    result = _inspect_source.remote(
        volume_path=volume_path,
        manifest_name=manifest_name,
        limit_per_split=limit_per_split,
        cond_subdir=cond_subdir,
        cond_depth_subdir=cond_depth_subdir,
        warn_only=warn_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
