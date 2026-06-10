"""Build a combined v7+v8 manifest from Modal volume contents.

The combined dataset is split across volumes:
  /data/objaverse_v7/<uid>/... including latent.npy
  /data/objaverse_v8/<uid>/... frames/depth/cameras
  /latents_v8/objaverse_v8/<uid>/latent.npy

This script preserves v7's existing splits when /data/objaverse_v7/manifest.json
is present and creates deterministic v8 train/eval/test splits from the objects
that have both rendered data and a latent.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _entry_uid(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        uid = entry.get("uid") or entry.get("object_id") or entry.get("id")
        if isinstance(uid, str) and uid:
            return uid
    raise KeyError(f"manifest entry does not contain a uid: {entry!r}")


def _metadata_category(obj_dir: Path, fallback: str = "") -> str:
    p = obj_dir / "metadata.json"
    if not p.exists():
        return fallback
    try:
        data = json.loads(p.read_text())
    except Exception:
        return fallback
    cat = data.get("category") or data.get("label") or fallback
    return str(cat) if cat is not None else ""


def _valid_object(obj_dir: Path, latent_path: Path) -> bool:
    return (
        obj_dir.is_dir()
        and latent_path.exists()
        and (obj_dir / "cameras.json").exists()
        and (obj_dir / "frame_000.png").exists()
        and (obj_dir / "masks" / "mask_000.png").exists()
        and (obj_dir / "depth" / "depth_000.npy").exists()
    )


def _entry(uid: str, tier: str, obj_dir: Path, category: str = "",
           read_metadata: bool = False) -> dict[str, str]:
    cat = _metadata_category(obj_dir, fallback=category) if read_metadata else category
    out = {"uid": uid, "tier": tier}
    if cat:
        out["category"] = cat
    return out


def _load_v7(root: Path, read_metadata: bool) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    tier = "objaverse_v7"
    tier_root = root / tier
    manifest_path = tier_root / "manifest.json"
    out = {"train": [], "eval": [], "test": []}
    stats = {"manifest_entries": 0, "kept": 0, "missing": 0}

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for split in out:
            for raw in manifest.get(split, []):
                stats["manifest_entries"] += 1
                try:
                    uid = _entry_uid(raw)
                except KeyError:
                    stats["missing"] += 1
                    continue
                obj_dir = tier_root / uid
                latent = obj_dir / "latent.npy"
                if not _valid_object(obj_dir, latent):
                    stats["missing"] += 1
                    continue
                category = raw.get("category", "") if isinstance(raw, dict) else ""
                out[split].append(_entry(uid, tier, obj_dir, category=category,
                                         read_metadata=read_metadata))
                stats["kept"] += 1
        return out, stats

    for obj_dir in sorted(p for p in tier_root.iterdir() if p.is_dir()):
        latent = obj_dir / "latent.npy"
        if _valid_object(obj_dir, latent):
            out["train"].append(_entry(obj_dir.name, tier, obj_dir, read_metadata=read_metadata))
            stats["kept"] += 1
        else:
            stats["missing"] += 1
    stats["manifest_entries"] = stats["kept"] + stats["missing"]
    return out, stats


def _load_v8(root: Path, latents_root: Path, seed: int,
             eval_count: int, test_count: int,
             read_metadata: bool) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    tier = "objaverse_v8"
    tier_root = root / tier
    latent_root = latents_root / tier
    candidates: list[dict[str, str]] = []
    stats = {"dirs": 0, "kept": 0, "missing": 0}
    for obj_dir in sorted(p for p in tier_root.iterdir() if p.is_dir()):
        stats["dirs"] += 1
        latent = latent_root / obj_dir.name / "latent.npy"
        if not _valid_object(obj_dir, latent):
            stats["missing"] += 1
            continue
        candidates.append(_entry(obj_dir.name, tier, obj_dir, read_metadata=read_metadata))
        stats["kept"] += 1
        if stats["dirs"] % 500 == 0:
            print(
                f"[combined_manifest] v8 scanned={stats['dirs']} kept={stats['kept']} "
                f"missing={stats['missing']}",
                flush=True,
            )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    n = len(candidates)
    n_test = min(max(test_count, 0), n)
    n_eval = min(max(eval_count, 0), max(n - n_test, 0))
    test = candidates[:n_test]
    eval_ = candidates[n_test:n_test + n_eval]
    train = candidates[n_test + n_eval:]
    for split_entries in (train, eval_, test):
        split_entries.sort(key=lambda e: e["uid"])
    return {"train": train, "eval": eval_, "test": test}, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="/data")
    ap.add_argument("--latents_v8_root", default="/latents_v8")
    ap.add_argument("--out", default="/data/manifests/combined_v7_v8.json")
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--v8_eval_count", type=int, default=128)
    ap.add_argument("--v8_test_count", type=int, default=128)
    ap.add_argument("--read_metadata", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    latents_root = Path(args.latents_v8_root)
    read_metadata = bool(args.read_metadata)
    v7, v7_stats = _load_v7(root, read_metadata=read_metadata)
    v8, v8_stats = _load_v8(
        root,
        latents_root,
        args.seed,
        args.v8_eval_count,
        args.v8_test_count,
        read_metadata=read_metadata,
    )
    combined = {
        "train": [*v7["train"], *v8["train"]],
        "eval": [*v7["eval"], *v8["eval"]],
        "test": [*v7["test"], *v8["test"]],
    }
    summary = {
        "v7": v7_stats,
        "v8": v8_stats,
        "splits": {k: len(v) for k, v in combined.items()},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), **summary}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
