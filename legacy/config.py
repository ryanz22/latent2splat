"""Config-driven training: dataclasses + YAML loading for the latent->3DGS decoder.

Loss terms are weights; weight 0 disables the term (the loop skips it). Every
component (photometric L1/SSIM, silhouette BCE, depth, opacity reg, scale reg)
is toggled and reweighted purely from a YAML config — see configs/*.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataCfg:
    dataset_root: str
    manifest: str | None = None
    split: str = "train"
    uid: str = ""                  # "" = first entry in the split
    bg: float = 0.0                # dataset background gray level (animals = 0 black)
    bg_variant: str | None = None  # None (default frames) | black | white | gray (Trial R4+)
    random_bg: bool = False
    synthetic_latent: bool = False  # dry-run a config on a latent-less (format-preview) dataset


@dataclass
class ModelCfg:
    arch: str = "e"
    opacity_mode: str = "sigmoid"  # pdf (depth-PDF) | sigmoid (free opacity); arch e


@dataclass
class OptimCfg:
    lr: float = 1.0e-4
    steps: int = 1500
    warmup_steps: int = 100
    grad_clip: float = 1.0
    log_every: int = 50


@dataclass
class LossCfg:
    """Each term is a weight; 0 disables it (the loop skips zero-weight terms)."""
    l1: float = 1.0                  # photometric L1
    ssim: float = 0.2                # (1 - SSIM)
    fg_weight: float = 10.0          # foreground upweight inside the silhouette
    mask_bce: float = 0.0            # alpha-vs-silhouette BCE (needs a mask)
    depth: float = 0.0               # depth Huber (needs depth data)
    opacity_reg: float = 0.0         # opacity regularization
    scale_reg: float = 0.0           # mean-scale regularization
    fg_source: str = "gt"            # gt (dataset masks) | threshold (bg-color heuristic)
    opacity_reg_masked: bool = True  # penalize ONLY background-anchored Gaussians
    depth_delta: float = 0.1         # Huber delta for the depth loss


@dataclass
class TrainCfg:
    data: DataCfg
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    out_dir: str = "runs"
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainCfg":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            data=DataCfg(**raw["data"]),
            model=ModelCfg(**raw.get("model", {})),
            optim=OptimCfg(**raw.get("optim", {})),
            loss=LossCfg(**raw.get("loss", {})),
            out_dir=raw.get("out_dir", "runs"),
            device=raw.get("device", "cuda"),
        )

    def apply_overrides(self, overrides: list[str]) -> "TrainCfg":
        """Apply 'section.key=value' CLI overrides, e.g. 'loss.depth=1.0'."""
        for ov in overrides:
            key, _, val = ov.partition("=")
            if "." in key:                       # section.field, e.g. loss.depth
                section, _, name = key.partition(".")
                obj = getattr(self, section)
            else:                                # top-level field, e.g. out_dir
                obj, name = self, key
            cur = getattr(obj, name)
            if isinstance(cur, bool):
                newv: object = val.lower() in ("1", "true", "yes")
            elif isinstance(cur, int) and not isinstance(cur, bool):
                newv = int(val)
            elif isinstance(cur, float):
                newv = float(val)
            elif cur is None:
                newv = None if val.lower() in ("none", "null") else val
            else:
                newv = val
            setattr(obj, name, newv)
        return self
