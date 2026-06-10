"""Config-driven training entrypoint (the modular pipeline path).

    python legacy/run_train.py legacy/configs/overfit_animalsv1.yaml
    python legacy/run_train.py legacy/configs/overfit_trial_r4.yaml --set loss.depth=0.5 optim.steps=800

Loads a YAML TrainCfg (decoder/config.py), runs decoder.train.train_from_config,
and renders the target|render|alpha grid. Loss terms toggle/reweight purely via
the YAML or `--set section.key=value` overrides.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from legacy.config import TrainCfg
from legacy.train import train_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a YAML TrainCfg (see configs/)")
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v",
                    help="overrides, e.g. loss.depth=1.0 optim.steps=800 data.bg_variant=white")
    args = ap.parse_args()

    cfg = TrainCfg.from_yaml(args.config).apply_overrides(args.set)
    r = train_from_config(cfg)
    print(f"[train] DONE uid={r['uid']} first_l1={r['first_l1']:.4f} last_l1={r['last_l1']:.4f} "
          f"drop={r['l1_drop_ratio']:.1f}x final_psnr={r['final_psnr']:.2f}")
    if cfg.out_dir:
        from legacy.run_local import _viz
        _viz(str(Path(cfg.out_dir) / f"overfit_{cfg.model.arch}.pt"), cfg.out_dir)


if __name__ == "__main__":
    main()
