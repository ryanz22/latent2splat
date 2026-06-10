"""Checkpoint visualization: target | render | alpha grid from a saved run."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _viz(ckpt_path: str, out_dir: str, views=(0, 12, 24, 36)) -> None:
    ck = torch.load(ckpt_path, map_location="cpu")
    render, target = ck["render"], ck["target"]
    alpha = ck.get("alpha")
    res = ck["result"]
    print(f"[viz] {res.get('uid')} last_l1={res['last_l1']:.4f} "
          f"final_psnr={res['final_psnr']:.2f} V={render.shape[0]}")
    views = [v for v in views if v < render.shape[0]]

    def im(t):
        return (t.clamp(0, 1).numpy() * 255).astype(np.uint8)

    rows = []
    for v in views:
        cols = [im(target[v]), im(render[v])]
        if alpha is not None:
            a = alpha[v, ..., 0].clamp(0, 1).numpy()
            cols.append((np.stack([a, a, a], -1) * 255).astype(np.uint8))
        rows.append(np.concatenate(cols, 1))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = Path(ckpt_path).stem
    p = Path(out_dir) / f"viz_{name}.png"
    Image.fromarray(np.concatenate(rows, 0)).save(p)
    print(f"[viz] wrote {p}")
