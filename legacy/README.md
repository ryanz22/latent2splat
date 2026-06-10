# legacy/ — original strict-decoder lineage

This folder preserves the project's first phase, referenced in the paper and
poster as the strict (direct) latent→3DGS decoding approach: map the frozen
LTX VAE latent straight to a fixed budget of ray-anchored Gaussians, with no
RGBD evidence and no fusion. It is kept verbatim for the record and is not
maintained.

| File | What it is |
| --- | --- |
| `decoder_a.py` | Method A — token-aligned decoder (one Gaussian set per latent token) |
| `decoder_b.py` | Method B — learned-query decoder |
| `decoder_c.py` | Method C — pixel-aligned predicted-depth decoder (K per token); origin of the shared `ray_dirs_world` unprojection, whose maintained copy now lives in `decoder/clean/geometry.py` |
| `decoder_d.py` | Method D — C + GS-LRM activation shifts |
| `decoder_e.py` | Method E — transformer + deconv upsampler + depth-PDF opacity + scale floor |
| `gaussian_head.py` | shared raw-channel activations for A–C |
| `train.py` | single-object overfit + free-Gaussian control loops |
| `config.py`, `configs/` | YAML config for the config-driven trainer |
| `run_local.py`, `run_train.py` | local CLIs for the overfit/free-fit/viz runs |
| `tests/` | shape/range/unprojection tests for these decoders |

All of these collapse or fog in the multi-object setting — that negative
result motivated the rest of the project. The paper's main strict baseline,
`CleanGSDecoder` (transformer + upsampling to a dense pixel-aligned Gaussian
map), is part of the maintained tree at `decoder/clean/network.py`, trained
via `decoder/clean/train_phase2.py` / `train_clean.py`; the scaffolded RGBD
decoder target it is compared against lives in `decoder/clean/fusion.py`.

Run inside the compose container (see the root README):

```bash
# overfit one object with Method E
python legacy/run_local.py overfit --arch e --steps 400 \
  --dataset <dataset_dir> --manifest <manifest.json> --out runs/

# free-Gaussian control (no network)
python legacy/run_local.py freegs --steps 1000 --dataset ... --manifest ... --out runs/

# config-driven variant
python legacy/run_train.py legacy/configs/overfit_animalsv1.yaml

# tests
python -m pytest legacy/tests -q
```
