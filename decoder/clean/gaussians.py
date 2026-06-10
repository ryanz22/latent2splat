"""Raw head channels → activated 3D Gaussian params, with the proven
parameterization (GS-LRM bounded scale, 3DGS unit-quat, Splatter-Image
sigmoid-depth, low opacity/scale init biases). Plus opacity pruning."""
from __future__ import annotations

import torch
import torch.nn.functional as F

B_ALPHA = -4.5          # opacity init bias → sigmoid(-4.5)≈0.011. Far below the
                        # design's -2 (α≈0.12): with 98k overlapping ray-anchored
                        # Gaussians, α=0.12 saturates accumulated alpha to ~1.0
                        # EVERYWHERE at init, so mask-L1 collapses opacity clawing
                        # the background back down. ~0.011 keeps init alpha unsaturated.
B_SCALE = -2.3          # scale init bias → small splats at start
SCALE_CAP_FRAC = 0.05   # s_max = 0.05 * radius
# raw layout (12): [rgb(0:3), scale(3:6), quat(6:10), opacity(10:11), dist(11:12)]
# optional raw layout (15): previous 12 + bounded xyz mean offset (12:15)


def soft_cap_scale(raw_scale: torch.Tensor, cap: float | torch.Tensor) -> torch.Tensor:
    """Differentiably bound positive scales below `cap`.

    The previous hard clamp made scale gradients exactly zero once splats hit the
    cap. The smooth rational cap keeps the same upper bound but still lets the
    network shrink or grow capped-looking splats.
    """
    cap_t = torch.as_tensor(cap, dtype=raw_scale.dtype, device=raw_scale.device)
    return cap_t * raw_scale / (raw_scale + cap_t.clamp_min(1e-8))


def activate(raw: torch.Tensor, origins: torch.Tensor, dirs: torch.Tensor,
             d_near: float, d_far: float, radius: float,
             scale_cap_frac: float = SCALE_CAP_FRAC,
             mean_offset_frac: float = 0.0) -> dict:
    """raw (N,12|15), origins (N,3), dirs (N,3 unit) → Gaussian param dict."""
    rgb = torch.sigmoid(raw[:, 0:3])
    scale_raw = torch.exp(raw[:, 3:6] + B_SCALE)
    scale = soft_cap_scale(scale_raw, scale_cap_frac * radius)
    # bias toward identity so a zero raw quaternion → valid identity rotation
    # (avoids the degenerate zero-quat → singular-covariance NaN, failure mode D1)
    quat = F.normalize(raw[:, 6:10] + raw.new_tensor([1.0, 0.0, 0.0, 0.0]), dim=-1)
    opacity = torch.sigmoid(raw[:, 10:11] + B_ALPHA)
    t = d_near + (d_far - d_near) * torch.sigmoid(raw[:, 11:12])   # (N,1)
    mean_anchor = origins + t * dirs
    if raw.shape[-1] >= 15 and mean_offset_frac > 0:
        mean_offset = torch.tanh(raw[:, 12:15]) * (mean_offset_frac * radius)
    else:
        mean_offset = torch.zeros_like(mean_anchor)
    mean = mean_anchor + mean_offset
    return {"mean": mean, "quat": quat, "scale": scale, "opacity": opacity,
            "rgb": rgb, "depth": t, "scale_raw": scale_raw,
            "mean_anchor": mean_anchor, "mean_offset": mean_offset}


def prune_by_opacity(params: dict, thresh: float = 0.005) -> dict:
    """Drop Gaussians with opacity < thresh (Splatter-Image/Lyra culling)."""
    keep = (params["opacity"].reshape(-1) >= thresh)
    return {k: v[keep] for k, v in params.items()}
