"""Opacity/scale regularizers (3DGS-MCMC adapted) + binarization entropy. CPU."""
import torch
from decoder.clean.losses import opacity_scale_reg, opacity_entropy


def test_unmasked_reg_is_mean_magnitude():
    op = torch.full((100, 1), 0.3)
    sc = torch.full((100, 3), 0.02)
    op_term, sc_term = opacity_scale_reg(op, sc, fg=None, masked=False)
    assert torch.allclose(op_term, torch.tensor(0.3))
    assert torch.allclose(sc_term, torch.tensor(0.02))


def test_masked_reg_ignores_foreground():
    # 2 FG gaussians (fg=1) with high opacity, 2 BG (fg=0) with high opacity.
    op = torch.tensor([[0.9], [0.9], [0.9], [0.9]])
    sc = torch.zeros(4, 3)
    fg = torch.tensor([1.0, 1.0, 0.0, 0.0])
    op_term, _ = opacity_scale_reg(op, sc, fg=fg, masked=True)
    # only the 2 BG gaussians are penalized -> mean over BG = 0.9, FG excluded
    assert torch.allclose(op_term, torch.tensor(0.9))
    # all-foreground -> no penalty
    op_term_fg, _ = opacity_scale_reg(op, sc, fg=torch.ones(4), masked=True)
    assert op_term_fg.item() == 0.0


def test_entropy_zero_at_extremes_max_at_half():
    assert opacity_entropy(torch.zeros(10, 1)).item() < 1e-5
    assert opacity_entropy(torch.ones(10, 1)).item() < 1e-5
    assert opacity_entropy(torch.full((10, 1), 0.5)).item() > 0.2
