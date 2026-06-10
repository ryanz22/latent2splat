import torch

from decoder.clean.train_depth_anchor import DepthAnchorNet


def test_depth_anchor_residual_da3_zero_init_preserves_da3():
    model = DepthAnchorNet(
        decoder_ch=8,
        pretrained=False,
        use_da3_input=True,
        residual_da3=True,
    ).eval()
    rgb = torch.rand(1, 3, 64, 64)
    da3_log = torch.rand(1, 1, 64, 64)
    da3_valid = torch.ones(1, 1, 64, 64)

    with torch.no_grad():
        pred = model(rgb, da3_log, da3_valid)

    assert torch.allclose(pred, da3_log, atol=1e-6)
