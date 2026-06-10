from decoder.clean.decode_ltx_latents import default_decode_view_indices


def test_ltx_decode_view_indices_match_inclusive_encoder_spacing():
    assert default_decode_view_indices(55, 9, 49) == [0, 6, 12, 18, 24, 30, 36, 42, 48]
    assert default_decode_view_indices(49, 9, 49) == [0, 6, 12, 18, 24, 30, 36, 42, 48]


def test_ltx_decode_view_indices_handles_single_frame():
    assert default_decode_view_indices(49, 1, 49) == [0]
