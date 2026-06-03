import numpy as np
import pytest
import imageio.v3 as iio

from dpid_lean.io_dataset import (
    srgb_to_linear,
    linear_to_srgb,
    detect_layout,
    load_asset,
    save_asset,
)


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 50, dtype=np.float32).reshape(5, 10, 1)
    back = linear_to_srgb(srgb_to_linear(x))
    np.testing.assert_allclose(back, x, atol=1e-4)


def test_detect_layout_character_suffix():
    assert detect_layout("bottom_p25_d", "bottom_p25_n") == "character"


def test_detect_layout_packed_token():
    assert detect_layout("t_freezer_d_mb", "t_freezer_n_mb") == "packed"


def _write_uint8_rgba(path, arr01):
    iio.imwrite(str(path), (np.clip(arr01, 0, 1) * 255 + 0.5).astype(np.uint8))


def test_load_save_roundtrip_character(tmp_path):
    # character: _d = RGB albedo (no AO), _n = R=metal,G=nx,B=rough,A=ny
    tex = tmp_path / "Texture"
    tex.mkdir()
    d = np.zeros((4, 4, 3), dtype=np.float32)
    d[..., 0] = 0.5  # albedo R
    _write_uint8_rgba(tex / "asset_d.png", d)
    n = np.zeros((4, 4, 4), dtype=np.float32)
    n[..., 0] = 1.0   # metal = 1
    n[..., 1] = 0.5   # nx encoded 0.5 -> 0
    n[..., 2] = 0.3   # roughness
    n[..., 3] = 0.5   # ny encoded 0.5 -> 0
    _write_uint8_rgba(tex / "asset_n.png", n)

    (pbr, layout, d_stem, n_stem) = load_asset(tmp_path)
    A, N, R, M, AO = pbr
    assert layout == "character"
    assert d_stem == "asset_d" and n_stem == "asset_n"
    assert A.shape == (4, 4, 3) and N.shape == (4, 4, 3)
    assert R.shape == (4, 4, 1) and M.shape == (4, 4, 1) and AO.shape == (4, 4, 1)
    np.testing.assert_allclose(M, 1.0, atol=1e-2)          # metal preserved
    np.testing.assert_allclose(AO, 1.0, atol=1e-6)         # character AO defaults 1
    np.testing.assert_allclose(N[..., 2], 1.0, atol=1e-2)  # nz reconstructed (nx=ny=0)

    out_root = tmp_path / "out"
    save_asset(out_root / "asset", pbr, layout, d_stem, n_stem)
    assert (out_root / "asset" / "asset_d.png").exists()
    assert (out_root / "asset" / "asset_n.png").exists()
    # reload written n, channel order must match character layout (R=metal)
    raw = iio.imread(str(out_root / "asset" / "asset_n.png")).astype(np.float32) / 255.0
    np.testing.assert_allclose(raw[..., 0], 1.0, atol=1e-2)  # metal back in R


def test_load_packed(tmp_path):
    # packed: _d = RGB albedo + A=AO, _n = R=nx,G=ny,B=rough,A=metal
    tex = tmp_path / "Texture"
    tex.mkdir()
    d = np.zeros((4, 4, 4), dtype=np.float32)
    d[..., 0] = 0.5   # albedo R
    d[..., 3] = 0.7   # AO
    _write_uint8_rgba(tex / "t_x_d_mb.png", d)
    n = np.zeros((4, 4, 4), dtype=np.float32)
    n[..., 0] = 0.5   # nx encoded -> 0
    n[..., 1] = 0.5   # ny encoded -> 0
    n[..., 2] = 0.4   # roughness
    n[..., 3] = 1.0   # metal
    _write_uint8_rgba(tex / "t_x_n_mb.png", n)

    (pbr, layout, d_stem, n_stem) = load_asset(tmp_path)
    A, N, R, M, AO = pbr
    assert layout == "packed"
    np.testing.assert_allclose(AO, 0.7, atol=1e-2)         # AO from _d alpha
    np.testing.assert_allclose(M, 1.0, atol=1e-2)          # metal from _n alpha
    np.testing.assert_allclose(N[..., 2], 1.0, atol=1e-2)  # nz reconstructed


def test_load_asset_missing_file_raises(tmp_path):
    (tmp_path / "Texture").mkdir()
    with pytest.raises(FileNotFoundError):
        load_asset(tmp_path)
