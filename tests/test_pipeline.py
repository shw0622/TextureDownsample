import numpy as np
import pytest

from dpid_lean.pipeline import downsample_asset


def _make_pbr(h=4, w=4):
    A = np.full((h, w, 3), 0.5, dtype=np.float32)
    N = np.zeros((h, w, 3), dtype=np.float32)
    N[..., 2] = 1.0
    R = np.full((h, w, 1), 0.5, dtype=np.float32)
    M = np.zeros((h, w, 1), dtype=np.float32)
    AO = np.full((h, w, 1), 0.9, dtype=np.float32)
    return (A, N, R, M, AO)


def test_downsample_asset_shapes_factor2():
    A, N, R, M, AO = downsample_asset(_make_pbr(8, 8), factor=2)
    assert A.shape == (4, 4, 3)
    assert N.shape == (4, 4, 3)
    assert R.shape == (4, 4, 1)
    assert M.shape == (4, 4, 1)
    assert AO.shape == (4, 4, 1)
    # normals stay unit length
    np.testing.assert_allclose(np.linalg.norm(N, axis=-1), 1.0, atol=1e-6)


def test_downsample_asset_metal_threshold_default_binarizes():
    pbr = _make_pbr(4, 4)
    A, N, R, M, AO = pbr
    M[:] = 0.4  # below 0.5 everywhere
    M[0, 0, 0] = 1.0  # one hot pixel; block mean (4 px) = (1+0.4*3)/4 = 0.55 >= 0.5
    out = downsample_asset((A, N, R, M, AO), factor=2, metal_mode="threshold")
    assert set(np.unique(out[3])).issubset({0.0, 1.0})  # binarized


def test_downsample_asset_metal_avg_keeps_soft():
    pbr = _make_pbr(4, 4)
    A, N, R, M, AO = pbr
    M[:] = 0.4
    out = downsample_asset((A, N, R, M, AO), factor=2, metal_mode="avg")
    np.testing.assert_allclose(out[3], 0.4, atol=1e-6)  # not binarized


def test_downsample_asset_rejects_bad_metal_mode():
    with pytest.raises(ValueError):
        downsample_asset(_make_pbr(4, 4), factor=2, metal_mode="nope")
