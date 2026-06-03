import numpy as np
import pytest

from dpid_lean.filters import box_downsample, dpid_downsample


def test_box_downsample_2x_averages_blocks():
    x = np.array(
        [[[0.0], [2.0], [4.0], [6.0]],
         [[0.0], [2.0], [4.0], [6.0]]],
        dtype=np.float32,
    )  # shape (2, 4, 1)
    out = box_downsample(x, factor=2)
    assert out.shape == (1, 2, 1)
    # block 0: mean(0,2,0,2)=1 ; block 1: mean(4,6,4,6)=5
    np.testing.assert_allclose(out[..., 0], [[1.0, 5.0]], rtol=1e-6)


def test_box_downsample_rejects_indivisible():
    x = np.zeros((3, 4, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        box_downsample(x, factor=2)


def test_box_downsample_rejects_non_3d():
    with pytest.raises(ValueError):
        box_downsample(np.zeros((4, 4), dtype=np.float32), factor=2)


def test_dpid_downsample_shape_and_constant_block():
    # constant image -> dpid equals box (weights uniform); output equals input mean
    img = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = dpid_downsample(img, lam=1.0, support=4, factor=2)
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out, 0.5, atol=1e-6)


def test_dpid_downsample_rejects_indivisible():
    with pytest.raises(ValueError):
        dpid_downsample(np.zeros((3, 4, 3), dtype=np.float32), factor=2)
