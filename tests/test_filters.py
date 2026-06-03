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


def test_dpid_preserves_edge_differently_from_box():
    # 2x4 image: left block (cols 0-1) has one bright outlier pixel;
    # right block is uniform 0.5. We check the left output pixel only.
    # Box average of left block: (0+0+0+1)/4 = 0.25
    # DPID up-weights the outlier (it's farthest from local mean), so output > 0.25
    img = np.zeros((2, 4, 1), dtype=np.float32)
    img[0, 1, 0] = 1.0   # single bright pixel in left 2x2 block

    box = box_downsample(img, factor=2)
    dpid = dpid_downsample(img, lam=1.0, support=2, factor=2)

    assert box.shape == dpid.shape == (1, 2, 1)
    np.testing.assert_allclose(box[0, 0, 0], 0.25, atol=1e-6)
    assert np.isfinite(dpid).all()
    assert 0.0 <= float(dpid[0, 0, 0]) <= 1.0
    # DPID must up-weight the outlier pixel -> result strictly greater than box average
    assert float(dpid[0, 0, 0]) > float(box[0, 0, 0]), (
        f"DPID ({dpid[0,0,0]:.6f}) should exceed box ({box[0,0,0]:.6f}) "
        "when a single outlier pixel is present"
    )
