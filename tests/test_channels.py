import numpy as np
import pytest

from dpid_lean.channels import (
    downsample_normal,
    downsample_roughness_lean,
    downsample_metallic,
    downsample_ao,
)


def test_downsample_normal_unit_length_and_shape():
    N = np.zeros((4, 4, 3), dtype=np.float32)
    N[..., 2] = 1.0  # all +Z
    out = downsample_normal(N, factor=2)
    assert out.shape == (2, 2, 3)
    lengths = np.linalg.norm(out, axis=-1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)
    np.testing.assert_allclose(out[..., 2], 1.0, atol=1e-6)


def test_downsample_normal_rejects_wrong_channels():
    with pytest.raises(ValueError):
        downsample_normal(np.zeros((4, 4, 1), dtype=np.float32), factor=2)


def test_roughness_lean_flat_normal_equals_box_of_alpha():
    # flat +Z normal -> sigma^2 = 0 -> alpha2_lr = box((R^2)^2); R_lr = sqrt(sqrt(...))
    R = np.full((4, 4, 1), 0.5, dtype=np.float32)
    N = np.zeros((4, 4, 3), dtype=np.float32)
    N[..., 2] = 1.0
    out = downsample_roughness_lean(R, N, factor=2)
    assert out.shape == (2, 2, 1)
    # alpha = 0.25, alpha^2 = 0.0625, box=0.0625, R_lr=sqrt(sqrt(0.0625))=0.5
    np.testing.assert_allclose(out, 0.5, atol=1e-6)


def test_roughness_lean_variance_increases_roughness():
    # divergent normals in a block -> |N_avg|<1 -> sigma^2>0 -> R_lr > box roughness
    R = np.full((2, 2, 1), 0.1, dtype=np.float32)
    N = np.array(
        [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
         [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
        dtype=np.float32,
    )  # averages to ~0 vector -> max variance
    out = downsample_roughness_lean(R, N, factor=2)
    assert out.shape == (1, 1, 1)
    assert float(out[0, 0, 0]) > 0.1  # roughened by absorbed variance


def test_metallic_threshold_binarizes():
    M = np.array([[[0.0], [1.0], [0.0], [1.0]],
                  [[0.0], [1.0], [0.0], [1.0]]], dtype=np.float32)  # (2,4,1)
    out = downsample_metallic(M, threshold=0.5, factor=2)
    assert out.shape == (1, 2, 1)
    # block0 mean=0.5 -> >=0.5 -> 1 ; block1 mean=1.0 -> 1
    np.testing.assert_array_equal(out[..., 0], [[1.0, 1.0]])


def test_ao_box_average():
    AO = np.full((4, 4, 1), 0.8, dtype=np.float32)
    out = downsample_ao(AO, factor=2)
    assert out.shape == (2, 2, 1)
    np.testing.assert_allclose(out, 0.8, atol=1e-6)


def test_roughness_lean_rejects_spatial_mismatch():
    R = np.full((4, 4, 1), 0.5, dtype=np.float32)
    N = np.zeros((2, 2, 3), dtype=np.float32)
    N[..., 2] = 1.0
    with pytest.raises(ValueError):
        downsample_roughness_lean(R, N, factor=2)


def test_metallic_and_ao_reject_wrong_channels():
    with pytest.raises(ValueError):
        downsample_metallic(np.zeros((4, 4, 3), dtype=np.float32), factor=2)
    with pytest.raises(ValueError):
        downsample_ao(np.zeros((4, 4, 3), dtype=np.float32), factor=2)
