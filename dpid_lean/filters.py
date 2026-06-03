"""数组级下采样算子：box 平均与 DPID 细节保留下采样。

提取自 pbr_compress/filters.py，仅依赖 numpy / scipy。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def box_downsample(x: np.ndarray, factor: int = 2) -> np.ndarray:
    """逐 factor×factor 块求平均下采样。

    x: (H, W, C)，H/W 必须能被 factor 整除。返回 (H/factor, W/factor, C)；整数输入会被 mean 提升为 float。
    """
    if x.ndim != 3:
        raise ValueError(f"box_downsample: 期望 3D 输入 (H, W, C)，实际 ndim={x.ndim}")
    if not isinstance(factor, int) or factor < 2:
        raise ValueError(f"box_downsample: factor 必须是 >=2 的整数，实际 {factor}")
    H, W, C = x.shape
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"box_downsample: H、W 必须能被 factor={factor} 整除，实际 H={H}, W={W}"
        )
    f = factor
    return x.reshape(H // f, f, W // f, f, C).mean(axis=(1, 3))


def dpid_downsample(
    img: np.ndarray,
    lam: float = 1.0,
    support: int = 4,
    factor: int = 2,
) -> np.ndarray:
    """DPID（Detail-Preserving Image Downscaling）factor× 下采样。

    与局部均值距离越远的像素权重越大，保留细节。
    lam > 0 上权重给细节；lam = 0 退化为均匀权重（等价 box）；负值未定义。
    img: (H, W, C) float，H/W 必须能被 factor 整除。返回 (H/factor, W/factor, C)。
    """
    if img.ndim != 3:
        raise ValueError("dpid_downsample: 期望 3D 输入 (H, W, C)")
    if not isinstance(factor, int) or factor < 2:
        raise ValueError(f"dpid_downsample: factor 必须是 >=2 的整数，实际 {factor}")
    H, W, C = img.shape
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"dpid_downsample: H、W 必须能被 factor={factor} 整除，实际 H={H}, W={W}"
        )

    mu = np.stack(
        [uniform_filter(img[..., c], size=support, mode="reflect") for c in range(C)],
        axis=-1,
    )
    dist = np.linalg.norm(img - mu, axis=-1, keepdims=True)
    weights = (dist ** lam) + 1e-8                       # (H, W, 1)
    weighted = img * weights
    num = box_downsample(weighted, factor=factor)
    den = box_downsample(weights, factor=factor)
    return num / np.maximum(den, 1e-12)
