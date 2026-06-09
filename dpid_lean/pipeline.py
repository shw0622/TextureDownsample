"""顶层管线：dpid albedo + LEAN roughness + normal/metal/ao 下采样。"""
from __future__ import annotations

import numpy as np

from dpid_lean.filters import box_downsample, dpid_downsample
from dpid_lean.channels import (
    downsample_normal,
    downsample_roughness_lean,
    downsample_metallic,
    downsample_ao,
)
from dpid_lean.io_dataset import PBRSet

_DPID_LAM = 1.0
_DPID_SUPPORT = 4


def downsample_asset(
    pbr: PBRSet,
    factor: int = 2,
    metal_mode: str = "threshold",
    lam: float = _DPID_LAM,
    support: int = _DPID_SUPPORT,
) -> PBRSet:
    """对一组高分 PBR 通道做 factor× 下采样，返回低分 (A,N,R,M,AO)。

    albedo: DPID(lam, support)
    normal: box + 重归一化
    roughness: LEAN 方差补偿（依赖高分 N）
    metallic: threshold(二值, 阈值固定 0.5, 默认) 或 avg(box 平均)
    ao: box 平均
    """
    if metal_mode not in ("threshold", "avg"):
        raise ValueError(f"metal_mode 必须是 'threshold' 或 'avg'，实际 {metal_mode!r}")

    A_hr, N_hr, R_hr, M_hr, AO_hr = pbr

    A_lr = dpid_downsample(A_hr, lam=lam, support=support, factor=factor)
    N_lr = downsample_normal(N_hr, factor=factor)
    R_lr = downsample_roughness_lean(R_hr, N_hr, factor=factor)
    if metal_mode == "threshold":
        M_lr = downsample_metallic(M_hr, threshold=0.5, factor=factor)
    else:
        M_lr = box_downsample(M_hr, factor=factor)
    AO_lr = downsample_ao(AO_hr, factor=factor)

    return (
        A_lr.astype(np.float32, copy=False),
        N_lr.astype(np.float32, copy=False),
        R_lr.astype(np.float32, copy=False),
        M_lr.astype(np.float32, copy=False),
        AO_lr.astype(np.float32, copy=False),
    )
