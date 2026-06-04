"""PBR 5 通道下采样：normal / roughness(LEAN) / metallic / ao。

roughness 用 LEAN 各向同性近似（Dupuy 2013），把
footprint 内法线方差吸收进粗糙度。
"""
from __future__ import annotations

import numpy as np

from dpid_lean.filters import box_downsample


def downsample_normal(N_hr: np.ndarray, factor: int = 2) -> np.ndarray:
    """法线 factor× 下采样：box average + 重归一化。N_hr: (H,W,3) 单位向量。"""
    if N_hr.ndim != 3 or N_hr.shape[-1] != 3:
        raise ValueError("downsample_normal: 输入 shape 必须为 (H, W, 3)")
    N_avg = box_downsample(N_hr, factor=factor)
    length = np.linalg.norm(N_avg, axis=-1, keepdims=True)
    return N_avg / np.maximum(length, 1e-8)


def downsample_roughness_lean(
    R_hr: np.ndarray, N_hr: np.ndarray, factor: int = 2
) -> np.ndarray:
    """LEAN 方差补偿 roughness 下采样。

        alpha2_avg = box((R_hr ** 2) ** 2)      # = box(R_hr ** 4)
        sigma2     = clip((1 - |box(N_hr)|^2) / 2, 0, inf)
        alpha2_lr  = alpha2_avg + 2 * sigma2
        R_lr       = clip(sqrt(sqrt(alpha2_lr)), 0, 1)

    sigma2 独立用 box(N_hr) 计算，与 Normal 通道解耦。
    """
    if R_hr.ndim != 3 or R_hr.shape[-1] != 1:
        raise ValueError("downsample_roughness_lean: R_hr 末维必须为 1")
    if N_hr.ndim != 3 or N_hr.shape[-1] != 3:
        raise ValueError("downsample_roughness_lean: N_hr 末维必须为 3")
    if R_hr.shape[:2] != N_hr.shape[:2]:
        raise ValueError(
            f"downsample_roughness_lean: R_hr {R_hr.shape[:2]} 与 "
            f"N_hr {N_hr.shape[:2]} 空间分辨率必须一致"
        )
    alpha_sq_hr = (R_hr ** 2) ** 2
    alpha_sq_avg = box_downsample(alpha_sq_hr, factor=factor)
    N_avg = box_downsample(N_hr, factor=factor)
    N_avg_len_sq = np.sum(N_avg ** 2, axis=-1, keepdims=True)
    sigma_sq_normal = np.clip((1.0 - N_avg_len_sq) / 2.0, 0.0, None)
    alpha_sq_lr = alpha_sq_avg + 2.0 * sigma_sq_normal
    R_lr = np.sqrt(np.sqrt(alpha_sq_lr))
    return np.clip(R_lr, 0.0, 1.0)


def downsample_metallic(
    M_hr: np.ndarray, threshold: float = 0.5, factor: int = 2
) -> np.ndarray:
    """Metallic factor× 下采样：box average + 二值阈值（金属/非金属是物理跳变）。

    本管线输入为 float32，输出 {0.0, 1.0}；整数输入会得到 {0,1} 而非原量纲，故仅支持 float 输入。
    """
    if M_hr.ndim != 3 or M_hr.shape[-1] != 1:
        raise ValueError("downsample_metallic: M_hr 末维必须为 1")
    M_avg = box_downsample(M_hr, factor=factor)
    return (M_avg >= threshold).astype(M_hr.dtype)


def downsample_ao(AO_hr: np.ndarray, factor: int = 2) -> np.ndarray:
    """AO factor× 下采样：直接 box average。"""
    if AO_hr.ndim != 3 or AO_hr.shape[-1] != 1:
        raise ValueError("downsample_ao: AO_hr 末维必须为 1")
    return box_downsample(AO_hr, factor=factor)
