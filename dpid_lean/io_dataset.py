"""dataset 打包贴图的加载/保存，支持 packed 与 character 两种 _n 通道布局。

packed     : _d = RGB albedo(sRGB) + A=AO ; _n = R=nx, G=ny, B=rough, A=metal
character  : _d = RGB albedo(sRGB), AO=1   ; _n = R=metal, G=nx, B=rough, A=ny
两种布局的 normalZ 都由 z = sqrt(1 - x^2 - y^2) 重建。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import imageio.v3 as iio

# (A, N, R, M, AO)
PBRSet = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# --------------------------------------------------------------------------
# sRGB <-> linear
# --------------------------------------------------------------------------
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    low = x / 12.92
    high = ((x + a) / (1.0 + a)) ** 2.4
    return np.where(x <= 0.04045, low, high).astype(x.dtype, copy=False)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    low = x * 12.92
    high = (1.0 + a) * (x ** (1.0 / 2.4)) - a
    out = np.where(x <= 0.0031308, low, high)
    return np.clip(out, 0.0, 1.0).astype(x.dtype, copy=False)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _to01(raw: np.ndarray) -> np.ndarray:
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / 65535.0
    return raw.astype(np.float32)


def _reconstruct_normal(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """nx, ny in [-1, 1], shape (H, W, 1) each -> unit normal (H, W, 3)."""
    nz = np.sqrt(np.maximum(1.0 - nx * nx - ny * ny, 0.0))
    N = np.concatenate([nx, ny, nz], axis=-1).astype(np.float32)
    return N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-8)


def _texture_dir(path: Path) -> Path:
    return path / "Texture" if (path / "Texture").is_dir() else path


def _find_by_suffix(folder: Path, suffix: str) -> Path | None:
    suffix = suffix.lower()
    for f in sorted(folder.glob("*.png")):
        if f.stem.lower().endswith(suffix):
            return f
    return None


def _find_by_token(folder: Path, token: str) -> Path | None:
    token = token.lower()
    for f in sorted(folder.glob("*.png")):
        if token in f.stem.lower().split("_"):
            return f
    return None


def detect_layout(d_stem: str, n_stem: str) -> str:
    """character if either stem ends with _d/_n, else packed."""
    ds, ns = d_stem.lower(), n_stem.lower()
    if ds.endswith("_d") or ns.endswith("_n"):
        return "character"
    return "packed"


# --------------------------------------------------------------------------
# load / save
# --------------------------------------------------------------------------
def load_asset(asset_dir: Path) -> tuple[PBRSet, str, str, str]:
    """Return ((A,N,R,M,AO), layout, d_stem, n_stem). Albedo is linear."""
    folder = _texture_dir(Path(asset_dir))

    # prefer character (_d/_n suffix), then packed (d/n token)
    # layout 由匹配到的文件名推断，规则与 detect_layout() 一致：_d/_n 后缀=character，否则 d/n token=packed
    p_d = _find_by_suffix(folder, "_d")
    p_n = _find_by_suffix(folder, "_n")
    if p_d is not None and p_n is not None:
        layout = "character"
    else:
        p_d = _find_by_token(folder, "d")
        p_n = _find_by_token(folder, "n")
        layout = "packed"

    if p_d is None or p_n is None:
        raise FileNotFoundError(f"no _d/_n textures found in {folder}")

    raw_d = _to01(iio.imread(str(p_d)))
    raw_n = _to01(iio.imread(str(p_n)))
    if raw_n.ndim != 3 or raw_n.shape[-1] != 4:
        raise ValueError(f"_n must be RGBA, got {raw_n.shape} for {p_n}")
    if raw_d.ndim != 3 or raw_d.shape[-1] < 3:
        raise ValueError(f"_d must be RGB/RGBA, got {raw_d.shape} for {p_d}")
    if raw_d.shape[:2] != raw_n.shape[:2]:
        raise ValueError(f"_d/_n size mismatch: {raw_d.shape[:2]} vs {raw_n.shape[:2]}")

    H, W = raw_n.shape[:2]
    A = srgb_to_linear(raw_d[..., :3]).astype(np.float32, copy=False)

    if layout == "character":
        AO = np.ones((H, W, 1), dtype=np.float32)
        M = raw_n[..., 0:1].astype(np.float32)
        nx = raw_n[..., 1:2] * 2.0 - 1.0
        R = raw_n[..., 2:3].astype(np.float32)
        ny = raw_n[..., 3:4] * 2.0 - 1.0
    else:  # packed
        AO = (raw_d[..., 3:4].astype(np.float32)
              if raw_d.shape[-1] >= 4 else np.ones((H, W, 1), dtype=np.float32))
        nx = raw_n[..., 0:1] * 2.0 - 1.0
        ny = raw_n[..., 1:2] * 2.0 - 1.0
        R = raw_n[..., 2:3].astype(np.float32)
        M = raw_n[..., 3:4].astype(np.float32)

    N = _reconstruct_normal(nx, ny)
    return (A, N, R, M, AO), layout, p_d.stem, p_n.stem


def _u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_asset(
    out_dir: Path,
    pbr: PBRSet,
    layout: str,
    d_stem: str,
    n_stem: str,
) -> None:
    """Write (A,N,R,M,AO) back in the same packed layout it was loaded from.
    输出始终为 8-bit uint8（dataset 贴图惯例）；16-bit 源贴图保存时会量化到 8-bit。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    A, N, R, M, AO = pbr

    A_srgb = linear_to_srgb(A)
    nx01 = np.clip(N[..., 0:1] * 0.5 + 0.5, 0.0, 1.0)
    ny01 = np.clip(N[..., 1:2] * 0.5 + 0.5, 0.0, 1.0)
    R01 = np.clip(R, 0.0, 1.0)
    M01 = np.clip(M, 0.0, 1.0)
    AO01 = np.clip(AO, 0.0, 1.0)

    if layout == "character":
        # character: _d 无 AO 通道，写 3 通道 RGB；_n = R=metal,G=nx,B=rough,A=ny
        d_out = A_srgb
        n_rgba = np.concatenate([M01, nx01, R01, ny01], axis=-1)
    else:  # packed: _d = RGB albedo + A=AO ; _n = R=nx,G=ny,B=rough,A=metal
        d_out = np.concatenate([A_srgb, AO01], axis=-1)
        n_rgba = np.concatenate([nx01, ny01, R01, M01], axis=-1)

    iio.imwrite(str(out_dir / f"{d_stem}.png"), _u8(d_out))
    iio.imwrite(str(out_dir / f"{n_stem}.png"), _u8(n_rgba))
