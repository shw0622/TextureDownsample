import subprocess
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio

PROJECT = Path(__file__).resolve().parent.parent


def _write_u8_rgba(path, arr01):
    iio.imwrite(str(path), (np.clip(arr01, 0, 1) * 255 + 0.5).astype(np.uint8))


def _make_character_asset(asset_dir):
    tex = asset_dir / "Texture"
    tex.mkdir(parents=True)
    d = np.full((8, 8, 3), 0.5, dtype=np.float32)
    _write_u8_rgba(tex / "asset_d.png", d)
    n = np.zeros((8, 8, 4), dtype=np.float32)
    n[..., 0] = 0.0   # metal
    n[..., 1] = 0.5   # nx -> 0
    n[..., 2] = 0.4   # rough
    n[..., 3] = 0.5   # ny -> 0
    _write_u8_rgba(tex / "asset_n.png", n)


def test_cli_end_to_end(tmp_path):
    asset = tmp_path / "myasset"
    _make_character_asset(asset)
    out = tmp_path / "out"

    result = subprocess.run(
        [sys.executable, str(PROJECT / "run.py"),
         "--src-dir", str(asset), "--out", str(out), "--factor", "2"],
        cwd=str(PROJECT), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "myasset" / "asset_d.png").exists()
    assert (out / "myasset" / "asset_n.png").exists()
    # output is half-resolution
    d_out = iio.imread(str(out / "myasset" / "asset_d.png"))
    assert d_out.shape[:2] == (4, 4)


def test_cli_missing_textures_errors(tmp_path):
    empty = tmp_path / "empty"
    (empty / "Texture").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, str(PROJECT / "run.py"),
         "--src-dir", str(empty), "--out", str(tmp_path / "out")],
        cwd=str(PROJECT), capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_cli_bad_factor_errors(tmp_path):
    asset = tmp_path / "myasset"
    _make_character_asset(asset)
    result = subprocess.run(
        [sys.executable, str(PROJECT / "run.py"),
         "--src-dir", str(asset), "--out", str(tmp_path / "out"), "--factor", "1"],
        cwd=str(PROJECT), capture_output=True, text=True,
    )
    assert result.returncode == 2          # clean argparse error, not a traceback
    assert "Traceback" not in result.stderr  # must NOT leak an unguarded exception
