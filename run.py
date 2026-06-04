"""CLI: 对单个 dataset 资产做 dpid+LEAN 下采样并同构回写。

用法:
    python run.py --src-dir <asset_folder> [--out out] [--factor 2] [--metal threshold|avg]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dpid_lean.io_dataset import load_asset, save_asset
from dpid_lean.pipeline import downsample_asset


def _asset_name(src_dir: Path) -> str:
    # if pointed at a Texture/ dir, use its parent name
    if src_dir.name.lower() == "texture":
        return src_dir.parent.name
    return src_dir.name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", type=Path, required=True,
                   help="资产目录（含 _d/_n 打包贴图，或其 Texture/ 子目录）")
    p.add_argument("--out", type=Path, default=Path("out"),
                   help="输出根目录，默认 ./out")
    p.add_argument("--factor", type=int, default=2, help="下采样倍数，默认 2")
    p.add_argument("--metal", choices=["threshold", "avg"], default="threshold",
                   help="metallic 下采样模式，默认 threshold（二值）")
    args = p.parse_args(argv)

    src_dir = args.src_dir
    if not src_dir.is_dir():
        print(f"[dpid_lean] src-dir not found: {src_dir}", file=sys.stderr)
        return 2

    try:
        pbr, layout, d_stem, n_stem = load_asset(src_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[dpid_lean] load failed: {e}", file=sys.stderr)
        return 2

    print(f"[dpid_lean] {src_dir} layout={layout} factor={args.factor} metal={args.metal}")
    lr = downsample_asset(pbr, factor=args.factor, metal_mode=args.metal)

    asset = _asset_name(src_dir)
    out_dir = args.out / asset
    save_asset(out_dir, lr, layout, d_stem, n_stem)
    print(f"[dpid_lean] wrote -> {out_dir / (d_stem + '.png')} , {out_dir / (n_stem + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
