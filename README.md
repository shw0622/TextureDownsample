# DPID + LEAN PBR Downsample Pipeline

## 安装依赖

```bash
pip install -r requirements.txt
```

## 用法

```bash
python run.py --src-dir <asset_folder> [--out out] [--factor 2] [--metal threshold|avg]
```

- `--src-dir`：单个资产目录，含打包贴图 `_d` / `_n`（或其 `Texture/` 子目录）。
- `--out`：输出根目录，默认 `./out`，结果写到 `<out>/<asset>/`。
- `--factor`：下采样倍数（>=2），默认 2。
- `--metal`：`threshold`（默认，二值化，物理正确）或 `avg`（box 平均）。

## 通道处理

| 通道 | 方法 |
|------|------|
| Albedo | DPID（lam=1.0, support=4） |
| Normal | box 平均 + 重归一化 |
| Roughness | LEAN 方差补偿（吸收 footprint 法线方差） |
| Metallic | box 平均 + 二值阈值 0.5（默认）/ box 平均（avg） |
| AO | box 平均 |

## 输入 / 输出格式

支持两种 dataset 打包布局，自动识别，输出与输入同构：

- **packed**（文件名 stem 含 `d`/`n` token，如 `t_x_d_mb` / `t_x_n_mb`）：
  `_d` = RGB albedo(sRGB) + A(AO)；`_n` = R(nx) G(ny) B(rough) A(metal)。
- **character**（stem 以 `_d`/`_n` 结尾，如 `asset_d` / `asset_n`）：
  `_d` = RGB albedo(sRGB)（无 AO，默认 1.0）；`_n` = R(metal) G(nx) B(rough) A(ny)。

normalZ 不存储，由 `z=√(1-x²-y²)` 重建。输出始终 8-bit。
