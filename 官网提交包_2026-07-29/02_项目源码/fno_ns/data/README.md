# 数据说明（FNO-NS）

## 当前默认（离线可复现 · v2）

本竞赛 Docker **HuggingFace 不可达**（已探测）。默认使用强化版离线 NS-like：

| 项 | v1（旧） | **v2（正式）** |
|----|----------|----------------|
| 缓存文件 | `ns_like_v1e-3_N512_T20_64.pt` | `ns_like_v2_N1024_T30_64.pt` |
| 样本 / 时间步 | 512 / 20 | **1024 / 30** |
| 粘度 | 1e-3 | 1e-3 |
| 非线性 | 弱 sin 混合 | 略强结构化非线性 |
| 生成器 | `dataset.py` | 同 |

有公开 `.pt`（布局 `[N,T,H,W]` 或 `[N,H,W,T]`）时可直接放进本目录；加载器会优先使用**非** `ns_like*` 文件名。

## 公开数据（有网时）

推荐：

- HuggingFace：`abelsr1710/navier-stokes-2d-fno` → `navier_stokes_v1e-3_N1200_T20.pt`
- 或 PDEBench / 原作者 NS 64×64 粘度数据

```bash
# huggingface-cli download abelsr1710/navier-stokes-2d-fno \
#   navier_stokes_v1e-3_N1200_T20.pt --local-dir ./
```

## 运行

```bash
cd submission/fno_ns
python3 test_forward.py   # 自动生成/加载 v2 并训练评估
python3 visualize.py
```
