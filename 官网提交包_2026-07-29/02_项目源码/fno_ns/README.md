# FNO · Navier-Stokes（进阶 C）

基于 `../spectral_conv` 自定义算子搭建完整 FNO，求解二维 Navier-Stokes 涡度方程。

## 验收

- Fourier Layer ≥ 4 层
- 数据：公开 NS 2D 推荐；本机默认离线 NS-like（见 `data/README.md`）
- 报告相对 L2；可视化预测 vs ground truth
- 推理走 fused SUPA 路径；训练走 CPU torch 可微路径

## 超参（当前正式）

| 项 | 值 |
|----|-----|
| resolution / modes / width | 64 / 12 / 32 |
| layers | 4 |
| n_train / n_test / epochs | 256 / 32 / 40 |
| lr | 8e-4（Adam + cosine） |
| 数据 | `generated_ns_like_v1e-3` |
| 相对 L2（SUPA） | ≈ **0.0173** |

## 文件

| 文件 | 作用 |
|------|------|
| `model.py` | FNO（eval：fused 双角；train：torch einsum） |
| `dataset.py` | NS-like 生成 / 缓存 / 划分 |
| `data/README.md` | 数据来源与公开集替换说明 |
| `test_forward.py` | 加长训 + SUPA 前向 + 相对 L2 |
| `visualize.py` | 流场对比图 → `../results/figures/` |

## 运行

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission/fno_ns
python3 test_forward.py
python3 visualize.py
```

## 状态

加长训已通；相对 L2 较初版 ~0.05 / 加强版 ~0.036 降至 **~0.017**。外网可用时优先替换公开 HDF5/HF。
