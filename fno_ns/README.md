# FNO · Navier-Stokes（进阶）

复用 `../spectral_conv` 必选算子，≥4 层 FNO，公开 NS64 涡度 10→1。

## 现行主报

| 项 | 值 |
|----|-----|
| 数据 | `data/navier_stokes_v1e-3_N1200_T20.pt`（1000/128 · seed 20260722） |
| 协议 | eval 10→1 · residual · 相对 L2 |
| Checkpoint | `checkpoints/fno_ns_public_demo.pt` |
| Tag / 版本 | `spec_ref_r2` · **v10** |
| 公开 L2 | **0.035012** |
| 结构 | 4 层 · width=32 · modes=16 · 64×64 |

工程旁注（**非公开分**）：自建 v2 `fno_ns_demo.pt` L2 ≈ 0.005144。口径见 `../results/data_disclosure.md`。

## 评委入口

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission/fno_ns
python3 test_forward.py     # 公开集复评
python3 visualize.py        # Pred / GT
```

## 文件（现行 vs 探针）

| 文件 | 角色 |
|------|------|
| `model.py` | FNO（eval：fused；train：CPU 可微） |
| `test_forward.py` | 公开集前向 + 相对 L2 |
| `visualize.py` | 流场图 |
| `train_public_spectral_refiner_probe.py` | v10 机制（Spectral-Refiner lite） |
| `promote_public_ckpt.py` | 破 gate 后人工 promote |
| 其它 `train_public_*.py` | **历史探针**（答辩轨迹，勿当现行入口） |

权重说明：`checkpoints/README.md`。
