# 扩展项展示闭环（bwd / 3D / irregular）

> 功能已实现并通过正确性测试；本页提供评委「怎么跑 + 数字锚 + 边界」。  
> **3D 是算子扩展，不是完整 3D FNO。**  
> 抽查入口：[`SPECTRAL_BONUS_AUDIT_CARD.md`](SPECTRAL_BONUS_AUDIT_CARD.md) · README「评委 3 分钟路径」步 3。  
> **SOL / sol_proxy 仅旁注，禁止写成官方得分句。**

## 命令三件套

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission

# Backward（spectral_mul autograd）
cd spectral_conv && python3 test_backward.py

# 3D SpectralConv（官网四角：CPU rfftn + SUPA mul ×4 + irfftn）
python3 test_3d_accuracy.py

# Irregular shapes（稳健性）
python3 test_irregular_shapes.py
```

也可：`./scripts/run_tests.sh backward|3d|irregular`（`all` 已含 irregular）。

## 数字锚（材料层）

| 扩展 | 结果 | 日志 / 字段 |
|------|------|-------------|
| Backward | 3/3 PASS，worst grad rel ≈ **6.25e-8** | `results.md` §2.4；`summary.optimization.p4_backward` |
| 3D | 2/2 PASS，worst rel ≈ **1.19e-7**；**四角** weights1–4 | `spectral_3d_accuracy_*.md`；`p6_3d` |
| irregular | 9/9 PASS，worst rel ≈ **3.20e-7** | `spectral_irregular_2026-08-02.md`；`summary.spectral_conv.irregular` |

## 诚实边界

| 主张 | 边界 |
|------|------|
| 可微扩展 | mul 路径有 bwd；**长训仍走 CPU torch**（见 `supa_diff_loop_story.md`） |
| 3D | 官方四角正确性冒烟；非 suFFT3d 全链路、非 3D FNO 重训 |
| irregular | 非方/非常规 modes 稳健性；不进 formal 三档 perf |

## 可微鸿沟（A5）

CPU 训 → SUPA 推 → P4 bwd 单测：[`supa_diff_loop_story.md`](supa_diff_loop_story.md)
