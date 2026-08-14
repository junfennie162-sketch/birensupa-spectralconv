# Long push · SIGNAL（2026-08-06）· **已 promote → v9**

## 复评确认（官方 1000/128 · residual）

| ckpt | reeval L2 | vs gate 0.035123 |
|------|-----------|------------------|
| demo `freeze_r11` | 0.03522327 | — |
| `last_thaw_r2` | 0.03511602 | **破 gate** |
| **`dualview_r2`** | **0.03511498** | **破 gate · 最优** |

相对 demo Δ≈**1.08e−4**（略过 1e−4 纪律线）。

## 路径

- 最优：`fno_ns/checkpoints/fno_ns_public_dualview_r2_best.pt`
- 轨迹：pf_delta → qt/thaw/dualview wave1 → **last_thaw_r2 (k=2) → dualview_r2**
- **2026-08-06 人口头 promote**：`promote_public_ckpt.py --tag dualview_r2`  
  verified_test_l2=**0.035114976112** · 评测报告 **v9** · demo 已替换  
- 回滚备份：`fno_ns_public_demo_pre_dualview_r2.pt`
