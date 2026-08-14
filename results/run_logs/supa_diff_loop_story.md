# 可微鸿沟闭环 · CPU 训 → SUPA 推 → P4 Backward（旁注叙事）

> 对应创新方向 A5。不替换主训路径；不写入 Spectral formal ms 主表。

## 为什么这样设计

| 路径 | 用途 | 设备 |
|------|------|------|
| 训练 | `FNO2d(..., use_supa=False)`，纯 torch 可微 | CPU |
| 推理 / chain | `spectral_conv2d_supa` / fused auto | SUPA |
| 扩展证据 | `spectral_conv/test_backward.py` | SUPA mul + CPU 参考 grad |

fused 前向对权重 `detach()`，长训接 SUPA mul 曾导致 Host↔Device 风暴（见 development_log 记录 7）。因此正式训练保持 CPU；必选题扩展分用独立 backward 单测证明可微能力。

## 可核查命令

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-f/submission/spectral_conv
python3 test_backward.py          # worst grad rel ~6e-8
cd ../fno_ns
python3 test_chain_cpu_supa_consistency.py   # ckpt chain ≤1e-4
```

## 与主报的关系

- 公开 L2 **0.035725**（`sched_samp_r5`）来自 CPU 训练 + 公开集评测，checkpoint：`fno_ns_public_demo.pt`。  
- SUPA 负责推理正确性/性能与算子扩展，不要求「真 SUPA 长训」才能交卷。  
- 若需演示微 batch 混合可微 smoke：仅旁注，禁止 promote / 写 `spectral_conv.perf`。  
- 扩展命令入口见 `extension_showcase.md` / `SPECTRAL_BONUS_AUDIT_CARD.md`。
