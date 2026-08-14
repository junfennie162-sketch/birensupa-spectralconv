# FNO 评测协议速查（2026-07-25）

依据：`/workspace/赛题文档/赛道验收与提交清单.md` §4。

## 1. 推理主指标（batch=16）

- 设备：BIREN 单卡；输入 64×64；建议 `batch_size=16`
- warmup ≥10，正式 iters ≥50；计时边界同步
- 主指标：`grid_points/s = N_iter × B × H × W / t_wall`（单通道 H×W）
- 辅指标：`samples/s`、`ms/sample`、`forward_ms/batch`、峰值显存 MB
- 分开报「纯 forward」与「含 DataLoader」；主表优先纯 forward
- 脚本：`fno_ns/benchmark_fno_batch16.py`

## 2. 正确性前置门禁

- 同一权重/输入下，CPU reference 与 `forward_supa_chain` 相对误差建议 ≤ `1e-4`
- 未过门禁的快速路径不得写入正式性能表
- SUPA-origin fused 输入经 **pinned host staging** 再回 SUPA（suFFT 正确性必需）；CPU 起源输入不走该路径
- 脚本：`fno_ns/test_chain_cpu_supa_consistency.py`

## 3. 数据披露

- 当前正式 L2 使用自生成 `generated_ns_like_v2`，**不是**公开 NS64
- 划分与训练量见 `results/data_disclosure.md`、`checkpoints/checkpoint_meta.json`

## 4. 训练吞吐加分

- 同量纲 `grid_points/s`，但必须写明包含 forward + loss + backward + optimizer step
- 本包训练路径为 CPU / `use_supa=False`（与提交 checkpoint 一致），如实标注
- 脚本：`fno_ns/benchmark_train_throughput.py`
