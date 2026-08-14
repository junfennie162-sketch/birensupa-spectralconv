# FNO 评测协议速查（2026-08-02）

依据：[`/workspace/赛题文档/算子与模型赛道选手手册.md`](/workspace/赛题文档/算子与模型赛道选手手册.md) 提交/验收相关章节；行动方针见 `results/run_logs/CURRENT.md`；轮次纪律见 `skills/operator_opt_loop/LOOP_PROCESS.md`。

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
- SUPA-origin fused 输入经 **pinned / host-seeded staging** 再回 SUPA（suFFT 正确性必需）；CPU 起源输入不走该路径
- 脚本：`fno_ns/test_chain_cpu_supa_consistency.py`

## 3. 数据披露与 L2 主报

- **正式 L2（主报）**：公开 NS64，`navier_stokes_v1e-3_N1200_T20.pt`，划分 **1000/128**，seed=`20260722`
- 当前主报 relative L2：**0.035012**（`spec_ref_r2` · **v10**；`fno_ns/checkpoints/fno_ns_public_demo.pt`）
- 上一正式 v9：**0.035115**（`dualview_r2`）
- batch16 协议复测（公开 NS64）：≈**1.60M** grid_points/s · peak≈**202 MB**（见 `fno_batch16_benchmark_public_ns64_2026-08-02.md`）；chain 门禁以 **B=4** 为准
- 自建 `generated_ns_like_v2`（含 continue3≈0.005144、768 旁注≈0.00249）仅为工程对照，**禁止**写成公开/官方成绩
- 详见 `results/data_disclosure.md`、`results/summary.json` → `fno_ns.public_ns64`

## 4. 训练吞吐加分

- 同量纲 `grid_points/s`，但必须写明包含 forward + loss + backward + optimizer step
- 本包训练路径为 CPU / `use_supa=False`（与提交 checkpoint 一致），如实标注
- 脚本：`fno_ns/benchmark_train_throughput.py`
