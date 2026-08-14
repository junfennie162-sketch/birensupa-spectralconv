# Metrics snapshot · 2026-08-03（提交）

## 正式（公开 NS64）

- FNO official-split (1000/128) relative L2: **0.03530218452215195**（`freeze_r9` promote · 评测报告 v8）
- Data: `fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`
- Checkpoint: `fno_ns/checkpoints/fno_ns_public_demo.pt`
- Train: … → sched_samp_r5 → R8 soft → **freeze_r9 promote**
- ROUND10：freeze/soft 探针 NO_SIGNAL（未改主报）

## SpectralConv（与 NS 数据无关）

- idle formal: **3.811 / 8.054 / 29.560 ms** @64/128/256
- accuracy worst rel: ≈2.17e-7（阈值 1e-4）
- 3D four-corner worst rel: ≈1.19e-7（算子扩展，≠3D FNO）

## FNO batch16（公开语义）

- pure forward ≈ **1.600M** grid_points/s · ≈391 samp/s · peak ≈ **202 MB**（2026-08-03 · freeze_r9 ckpt）
- 日志：`results/run_logs/fno_batch16_benchmark_public_ns64_2026-08-03.md`

## 工程对照（自建 v2，非公开分）

- continue3 L2: **0.005143815**（`fno_ns_demo.pt`）
- 零样本→公开集: ≈0.4115（说明必须重训）

## 图

- `fno_ns_pred_vs_gt_2026-08-02.png`, `fno_ns_sample_strip_2026-08-02.png`（freeze_r9 后刷新）
