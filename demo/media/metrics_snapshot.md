# Metrics snapshot · 2026-08-14（提交）

## 正式（公开 NS64）

- FNO official-split (1000/128) relative L2: **0.035011906176805496**（`spec_ref_r2` promote · 评测报告 **v10**）
- Data: `fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`
- Checkpoint: `fno_ns/checkpoints/fno_ns_public_demo.pt`
- 轨迹：… → freeze_r9 → dualview_r2 → **spec_ref_r2**

## SpectralConv（与 NS 数据无关）

- idle（2026-08-14 复测）：**3.797 / 8.037 / 29.295 ms** @64/128/256
- 07-31 冻结板：3.811 / 8.054 / 29.560 ms（噪声内）
- accuracy worst rel: **2.170e-7**（阈值 1e-4）
- 3D four-corner worst rel: ≈1.19e-7（算子扩展，≠3D FNO）

## FNO batch16（公开语义）

- pure forward ≈ **1.600M** grid_points/s · ≈391 samp/s · peak ≈ **202 MB**（2026-08-03 · 旁注）

## 工程对照（自建 v2，非公开分）

- continue3 L2: **0.005143815**（`fno_ns_demo.pt`）

## 图

- `fno_ns_pred_vs_gt_2026-08-02.png`, `fno_ns_sample_strip_2026-08-02.png`
