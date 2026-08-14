# demo/media · 评委现行入口（2026-08-14）

> **只看本页列出的现行文件。** 旧日期 PNG 已移至 [`archive_history/`](archive_history/)（轨迹保留，勿当主展示）。  
> 主报：**v10** · L2 **0.035012** · `spec_ref_r2`。流场图仍钉 08-02（promote 后未重画，场形态与 v8/v9 同协议）。

## 评委必看

| 文件 | 用途 |
|------|------|
| [`fno_ns_pred_vs_gt_2026-08-02.png`](fno_ns_pred_vs_gt_2026-08-02.png) | Pred / GT / error 主图 |
| [`fno_ns_sample_strip_2026-08-02.png`](fno_ns_sample_strip_2026-08-02.png) | best / median / worst strip |
| [`metrics_snapshot.md`](metrics_snapshot.md) | L2 **0.035012** + idle 三档 |
| [`brsmi_snapshot.txt`](brsmi_snapshot.txt) | 单卡 Biren 运行日志快照（08-14 刷新） |
| [`official_recheck_2026-08-14.log`](official_recheck_2026-08-14.log) | 交卷复测原始日志 |

## 瓶颈解剖（Error Autopsy D · 2026-08-04）

| 文件 | 用途 |
|------|------|
| [`protocol_vs_fno_paper_0128.png`](protocol_vs_fno_paper_0128.png) | 论文 0.0128 与队内 clean 10→1 **不可横比** |
| [`error_decomp_e1_tf_ar.png`](error_decomp_e1_tf_ar.png) | e1 / e2_TF / e2_AR / exposure gap g |
| [`spectrum_best_median_worst.png`](spectrum_best_median_worst.png) | 径向频谱误差贡献（best/median/worst） |
| [`spectrum_best_median_worst_heatmaps.png`](spectrum_best_median_worst_heatmaps.png) | 涡度 GT/Pred/Error 条带 |
| 裁决卡 | [`../../results/run_logs/ERROR_AUTOPSY_VERDICT_2026-08-04.md`](../../results/run_logs/ERROR_AUTOPSY_VERDICT_2026-08-04.md) |

## 旁注

- 历史图：`archive_history/`（07-21～08-01）
- SCP 文案：[`../scp_description.md`](../scp_description.md)
- 评委一页包：[`../../results/run_logs/JUDGE_3MIN_PACK_2026-08-04.md`](../../results/run_logs/JUDGE_3MIN_PACK_2026-08-04.md)
