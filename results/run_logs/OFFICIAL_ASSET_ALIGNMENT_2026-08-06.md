# 官方要求 ↔ 资产对齐卡（2026-08-06 · v9 promote 后）

> live 树：`/workspace/ai4s-f/submission`（已 sync `/workspace/ai4s/submission`）。  
> 主报：公开 NS64 L2=**0.035115**（`dualview_r2` · v9）；Spectral idle **3.811 / 8.054 / 29.560 ms**（冻结）。  
> 评测报告：`/workspace/评测报告_最新指标_2026-08-06_174400.md`。  
> 旧卡 [`OFFICIAL_ASSET_ALIGNMENT_2026-08-04.md`](OFFICIAL_ASSET_ALIGNMENT_2026-08-04.md) 标为历史。

| 官方条目 | 资产锚 | 复核 |
|----------|--------|:----:|
| 源码 SUPA/Extension | `spectral_conv/` + `fno_ns/` | OK |
| 依赖与命令 | `README.md` / `scripts/setup_env.sh` / `skill.md` | OK |
| 正确性脚本+结果 | `test_accuracy.py` + `summary.json` rel≈2.17e-7 | OK |
| 性能脚本+报告 | formal idle 三档冻结；六轴口播 | OK |
| 单卡日志/截图 | `demo/media/brsmi_snapshot.txt` | OK |
| Agent 日志 ≥5 段 ≥3 类 | `development_log.md` 记录 **1–40** | OK |
| skill.md | 提交根 `skill.md` + `skills/*` | OK |
| 展示（建议） | PPT / JUDGE / demo 现行图 | OK |
| FNO≥4 层 + L2 + 可视化 | L2 **0.035115** · dualview_r2；promote 刷新 strip | OK |
| 数据披露 | `data_disclosure.md` · v9 | OK |
| 失败矩阵 | `experiment_matrix.md` KEEP=`dualview_r2` | OK |
| 评委入口 | README → `JUDGE_3MIN_PACK_2026-08-04.md`（数字已跟 v9） | OK |
| 提交包 | `fandougarden_submit_20260806_181107.tar.gz` · 已入 ai4s archives | OK |

**清单真源**：[`SUBMISSION_CHECKLIST.md`](../../SUBMISSION_CHECKLIST.md)  
**自检（2026-08-07）**：`run_loop --dry-run --strict` · `maintain check submit_gate` → hard PASS
