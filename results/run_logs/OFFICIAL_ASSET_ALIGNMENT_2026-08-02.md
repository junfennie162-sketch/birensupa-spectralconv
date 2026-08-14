# 官方要求 ↔ 资产对齐卡（2026-08-02）

> **历史** → 现行见 [`OFFICIAL_ASSET_ALIGNMENT_2026-08-03.md`](OFFICIAL_ASSET_ALIGNMENT_2026-08-03.md)（主报已升 v8 · 0.035302）。  
> 对照官网「提交规范 / Agent 必须项」与手册「最低提交物」。live 树：`/workspace/ai4s-f/submission`。

| 官方条目 | 资产锚 | 复核 |
|----------|--------|:----:|
| 源码 SUPA/Extension | `spectral_conv/` + `fno_ns/` | OK |
| 依赖与命令 | `README.md` / `scripts/setup_env.sh` / `skill.md` | OK |
| 正确性脚本+结果 | `test_accuracy.py` + `summary.json` rel≈2.17e-7 | OK |
| 性能脚本+报告 | `test_perf.py` idle 三档；`results.md` | OK |
| 单卡日志/截图 | `demo/media/brsmi_snapshot.txt`（08-02 刷新） | OK |
| Agent 日志 ≥5 段 ≥3 类 | `development_log.md` 记录 1–28 | OK |
| skill.md | 提交根 `skill.md` | OK |
| 展示（建议） | PPT + figures + demo media | OK |
| FNO≥4 层 + L2 + 可视化 | model 4 层；L2 0.035725；图 08-02 | OK |

**清单真源**：[`SUBMISSION_CHECKLIST.md`](../../SUBMISSION_CHECKLIST.md)
