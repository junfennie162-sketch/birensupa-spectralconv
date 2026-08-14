# 官方要求 ↔ 资产对齐卡（2026-08-03 · 材料维护复核）

> 对照官网「提交规范 / Agent 必须项」与手册「最低提交物」。live 树：`/workspace/ai4s-f/submission`。  
> 主报：公开 NS64 L2=**0.035302**（`freeze_r9` · v8）；Spectral idle **3.811 / 8.054 / 29.560 ms**（冻结）。  
> 评测报告：`/workspace/评测报告_最新指标_2026-08-03_214400.md`（本轮无新 promote · §1 持平）。  
> 旧卡 [`OFFICIAL_ASSET_ALIGNMENT_2026-08-02.md`](OFFICIAL_ASSET_ALIGNMENT_2026-08-02.md) 标为历史。

| 官方条目 | 资产锚 | 复核 |
|----------|--------|:----:|
| 源码 SUPA/Extension | `spectral_conv/` + `fno_ns/` | OK |
| 依赖与命令 | `README.md` / `scripts/setup_env.sh` / `skill.md` | OK |
| 正确性脚本+结果 | `test_accuracy.py` + `summary.json` rel≈2.17e-7 | OK |
| 性能脚本+报告 | formal idle 三档冻结；`results.md`；batch16 2026-08-03 | OK |
| 单卡日志/截图 | `demo/media/brsmi_snapshot.txt`（**2026-08-03 21:43**） | OK |
| Agent 日志 ≥5 段 ≥3 类 | `development_log.md` 记录 **1–34**；精品 **24–34** | OK |
| skill.md | 提交根 `skill.md` + `skills/*` | OK |
| 展示（建议） | PPT + figures + demo media + 90s 分镜 | OK |
| FNO≥4 层 + L2 + 可视化 | L2 **0.035302**；图 08-02；strip 字段齐全 | OK |
| 数据披露 | `results/data_disclosure.md` 主报已写 freeze_r9 | OK |
| 失败矩阵 | `results/experiment_matrix.md` KEEP=freeze_r9 | OK |
| 评委入口 | README「评委 3 分钟路径」+ `CURRENT.md` | OK |
| 扩展抽查 | `extension_showcase.md` + `SPECTRAL_BONUS_AUDIT_CARD.md` | OK |
| 性能叙事 | `spectral_perf_narrative_index.md`（六轴；不解冻 ms） | OK |

**清单真源**：[`SUBMISSION_CHECKLIST.md`](../../SUBMISSION_CHECKLIST.md)  
**计划真源**：[`OPT_WAVE_MULTIAGENT_PLAN_2026-08-03.md`](OPT_WAVE_MULTIAGENT_PLAN_2026-08-03.md)
