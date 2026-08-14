# 官方要求 ↔ 资产对齐卡（2026-08-04 · 材料/答辩闭环）

> 对照官网「提交规范 / Agent 必须项」与手册「最低提交物」。live 树：`/workspace/ai4s-f/submission`。  
> 主报：公开 NS64 L2=**0.035302**（`freeze_r9` · v8）；Spectral idle **3.811 / 8.054 / 29.560 ms**（冻结）。  
> 评测报告：`/workspace/评测报告_最新指标_2026-08-04_155200.md`（**无新 promote** · §1 持平 0% · §2.1 含 Autopsy/PF）。  
> 旧卡 [`OFFICIAL_ASSET_ALIGNMENT_2026-08-03.md`](OFFICIAL_ASSET_ALIGNMENT_2026-08-03.md) 标为历史。

| 官方条目 | 资产锚 | 复核 |
|----------|--------|:----:|
| 源码 SUPA/Extension | `spectral_conv/` + `fno_ns/` | OK |
| 依赖与命令 | `README.md` / `scripts/setup_env.sh` / `skill.md` | OK |
| 正确性脚本+结果 | `test_accuracy.py` + `summary.json` rel≈2.17e-7 | OK |
| 性能脚本+报告 | formal idle 三档冻结；六轴口播单页 | OK |
| 单卡日志/截图 | `demo/media/brsmi_snapshot.txt` | OK |
| Agent 日志 ≥5 段 ≥3 类 | `development_log.md` 记录 **1–36**；精品含 **35–36** | OK |
| skill.md | 提交根 `skill.md` + `skills/*` | OK |
| 展示（建议） | PPT 冻结稿 + JUDGE 一页包 + demo 现行图 + 90s 分镜 | OK |
| FNO≥4 层 + L2 + 可视化 | L2 **0.035302**；图仅钉 08-02（旧图 archive） | OK |
| 数据披露 | `data_disclosure.md` · freeze_r9；r10 旁注已回滚 | OK |
| 失败矩阵 | `experiment_matrix.md` KEEP=freeze_r9；ABORT/NO_SIGNAL 齐全 | OK |
| 评委入口 | README 步0 → `JUDGE_3MIN_PACK_2026-08-04.md` | OK |
| 扩展抽查 | `extension_showcase` + `SPECTRAL_BONUS_AUDIT_CARD` | OK |
| 性能叙事 | `SPECTRAL_SIX_AXIS_ORAL_2026-08-04.md` + narrative index | OK |

**清单真源**：[`SUBMISSION_CHECKLIST.md`](../../SUBMISSION_CHECKLIST.md)  
**计划**：材料闭环 `/root/.cursor/plans/materials_defense_loop_20260804.plan.md` · OPT_WAVE 仍为精度停真源
