# 官方要求 ↔ 资产对齐卡（2026-08-14 · v10）

> live 树：`/workspace/ai4s-f/submission`。  
> 主报：公开 NS64 L2=**0.035012**（`spec_ref_r2` · **v10**）；Spectral idle **3.797 / 8.037 / 29.295 ms**（08-14 复测）。  
> 评测报告：`/workspace/评测报告_最新指标_2026-08-14_095200.md`。  
> 旧卡 08-04 / 08-03 为历史。

| 官方条目 | 资产锚 | 复核 |
|----------|--------|:----:|
| 源码 SUPA/Extension | `spectral_conv/` + `fno_ns/` | OK |
| 依赖与编译命令 | `README.md`「交卷必跑」+ `spectral_conv/build.sh` | OK |
| 正确性脚本+结果 | `test_accuracy.py` + `正确性验证报告_2026-08-14.md` | OK |
| 性能脚本+报告 | `test_perf.py` + `性能检测报告_2026-08-14.md` | OK |
| 单卡日志 | `demo/media/brsmi_snapshot.txt`（08-14 刷新） | OK |
| Agent ≥5 段 ≥3 类 | `AGENT_OFFICIAL.md` 6 段 + `development_log.md` | OK |
| skill.md | 提交根 `skill.md` | OK |
| FNO≥4 层 + L2 + 可视化 | L2 **0.035012**；`demo/media/` 08-02 主图 | OK |
| 失败矩阵 | `experiment_matrix.md` KEEP=`spec_ref_r2` | OK |
| 评委入口 | `JUDGE_3MIN_PACK_2026-08-04.md`（正文已对齐 v10） | OK |

**清单真源**：[`SUBMISSION_CHECKLIST.md`](../../SUBMISSION_CHECKLIST.md)
