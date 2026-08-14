# 提交对照清单

对照官网赛道五「提交规范」。只列必须交的条目，以及我们对应准备好的文件。

| 官方要求 | 我们的文件 | 说明 |
|----------|------------|------|
| `skill.md`（必须） | `skill.md` | 提交根，文件名不改 |
| 项目源码 | `spectral_conv/`、`fno_ns/` | SUPA kernel + FNO |
| 依赖与编译 / 运行命令 | `README.md`；`spectral_conv/build.sh` | 先 source 环境再编译 |
| 正确性验证脚本与结果 | `spectral_conv/test_accuracy.py`；`results/run_logs/` | 相对误差 ≤ 1e-4 |
| 性能测试脚本与报告 | `spectral_conv/test_perf.py`；`results/run_logs/` | 64 / 128 / 256 |
| 运行日志或截图 | `results/run_logs/`；`demo/media/` | 含 `brsmi` 快照 |
| Agent 开发日志（≥5 段、≥3 类场景） | `development_log.md`；`AGENT_OFFICIAL.md` | 摘要；交卷包另有原始对话 |
| 测试结果说明 | `results.md` | 官方数据上的对比与改进 |
| 展示材料（建议） | `demo/` | 官方数据流场图 |

根上这四个文件名保持英文：`README.md`、`skill.md`、`results.md`、`development_log.md`。
