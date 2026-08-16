# Submission checklist

Mapped to Track 5 “submission requirements”. Official filenames at repo root stay English.

| 官方要求 | 我们的文件 | 说明 |
|----------|------------|------|
| `skill.md`（必须） | `skill.md`；`skills/README.md` | 提交根文件名不改；中文可读全文在 `skills/` |
| 项目源码 | `spectral_conv/`、`fno_ns/` | SUPA kernel + FNO |
| 依赖与编译 / 运行命令 | `README.md`；`spectral_conv/build.sh` | 先 source 环境再编译 |
| 正确性验证脚本与结果 | `spectral_conv/test_accuracy.py`；`results/run_logs/` | 相对误差 ≤ 1e-4 |
| 性能测试脚本与报告 | `spectral_conv/test_perf.py`；`results/run_logs/` | 64 / 128 / 256 |
| 运行日志或截图 | `results/run_logs/`；`demo/media/` | 含 `brsmi` 快照 |
| Agent 开发日志（≥5 段、≥3 类场景） | `development_log.md`；`AGENT_OFFICIAL.md`；`agent_logs/` | 摘要；完整对话见 `agent_logs/*.md`；聊天截图见 `03_*.md` |
| 测试结果说明 | `results.md` | 官方数据上的对比与改进 |
| 展示材料（建议） | `demo/` | 官方数据流场图 |

Root filenames: `README.md`, `skill.md`, `results.md`, `development_log.md`.
