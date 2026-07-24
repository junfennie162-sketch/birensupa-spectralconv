# 运行产物

| 路径 | 作用 |
|------|------|
| `run_logs/` | 命令行与评测日志 |
| `figures/` | 误差曲线、流场对比等 |
| `summary.json` | 机器可读指标（测试脚本写入） |
| `phase_status.json` | Phase 状态机与资产清单 |

## Phase 维护

每个开发 phase 结束后：

```bash
cd /workspace/ai4s-n/submission
./scripts/maintain_assets.sh check <phase>
# 回填 results.md / development_log / summary.json 业务字段后
./scripts/maintain_assets.sh mark-done <phase>
```

`mark-done` 会：

1. 校验本 phase 声明的资产路径存在
2. 将 `phase_status.json` 中该 phase 标为 `done`，推进 `current_phase`
3. 在 `summary.json.meta` 记录 `last_phase_marked`

语义验收（相对误差、相对 L2、日志段数）仍由 Agent/人手确认；脚本不伪造指标。

## 官方文档引用

- 选手手册：`/workspace/赛题文档/算子与模型赛道选手手册.md`
- 写赛题前的环境与 GEMV 基准：手册 Part A「快速开始」

## 环境基线（2026-07-21 已测）

服务器环境已通过；各组件用途见 `AGENTS.md` / 上级 `results.md` / `summary.json.env.purpose`。  
日志：`run_logs/env_baseline_2026-07-21.md`。Agent 默认进入 `spectral_accuracy`，勿无故重跑 GEMV 冒烟。
