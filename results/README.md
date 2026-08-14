# 运行产物

| 路径 | 作用 |
|------|------|
| `summary.json` | **主报真源**（公开 L2 / Spectral formal） |
| `phase_status.json` | Phase 状态机 |
| `run_logs/CURRENT.md` | **现行指针**（新会话先读） |
| `run_logs/` | 日志与计划卡（现行 / 历史并存；见 `FILE_CONVENTIONS.md`） |
| `figures/` | 流场对比等（主展示用最新日期戳） |
| `archives/` | 提交包快照（冻结，勿回写 live） |
| `data_disclosure.md` / `experiment_matrix.md` | 数据口径与 KEEP/KILL |
| `PPT技术总览_*.md` | 答辩叙事（文件名可旧，正文对齐现行） |

规范全文：[`../FILE_CONVENTIONS.md`](../FILE_CONVENTIONS.md)

## 现行 vs 历史（速查）

| 现行 | 历史（勿当任务单） |
|------|-------------------|
| `CURRENT.md` · `OPT_WAVE_MULTIAGENT_PLAN_*` · `LOOP_PROCESS.md` | `OPT_MASTER` · `OPT_ROUND2…10` · `HANDOFF_*` |
| `/workspace/评测报告_最新指标_*.md`（唯一） | 已删旧戳；archives 内旧报告 |

## Phase 维护

```bash
cd /workspace/ai4s-f/submission
./scripts/maintain_assets.sh check <phase>
./scripts/maintain_assets.sh mark-done <phase>   # 校验路径后人工确认语义
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
```

## 官方文档引用

- 选手手册：`/workspace/赛题文档/算子与模型赛道选手手册.md`
- 官网提交规范：`/workspace/赛题文档/官网-赛道五-模型与算子详情页.md`
- 队内：根 `AGENTS.md`（评测报告 / Agent 日志 / 官方资产 / OPT Loop）
