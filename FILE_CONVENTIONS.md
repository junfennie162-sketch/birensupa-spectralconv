# 提交树文件规范（2026-08-02）

> 工作区：`/workspace/ai4s-f/submission`。精度优化另议；本文件只管**命名、入口、保留策略**。

## 1. 三层文档

| 层 | 含义 | 例子 | 可否当「现行」 |
|----|------|------|:-------------:|
| **现行** | 主报 / 自检 / 官方必须 | `summary.json`、`skill.md`、`SUBMISSION_CHECKLIST.md`、`LOOP_PROCESS.md`、唯一评测报告 | ✅ |
| **历史** | 已收口轮次/实验痕迹 | `OPT_ROUND2…6`、`OPT_MASTER`、旧 `spectral_perf_*.md` | ❌（文首须标历史） |
| **归档** | 打包快照 | `results/archives/fandougarden_submit_*` | ❌（冻结，勿回写） |

## 2. 现行入口（只认这些）

| 用途 | 路径 |
|------|------|
| 主报数字 | `results/summary.json` → `fno_ns.public_ns64` / `spectral_conv.perf` |
| 官方对照 | `SUBMISSION_CHECKLIST.md` |
| OPT 流程 | `skills/operator_opt_loop/LOOP_PROCESS.md` |
| 自检 | `python3 skills/operator_opt_loop/run_loop.py --dry-run --strict` |
| 行动姿态 | `results/run_logs/CURRENT.md` |
| 新会话交接 | `results/run_logs/HANDOFF_NEW_SESSION_YYYY-MM-DD.md`（现行：`HANDOFF_NEW_SESSION_2026-08-04.md`） |
| 最新计划卡 | `results/run_logs/OPT_WAVE_MULTIAGENT_PLAN_2026-08-03.md`（精度线已停） |
| Agent 日志 | `development_log.md` |
| 评测报告 | `/workspace/评测报告_最新指标_*.md`（**全局唯一**） |
| SCP / 指标快照 | `demo/scp_description.md` · `demo/media/metrics_snapshot.md` |

## 3. 命名约定

| 类型 | 模式 | 说明 |
|------|------|------|
| 计划卡 | `OPT_ROUND{N}_PLAN_YYYY-MM-DD.md` | 新轮次新文件；旧卡文首写「历史 / 已接棒」 |
| 交接稿 | `HANDOFF_NEW_SESSION_YYYY-MM-DD.md` | 新会话换戳；旧稿文首标「历史」并链到现行 |
| 对齐卡 | `OFFICIAL_ASSET_ALIGNMENT_YYYY-MM-DD.md` | 材料对照 |
| 评测报告 | `评测报告_最新指标_YYYY-MM-DD_HHMMSS.md` | 换戳删旧，不做 `.bak` |
| 运行日志 | `{topic}_{YYYY-MM-DD}.md` | 可并存多日；正式数字以 summary 为准 |
| 可视化 | `fno_ns_pred_vs_gt_YYYY-MM-DD.png` | demo/media 与 figures 可同戳；**主展示用最新戳** |
| 提交包 | `fandougarden_submit_YYYYMMDD_HHMMSS.tar.gz` | 只进 `results/archives/` |

## 4. 禁止

1. 把 `archives/` 里旧数字回写到 live 主报  
2. 无 promote 时编新评测报告正式 `v` 号  
3. 默认重跑 `test_perf.py` 覆写 formal idle ms  
4. 长 `AwaitShell` 挂训练；精度探针用 `nohup` + `--stop-on-gate`  
5. 新建「又一份最新评测报告」而不删旧戳  

## 5. 历史文件怎么处理

- **不删**旧 `run_logs` / 旧 figures：答辩要轨迹。  
- 若文内仍写「现行方针」且已过时：文首加一行 `> **历史** · 现行见 CURRENT.md`。  
- PPT 文件名可保留旧日期；正文标题与指针必须对齐现行主报。

## 6. 维护口令

```bash
cd /workspace/ai4s-f/submission
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
./scripts/maintain_assets.sh check submit_gate
# 材料变更后：刷新 CURRENT.md / checklist / 唯一评测报告旁注；稳定文件 sync ai4s
```
