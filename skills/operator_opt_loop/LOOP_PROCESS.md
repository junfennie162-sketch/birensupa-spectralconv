# operator_opt_loop · 规范流程（2026-08-02）

> 服务 Agent/Skills 抽查与队内 OPT 轮次。默认 **dry-run**：不重训、不写 formal perf。

## 一句话

读主报 → 判姿态 → 可选探针（快路径）→ 护栏 → 材料闭环 → 合入。

## P0–P6

| 步 | 名称 | 必须遵守 |
|----|------|----------|
| P0 | 环境与单卡 | `source brsw_set_env.sh`；f/n **禁止并发** GPU |
| P1 | 读主报 | 只认 `summary.json` 公开 L2 / idle ms；v2·SOL·tune 旁注 |
| P2 | 精度探针 | `nohup`；`--stop-on-gate`；`epochs≤4`；`patience≤2`；**禁长 AwaitShell** |
| P3 | gate 裁决 | `best<gate` → promote+visualize；近失 → NO_SIGNAL / 停精度 |
| P4 | 护栏 | accuracy / chain；**勿默认** `test_perf` 覆写 formal |
| P5 | 材料 | checklist · Agent 日志段 · **唯一**评测报告换戳 · `maintain check` |
| P6 | 合入 | sync `ai4s`；**有 promote** 再 pack |

## 命令

```bash
cd /workspace/ai4s-f/submission
python3 skills/operator_opt_loop/run_loop.py --dry-run
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict   # 硬门禁失败 exit 1
python3 skills/operator_opt_loop/run_loop.py --json-only          # 仅 JSON
# 可选占卡：
python3 skills/operator_opt_loop/run_loop.py --run-accuracy
```

快照：`results/run_logs/operator_opt_loop_last.json`  
字段：`required_assets` / `consistency` / `process_sop` / `precision_posture` / `probe_cmd` / `pass`

## 精度线姿态

- ROUND7 后默认 `stopped_*`：同构 sched deepen **不再默认开**。  
- 仅当有**新机制**（非同构）时复制 JSON 内 `probe_cmd`，仍守 P2。  
- 未破 gate 的探针**不得**进入评测报告正式 `v` 号。

## 红线

1. SOL / proxy / tune median ≠ 得分句  
2. formal Spectral ms 冻结  
3. 禁长 `AwaitShell` 等训练  
4. 单卡串行  
5. 评测报告全局只留一份  

## 与官方资产

材料轮对照：`SUBMISSION_CHECKLIST.md` + `OFFICIAL_ASSET_ALIGNMENT_*.md`（根 `AGENTS.md`「官方提交物与资产对照」）。
