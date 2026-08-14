# Skill · operator_opt_loop

## 名称

`operator_opt_loop`

## 目标

一键回放 **规范 OPT 闭环**：读主报 → 资产/一致性门禁 → P0–P6 流程清单 → 建议护栏/探针命令。  
服务 Agent/Skills 评审抽查与队内轮次纪律。  
**默认 dry-run**：不重训、不写 formal perf。

流程全文：[`LOOP_PROCESS.md`](LOOP_PROCESS.md)

## 输入

- 工作根：`submission/`
- `results/summary.json`、`results/phase_status.json`、`results/run_logs/`
- 官方资产：`skill.md`、`SUBMISSION_CHECKLIST.md`、`development_log.md`、`demo/media/brsmi_snapshot.txt`

## 步骤（dry-run）

```bash
cd /workspace/ai4s-f/submission
python3 skills/operator_opt_loop/run_loop.py --dry-run
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
```

脚本将：

1. 打印公开 L2 / tag / baseline→gate（只读 summary）  
2. 核对 formal idle 三档是否仍贴冻结板  
3. 检查官方必须资产 + 叙事产物 + 唯一评测报告  
4. 打印 P0–P6 SOP、红线、建议护栏；精度线默认停时只给条件探针命令  
5. 写 `results/run_logs/operator_opt_loop_last.json`；`--strict` 时硬门禁失败 exit 1

## 可选实跑档（占卡，需显式打开）

```bash
python3 skills/operator_opt_loop/run_loop.py --run-accuracy   # 仅 test_accuracy，不写 perf
```

## 输出

- 终端清单 + `operator_opt_loop_last.json`（含 `pass` / `process_sop` / `probe_cmd`）

## 边界

- 禁止把 SOL proxy 写成得分句  
- 禁止在非 idle / 争用时写 `spectral_conv.perf`  
- 主报 L2/ms 以 `summary.json` 为准，本 Skill **不 promote**  
- 精度探针须 `nohup` + `--stop-on-gate`；禁止长 Await 挂起
