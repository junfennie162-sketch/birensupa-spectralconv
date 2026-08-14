# Handoff · 新 Session（2026-08-02）

> **历史** · 仍写 v7 / 0.035725 口径。  
> **现行交接**请用 [`HANDOFF_NEW_SESSION_2026-08-04.md`](HANDOFF_NEW_SESSION_2026-08-04.md) · 指针 [`CURRENT.md`](CURRENT.md)。  
> 更早：[`HANDOFF_NEW_SESSION_2026-07-31.md`](HANDOFF_NEW_SESSION_2026-07-31.md)（boost/squeeze 时代）。

## 可复制 Prompt

```text
你是接管 ai4s-f（队长侧）的 Agent。只写 /workspace/ai4s-f；稳定后 sync /workspace/ai4s/submission；禁止改 ai4s-n。单卡 GPU，禁止并发训；禁止默认重跑 test_perf 覆写 formal ms。

## 先读（门禁）
1. /workspace/AGENTS.md · /workspace/ai4s-f/AGENTS.md
2. submission/FILE_CONVENTIONS.md
3. submission/results/run_logs/CURRENT.md
4. summary.json（public_ns64 + spectral_conv.perf）
5. skills/operator_opt_loop/LOOP_PROCESS.md
6. 评测报告：/workspace/评测报告_最新指标_*.md（须唯一）

## 现行主报
- 公开 NS64 L2=0.035725（sched_samp_r5 · v7）
- Spectral idle 3.811/8.054/29.560 ms（冻结）
- 精度线默认停（ROUND7 NO_SIGNAL）

## 默认任务
- 材料规范 / 答辩口径 / Agent 日志；勿盲目开同构 sched deepen
- 自检：python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
- 有新机制再探针：nohup + --stop-on-gate；禁长 AwaitShell
```

指针真源：[`CURRENT.md`](CURRENT.md)
