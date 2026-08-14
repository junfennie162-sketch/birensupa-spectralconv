# Handoff · 新 Session（2026-08-14 · 现行）

> **现行交接稿。**  
> [`HANDOFF_NEW_SESSION_2026-08-06.md`](HANDOFF_NEW_SESSION_2026-08-06.md) 及更早为 **历史**。  
> 数字以 [`CURRENT.md`](CURRENT.md) + [`summary.json`](../summary.json) 为准。

---

## 可复制 Prompt

```text
你是接管 ai4s-f（队长侧）的 Agent。只写 /workspace/ai4s-f；稳定后 sync /workspace/ai4s/submission；禁止改 ai4s-n。单卡 GPU，禁止 f/n 并发；禁止默认重跑 test_perf 覆写 formal ms。

## 先读
1. /workspace/AGENTS.md · /workspace/ai4s-f/AGENTS.md
2. submission/FILE_CONVENTIONS.md · LAYOUT.md
3. submission/results/run_logs/CURRENT.md
4. submission/results/summary.json
5. /workspace/评测报告_最新指标_*.md（须全局唯一）

## 现行主报
- 公开 NS64 L2 = 0.035012 · spec_ref_r2 · v10
- Spectral idle = 3.797 / 8.037 / 29.295 ms（2026-08-14 复测）
- Phase = submit_gate done
- 精度姿态 = v10 后默认可停；仅用户 Go + 新机制才短探针；gate = 0.034912
```

## 现行主报与姿态

| 项 | 值 |
|----|-----|
| 公开 NS64 L2 | **0.035011906177** · `spec_ref_r2` · **v10** |
| Spectral idle | **3.797 / 8.037 / 29.295 ms** |
| Checkpoint | `fno_ns/checkpoints/fno_ns_public_demo.pt` |
| Phase | `submit_gate` **done** |
| 提交包 | `fandougarden_submit_20260811_103945.tar.gz`（不含 08-14 复测文档） |
| Agent | `AGENT_OFFICIAL.md` + `development_log.md` |

## 红线

- 禁止改 `ai4s-n`
- 禁止 SOL/proxy 当正式得分句
- 禁止未破 gate 编 v 号；禁止自动 promote
- 历史文件在 `run_logs/_history/`，勿当任务单
