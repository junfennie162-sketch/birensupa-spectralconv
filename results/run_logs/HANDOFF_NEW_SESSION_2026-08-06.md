# Handoff · 新 Session（2026-08-06 · 现行）

> **现行交接稿。**  
> 旧稿 [`HANDOFF_NEW_SESSION_2026-08-04.md`](HANDOFF_NEW_SESSION_2026-08-04.md)（及更早）为 **历史**，勿当任务单。  
> 数字 / 姿态以 [`CURRENT.md`](CURRENT.md) + [`summary.json`](../summary.json) 为准。

---

## 可复制 Prompt（粘贴给新 Agent）

```text
你是接管 ai4s-f（队长侧）的 Agent。只写 /workspace/ai4s-f；稳定后 sync /workspace/ai4s/submission；禁止改 ai4s-n。单卡 GPU，禁止 f/n 并发；禁止默认重跑 test_perf 覆写 formal ms。

## 先读（门禁，按序）
1. /workspace/AGENTS.md · /workspace/ai4s-f/AGENTS.md
2. submission/FILE_CONVENTIONS.md
3. submission/results/run_logs/CURRENT.md
4. submission/results/run_logs/HANDOFF_NEW_SESSION_2026-08-06.md
5. submission/results/summary.json（public_ns64 + spectral_conv.perf）
6. submission/README.md · 仓库根 README.md
7. skills/operator_opt_loop/LOOP_PROCESS.md
8. 评测报告：/workspace/评测报告_最新指标_*.md（须全局唯一）

## 现行主报
- 公开 NS64 L2 = 0.035115 · dualview_r2 · 评测报告 v9
- Spectral idle = 3.811 / 8.054 / 29.560 ms（冻结）
- Phase = submit_gate done
- 精度姿态 = promote 后默认可停；仅用户 Go + 新机制才短探针
- 已合入 /workspace/ai4s/submission/；包 fandougarden_submit_20260806_181107.tar.gz

## 默认任务
1. 答辩演练：JUDGE + Autopsy 三图 + PPT（数字跟 v9）
2. run_loop --dry-run --strict · maintain check submit_gate
3. 需要时再 sync / pack；勿堆实验垃圾进 ai4s
4. 精度：默认不开；gate = baseline−1e−4；禁自动 promote

## 红线
- 禁止 SOL/proxy/tune 当正式得分句
- 禁止默认 test_perf 覆写 formal ms
- 禁止长 AwaitShell 挂训练
- 脏树 archives 解压目录 / 重复 figures 勿盲目 git add .
```

---

## 1. 工作区边界

| 路径 | 角色 | Agent 行为 |
|------|------|------------|
| `/workspace/ai4s-f/` | 队长侧 | **默认开发区** |
| `/workspace/ai4s-n/` | 搭档侧 | **禁止修改**（除非授权） |
| `/workspace/ai4s/` | 合并主线 | 已 sync v9；勿堆实验 |
| `/workspace/赛题文档/` | 手册 | 只读 |

---

## 2. 现行主报与姿态

| 项 | 值 |
|----|-----|
| 公开 NS64 L2 | **0.03511497611179948** · `dualview_r2` · **v9** |
| Checkpoint | `fno_ns/checkpoints/fno_ns_public_demo.pt` |
| Spectral idle | **3.811 / 8.054 / 29.560 ms**（冻结） |
| Phase | `submit_gate` **done** |
| 精度 | promote 完成；默认可停再冲 |
| 提交包 | `fandougarden_submit_20260806_181107.tar.gz`（≈2.7G · sha256 见旁文件） |
| Agent 日志 | 记录 **1–40**（39 promote / 40 合入 pack） |

评测报告：`/workspace/评测报告_最新指标_2026-08-06_174400.md`（相对 v8 **+0.53%**）。

---

## 3. 08-06 已落地（摘要）

1. long_push：qt 过采样 → 末层解冻 → 双视图 → wave2 **last_thaw_r2 / dualview_r2**  
2. 复评 SIGNAL → 人口头 **promote** → demo/summary/meta = dualview_r2  
3. 评测报告编 **v9**；JUDGE/PPT/checklist/披露/matrix 刷数  
4. `pack_submission.sh` → tar + **sync ai4s**  
5. 自检：`run_loop --strict` · `maintain check` **PASS**（2026-08-07 复核）

关键路径：

| 产物 | 路径 |
|------|------|
| SIGNAL 卡 | `LONG_PUSH_SIGNAL_2026-08-06.md` |
| 长链摘要 | `fno_public_long_push_wave2_summary.json` |
| 脚本 | `train_public_{qt_oversample,last_thaw,dualview,pf_delta_hybrid,delta_match}_probe.py` |
| 备份 demo | `fno_ns_public_demo_pre_dualview_r2.pt` |

---

## 4. 建议下一步（默认答辩）

### P0

1. [`JUDGE_3MIN_PACK_2026-08-04.md`](JUDGE_3MIN_PACK_2026-08-04.md) 按 **0.035115 / v9** 口播  
2. Autopsy 三图仍用于「协议不可横比」叙事  

### P1

1. 官网上传 `fandougarden_submit_20260806_181107.tar.gz`（若赛方开放）  
2. 本地未提交改动较多：用户要求再 `git commit` / `push`  

### P2（仅口头 Go）

- 新机制短探针 + gate（现 baseline−1e−4 ≈ **0.035015**）  
- 勿默认解冻 formal ms / STLW / 盲扩 modes  

---

## 5. 脏树与 git

常见未提交：长链新脚本、v9 文档、archives 解压树、ckpt（gitignore）。  
合入主线已用 rsync，**不依赖**必须先 push GitHub。
