# Skill · 优化闭环（operator_opt_loop）

## 这份说明给谁看

给**用 Cursor / Agent 一轮轮改算子和 FNO** 的竞赛队，以及要抽查「你们有没有过程纪律」的评审。也给工程负责人：防止正式性能表被一次脏测覆写、未过线的探针混进评测报告。

**目的**：把「读主报 → 探针 → 护栏 → 材料 → 合入」收成可回放的闭环；默认 **dry-run**，不重训、不写 formal perf。  
**价值**：做国产 GPU 性能迭代、或带 Agent 协作的人，能拿走单卡纪律、冻结口径、精度探针后台跑这些规则，减少「越优化越不可复现」。

流程条文：[`LOOP_PROCESS.md`](LOOP_PROCESS.md)。脚本：`run_loop.py`。

---

## 1. 技术背景

算子赛不是改一次就结束。我们在 BIREN 单卡上同时有：正确性门禁、三档 idle ms、FNO 公开集 L2、提交材料（`skill.md`、Agent 日志、运行 log）。Agent 若一边长等待训练、一边重跑 `test_perf.py`，主报会被噪声盖掉。

所以单独做一份 **OPT 闭环 Skill**：不负责发明新 kernel，负责**什么时候能改、什么数字能进主报、什么只能当旁注**。

主报只认 `results/summary.json` 里的公开 L2 和 Spectral idle。评测报告全局只留一份。未破 gate 的探针不得编正式版本号。

---

## 2. 优化思路（纪律本身就是在压榨官方 GPU）

卡只有一张。所谓优化思路，先是**别把 GPU 用废**，再是**别把数字用乱**。

| 思路 | 具体做法 | 为什么 |
|------|----------|--------|
| 单卡串行 | f 与 n 禁止同时跑 SUPA | 并发易 ErrorCode 719，测出来的 ms 无效 |
| 先读主报再动手 | P1 只认 summary 的 L2 / idle | 防止对着过期 Markdown 优化 |
| 精度探针短、可停 | `nohup` + `--stop-on-gate`；epoch 少、patience 小 | Agent 不能长 Await 挂死会话 |
| 过线才 promote | `best < gate` 才切正式 ckpt 和可视化 | 近失记 NO_SIGNAL，不进评测报告版本链 |
| formal ms 冻结 | 默认禁止 `test_perf.py` 覆写 `spectral_conv.perf` | 非空闲、有争用时数字不能进主表 |
| 材料闭环 | checklist、Agent 日志段、唯一评测报告换戳 | 官方必须项缺一项就不合格 |
| 旁注隔离 | SOL proxy、tune median、自建集 L2 | 评审问得分句时不会被带偏 |

P0–P6 一览（细节以 `LOOP_PROCESS.md` 为准）：

| 步 | 名称 | 必须遵守 |
|----|------|----------|
| P0 | 环境与单卡 | source SDK；禁止并发 GPU |
| P1 | 读主报 | 只认 summary 公开 L2 / idle ms |
| P2 | 精度探针 | nohup；`--stop-on-gate`；禁长 Await |
| P3 | gate 裁决 | 过线 promote；近失停精度 |
| P4 | 护栏 | accuracy / chain；勿默认写 perf |
| P5 | 材料 | 清单 + 日志 + 唯一评测报告 |
| P6 | 合入 | sync 主线；有 promote 再打包装 |

在 GPU 上真正压性能的手段（fused、缓存、残差训练）写在算子开发 / FNO 两份 Skill 里。本 Skill 保证那些手段**测得可复现、写得进主报**。

---

## 3. 技术与问题

**用到 / 学到的技术**

- 把过程编码成脚本：`run_loop.py --dry-run --strict` 读资产、对冻结板、失败 exit 1。
- 快照：`results/run_logs/operator_opt_loop_last.json`（`pass` / `process_sop` / `probe_cmd`）。
- Agent 日志要可打开：Markdown 时间线 + 截图，不只丢 JSONL。
- 「姿态」：精度线可以停；停了以后只允许新机制探针，不允许同构再刷。

**遇到的问题**

| 问题 | 学到什么 |
|------|----------|
| Agent 一轮改完直接重跑 test_perf | 会把冻结 idle 写成争用噪声 |
| 把 sidecar / 自建集写进 demo | 评审会对不上公开 NS64 |
| 评测报告堆很多历史版本号 | 主报只应对比上一正式版；未 promote 的探针不进版本链 |
| 训练挂在对话里等几小时 | 会话被掐、也占卡；必须后台 + 短查 |
| SOL / GB/s proxy 写进「我们赢了官方」 | proxy 不是官方 SOL-ExecBench |

**怎么回放（不占卡）**

```bash
cd /workspace/ai4s-f/submission
python3 skills/operator_opt_loop/run_loop.py --dry-run
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
```

可选占卡（仍不写 formal perf）：

```bash
python3 skills/operator_opt_loop/run_loop.py --run-accuracy
```

本 Skill **不 promote、不改主报数字**。主报以 `summary.json` 为准。

---

## 4. 红线（写进 Skill 就是为了以后还遵守）

1. SOL / proxy / tune median ≠ 得分句  
2. Spectral formal ms 冻结后，非空闲不写 perf  
3. 禁长 `AwaitShell` 等训练  
4. 单卡串行  
5. 评测报告全局只留一份  

若这份闭环说明对你有用，欢迎给仓库点 Star 收藏：https://github.com/junfennie162-sketch/birensupa-spectralconv
