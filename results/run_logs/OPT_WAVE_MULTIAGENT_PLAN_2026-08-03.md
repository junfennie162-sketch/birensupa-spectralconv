# OPT_WAVE · 五路并行交叉裁决 Plan（2026-08-03）

> 汇总协调 · 工作区仅 `/workspace/ai4s-f/submission/` · 勿改 `ai4s-n`  
> 现行指针 [`CURRENT.md`](CURRENT.md) · ROUND10 细卡 [`OPT_ROUND10_PLAN_2026-08-03.md`](OPT_ROUND10_PLAN_2026-08-03.md)  
> **本卡为执行真源**；与 ROUND10 冲突时以本卡「交叉消歧」为准。
>
> **收口修订（2026-08-04）**：soft_r10 后 autochain 以「优于 live」写入 `freeze_r10`/0.035252（未破 gate 0.035202）属违规半 promote → **已回滚 v8·freeze_r9·0.035302**；ROUND11（hybrid/modes20/freeze_r11）属 D9 KILL 路线 → **已停卡**；禁止再跑。默认 `maybe_promote` 须 gate + `ALLOW_AUTO_PROMOTE=1`。

| 项 | 值 |
|----|-----|
| 时间 | 2026-08-03 · 汇总协调 |
| 主报 | 公开 NS64 L2 **0.035302** · tag `freeze_r9` · 评测报告 **v8** |
| gate | **&lt; 0.035202**（baseline − 1e-4） |
| Spectral formal idle | **3.811 / 8.054 / 29.560 ms**（冻结；禁 `test_perf`） |
| Phase | `submit_gate` done |
| soft_r10 快照 | **DONE / NO_SIGNAL** · best=0.035302（=baseline）· early_stop ep3 · **停精度** |

### 五路 Agent 链接

| 路 | Agent id | 核心结论（吸收后） |
|----|----------|-------------------|
| 赛题评分轴 | [61e17423-0055-4c40-b7ff-73d86783e7da](61e17423-0055-4c40-b7ff-73d86783e7da) | 弹性在 FNO L2（需新机制）与 Spectral **叙事**；正确性再抠 No-Go；同构 freeze 低 ROI；三方向 A/B/C |
| Spectral 性能 | [2f168de9-e160-451f-b5af-3832e4483e18](2f168de9-e160-451f-b5af-3832e4483e18) | formal idle 冻结；墙在 C2R；解冻 formal No-Go；算力留给 FNO/材料 |
| FNO 精度 | [e869eb9f-761a-4e13-af68-d610072387f1](e869eb9f-761a-4e13-af68-d610072387f1) | 主报 0.035302；freeze_r10 NO_SIGNAL 0.035287；KEEP soft→裁决；KILL deepen/hybrid/soup/modes20/width48 |
| Agent 材料 | [6388a704-ebee-4c8a-943c-c84ff6ecf7be](6388a704-ebee-4c8a-943c-c84ff6ecf7be) | 硬门槛绿；入口仍漂 v7·0.035725；MAT-1/2/3 零 GPU 优先 |
| 扩展加分 | [7e236ac3-2474-48ea-993c-705df7d018ac](7e236ac3-2474-48ea-993c-705df7d018ac) | 扩展已做满；缺答辩闭环与六轴口径；SOL 禁得分句 |

### soft_r10 终态（2026-08-03 · 执行确认后）

| 探针 | 结果 |
|------|------|
| summary | `fno_public_sched_soft_r10_summary.json` |
| best / gate | 0.03530218 / 0.03520218 · `beat_gate=false` · Δ=0 |
| 停止 | early_stop ep3（test 变差） |
| 裁决 | **NO_SIGNAL → 停精度换轴**；**不开 D8**；**禁止 promote** |

---

## §1 交叉结论（共识 / 冲突消歧）

### 共识（五路咬合）

1. **主报冻结口径**：v8 / `freeze_r9` / 0.035302；未破 gate **不编新 v 号**、不擅自 promote。  
2. **Spectral formal ms 不挖**：墙在 C2R；mul 噪声级；解冻 formal = No-Go；算力与叙事转向「CPU 加速比 / 显存 / 融合诚实 / Agent 瓶颈」六轴。  
3. **同构精度路线已熄火**：freeze_r10 近失 NO_SIGNAL（0.035287）；KILL deepen / hybrid / soup / modes20 / width48。  
4. **材料与答辩闭环 ROI > 再烧同构精度**：入口漂移（README/PPT/对齐卡/FILE_CONVENTIONS 仍见 v7·0.035725）是硬伤；扩展「做满」≠「评委 3 分钟能闭环」。  
5. **GPU 纪律**：单卡串行；soft_r10 占卡期间只做零 GPU 材料；禁长 Await / 禁默认 `test_perf`。

### 冲突消歧（以本表为准）

| 冲突表象 | 消歧裁决 |
|----------|----------|
| 评分轴「A 精度换机制」vs 材料「零 GPU 优先」vs 精度「soft 后停」 | **并行分层**：soft_r10 **已在跑则不杀**；占卡期间 **P0=材料零 GPU**；soft 结束后按 §4 一次性裁决，默认停精度换轴（材料/叙事/吞吐） |
| ROUND10「R10-2 NO_SIGNAL→转性能叙事」vs 精度「还可 1 条 geom/部分解冻」 | **默认停**；仅当 soft **近失**（best∈[gate, gate+3e-5) 或明显优于 freeze_r10 且趋势未塌）才 **Conditional 开 1 条** geom **或** 部分解冻（二选一，禁并行） |
| Spectral「算力留给 FNO」vs 精度「停精度」 | soft 收口后 GPU 优先 **材料演示刷新 / FNO batch 吞吐旁注**，不是再开同构 train；**不是** Spectral ms |
| 扩展「性能六轴」vs Spectral「不挖 ms」 | 六轴 = **叙事与证据卡**（已有 fusion/memory/parallelism 卡），**禁止**为刷表解冻 formal |
| 「SIGNAL 即 promote」直觉 | **SIGNAL 只标注「待人工确认 promote」**；须人确认后才 visualize / 报告 v9 / pack |

### KEEP / KILL（精度侧继承）

| KEEP | KILL |
|------|------|
| 链：sched → soft → freeze（历史已 promote 点） | deepen / hybrid / soup / modes20 / width48 |
| 现行后备：`sched_soft_r10` 跑完裁决 | 解冻 Spectral formal / 重跑 `test_perf` |
| 条件：geom **或** 部分解冻（最多 1） | 同构 freeze 再 deepen |

---

## §2 可行性方向表

| ID | 方向 | 咬合分轴 | ROI | GPU | 风险 | Go/Conditional/No-Go | 优先级 |
|----|------|----------|-----|-----|------|----------------------|--------|
| D1 | MAT-1 入口真源对齐（README/PPT/对齐卡/FILE_CONVENTIONS/CURRENT→一律 v8·0.035302） | Agent15 + 搭建证据 | 极高 | 0 | 低（改错口径） | **Go** | **P0** |
| D2 | MAT-2 评委 3 分钟路径（README→demo→checklist→development_log→主报数字） | Agent15 + 可视化20 | 极高 | 0 | 低 | **Go** | **P0** |
| D3 | MAT-3 演示叙事冻结包（90s storyboard + brsmi + 指标快照与主报一致） | 可视化20 + Agent15 | 高 | 0 | 中（图旧） | **Go** | **P0** |
| D4 | EXT-1 扩展抽查闭环（showcase→命令可复现→勿写 SOL 得分句） | 扩展15 | 高 | 0 | 低 | **Go** | **P1** |
| D5 | SPEC-NARR 性能六轴叙事卡串（加速比/显存/融合诚实/瓶颈；formal ms 旁注冻结） | Spectral 性能25（叙事） | 高 | 0 | 中（夸大融合） | **Go** | **P1** |
| D6 | soft_r10 跑完裁决（见 §4）；SIGNAL→待人工 promote | FNO 精度25 | 中高 | 占用中 | 中（假 SIGNAL/过拟合） | **Go（等待）** | **P1** |
| D7 | FNO 吞吐旁注 + 可视化刷新（batch16 / pred-vs-gt 与 v8 ckpt 对齐） | FNO 性能10 + 可视化20 | 中 | 短测可选 | 中（占卡） | **Conditional**（soft 结束后） | **P2** |
| D8 | 精度末探针：geom **或** 部分解冻（≤1 条） | FNO 精度25 | 中低 | 中 | 高（再 NO_SIGNAL） | **Conditional**（§4 近失） | **P2** |
| D9 | 同构 deepen / soup / modes20 / width48 | FNO 精度25 | 低 | 高 | 高（已证伪） | **No-Go** | — |
| D10 | 解冻 Spectral formal / 挖 C2R ms | Spectral 性能25 | 低 | 高 | 极高（破冻结） | **No-Go** | — |
| D11 | 正确性阈值再抠 / 同构 freeze 堆叠 | Spectral 正确35 | 极低 | 中 | 高（No-Go 风险） | **No-Go** | — |

---

## §3 推荐执行序（按天 / 人时）

> 原则：**先材料后精度**；唯一例外 = soft_r10 **已在跑**（不杀、不等人肉盯 epoch）。

### Day 0（当下 · soft 占卡中 · ~2–4 人时 · **零 GPU**）

1. **D1** 全文检索 `0.035725` / `v7` / 旧 tag，入口文件改到 v8·0.035302·`freeze_r9`。  
2. **D2** 写/修「评委 3 分钟」路径（可嵌 README 顶栏表）。  
3. **D3** 核对 demo 媒体与 `metrics_snapshot` 数字 = 主报。  
4. **D4+D5** 扩展 showcase 抽查命令跑通（CPU/已有产物）；六轴叙事只链已有卡，不重测 formal。  
5. 秒查 soft：`tail -20 /tmp/fno_soft_r10.txt` + 找 `*soft_r10*summary.json`（禁长 Await）。

### Day 1（soft 结束后 · 分支）

| soft 结果 | 动作 |
|-----------|------|
| **NO_SIGNAL** / 未破 gate | §4 收口停精度；可选 **D7**（短测吞吐+刷图）；**不**默认开 D8 |
| **近失**（§4 定义） | 人工二选一开 **D8** 单探针；同时继续材料缺口 |
| **SIGNAL**（best&lt;gate） | plan/CURRENT 标「**待人工确认 promote**」→ **停手等人口头 Go** → 才 visualize / 报告 v9 / pack / sync |

### Day 2（答辩加固 · ~2 人时 · 默认零 GPU）

- development_log 补完整段（场景≥工具/瓶颈/可视化之一）。  
- `maintain_assets.sh check submit_gate` + `run_loop.py --dry-run --strict`。  
- 评测报告仅在 **正式 promote** 后换戳；否则 §1/§2 提升% 填 0% 持平。

### 人时粗算

| 块 | 人时 | GPU |
|----|------|-----|
| MAT-1/2/3 | 2–3h | 0 |
| EXT + SPEC-NARR | 1–2h | 0 |
| soft 等待 | 墙钟（已占用） | 1 卡 |
| D7 可选 | 0.5–1h | 短 |
| D8 条件 | 1–2h 墙钟 | 1 卡 |

---

## §4 ROUND10 收口规则（soft_r10 / 是否再开 geom）

### 现行状态

- R10-1 `freeze_r10`：**NO_SIGNAL** · best=**0.035287** · gate=0.035202 · Δ≈1.55e-5  
- R10-3 `sched_soft_r10`：**RUNNING**（汇总时）→ 结束后写入本卡/CURRENT 终态

### 裁决树（soft 结束后执行一次）

```
soft 结束
 ├─ best < gate  → SIGNAL → 标注「待人工确认 promote」；禁止自动 promote / 禁自动改 summary 主报
 ├─ gate ≤ best < gate+3e-5  或  best < freeze_r10.best 且 patience 未因发散停止
 │     → 近失 → Conditional：最多 1 条（geom XOR 部分解冻）；tag 新开；stop-on-gate
 │     → 该条仍 NO_SIGNAL → **永久停精度换轴**
 └─ 其余 NO_SIGNAL → **停精度**；转 D1–D5 / D7；禁止再开同构 soft/freeze deepen
```

### 硬规则

1. soft summary 未落盘前，不判 SIGNAL。  
2. SIGNAL ≠ promote；promote 仅人工确认后。  
3. geom 与部分解冻 **互斥**，总数 ≤1。  
4. 任何新探针未破 gate **不得**进入评测报告 v 号 / §9。  
5. 禁止并行第二 GPU 任务（含 ai4s-n）。

---

## §5 验收清单

| 项 | 命令 / 标准 | 通过条件 |
|----|-------------|----------|
| 资产 | `./scripts/maintain_assets.sh check submit_gate` | PASS |
| OPT Loop | `python3 skills/operator_opt_loop/run_loop.py --dry-run --strict` | PASS |
| CURRENT | 计划卡→本文件；精度姿态与 soft 终态一致 | 无旧 ROUND10「唯一真源」歧义 |
| 入口口径 | README/PPT/对齐卡/FILE_CONVENTIONS 无 v7·0.035725 主报残留 | 主报一律 0.035302 / v8 |
| 评测报告 | 无新 promote → 不换 v；有 promote（人工后）→ 新戳唯一稿 + §1/§9 | 全局只留一份最新稿 |
| soft 收口 | summary.json 存在且 beat_gate 字段已读 | 写入 CURRENT |
| **禁止项** | 长训加塞 / `test_perf` / 改 `ai4s-n` / 自动 promote / SOL 得分句 / 未破 gate 编 v 号 | 零违反 |

---

## §6 一句话决策

**soft_r10 占卡期间全力清入口漂移与答辩闭环（零 GPU）；跑完按门禁一次裁决——默认真停精度换材料/六轴叙事，仅近失才允许一条 geom/部分解冻，SIGNAL 只待人工 promote。**
