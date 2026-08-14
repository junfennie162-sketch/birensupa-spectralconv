# Handoff · 新 Session（2026-08-04 · 现行）

> **现行交接稿。**  
> 旧稿 [`HANDOFF_NEW_SESSION_2026-08-02.md`](HANDOFF_NEW_SESSION_2026-08-02.md)（及 07-31）为 **历史**，勿当任务单。  
> 数字 / 姿态以 [`CURRENT.md`](CURRENT.md) + [`summary.json`](../summary.json) 为准。

---

## 可复制 Prompt（粘贴给新 Agent）

```text
你是接管 ai4s-f（队长侧）的 Agent。只写 /workspace/ai4s-f；稳定后 sync /workspace/ai4s/submission；禁止改 ai4s-n。单卡 GPU，禁止 f/n 并发；禁止默认重跑 test_perf 覆写 formal ms。

## 先读（门禁，按序）
1. /workspace/AGENTS.md · /workspace/ai4s-f/AGENTS.md
2. submission/FILE_CONVENTIONS.md
3. submission/results/run_logs/CURRENT.md          ← 姿态真源
4. submission/results/run_logs/HANDOFF_NEW_SESSION_2026-08-04.md  ← 本文件
5. submission/results/summary.json（public_ns64 + spectral_conv.perf）
6. submission/README.md（交卷说明）· 仓库根 README.md（导航）
7. skills/operator_opt_loop/LOOP_PROCESS.md
8. 评测报告：/workspace/评测报告_最新指标_*.md（须全局唯一）

## 现行主报（勿改口径除非破 gate + 人工确认）
- 公开 NS64 L2 = 0.035302 · freeze_r9 · 评测报告 v8
- Spectral idle = 3.811 / 8.054 / 29.560 ms（冻结）
- Phase = submit_gate done
- 精度姿态 = 永久停训（勿再开同构抛光 / STLW / 扩 modes）

## 本轮已做完（勿重复空转）
- 回滚 v8 + promote_guard；材料/JUDGE/PPT/demo 去噪
- Error Autopsy D（epochs=0）→ CONDITIONAL_PF_ALLOWED
- pf_clean_r1 clean-anchor PF → best 0.035216 · NO_SIGNAL · 未 promote
- README：submission 详细版 + 根导航（折叠默认收起）

## 默认任务（按优先级）
1. 答辩演练材料：JUDGE 一页包 + Autopsy 三图 + PPT 冻结稿
2. 资产自检：run_loop --dry-run --strict · maintain_assets.sh check submit_gate
3. 需要时 sync 稳定文件到 /workspace/ai4s/submission（勿把实验垃圾堆进合并主线）
4. 精度线：默认不开训。仅当用户口头 Go 且有新机制时，才 nohup + --stop-on-gate（epochs≤4）

## 红线
- 未破 gate(0.035202) 禁止编评测报告新 v / 禁止自动 promote
- 禁止 SOL/proxy/tune 当正式得分句
- 禁止长 AwaitShell 挂训练；秒查日志即可
- 脏树里 archives 解压目录 / figures 重复 PNG 勿盲目 git add .
```

---

## 1. 工作区边界

| 路径 | 角色 | Agent 行为 |
|------|------|------------|
| `/workspace/ai4s-f/` | 队长侧（本仓库） | **默认开发区**；业务只写其 `submission/` |
| `/workspace/ai4s-n/` | 搭档侧 | **禁止修改**（除非用户明确授权） |
| `/workspace/ai4s/` | 合并主线 | 稳定成果再 sync；勿堆个人实验 |
| `/workspace/赛题文档/` | 手册 | 只读参考 |
| GitHub | [Aafff623/fandou-ai4s](https://github.com/Aafff623/fandou-ai4s) | `main` 已推送；根 README ≠ 提交 README |

---

## 2. 现行主报与姿态

| 项 | 值 |
|----|-----|
| 公开 NS64 L2 | **0.03530218452215195** · `freeze_r9` · **v8** |
| Checkpoint | `fno_ns/checkpoints/fno_ns_public_demo.pt`（与 `freeze_r9_best` 同源口径） |
| Spectral idle | **3.811 / 8.054 / 29.560 ms**（冻结） |
| 正确性 worst rel | ≈ 2.17e−7 |
| Phase | `submit_gate` **done** |
| 精度 | **永久停** · PF 近失 0.035216（gate 0.035202）· INCUBATE 旁注 · **不编 v9** |
| Agent 日志 | `development_log.md` 记录 **1–38**（精品含 35 回滚 / 36 材料 / 37 Autopsy / 38 PF） |

评测报告：`/workspace/评测报告_最新指标_2026-08-04_155200.md`（§1 持平 **0%**；§2.1 含 Autopsy/PF；无新 promote）。

---

## 3. 本会话已落地（2026-08-04 摘要）

按时间粗序，便于新会话避免重复劳动：

1. **回滚 V8**：停 R11；`freeze_r10` 弱 0.035252 demote；`promote_guard.py`；3D 四角合入  
2. **材料闭环**：JUDGE 一页包 · PPT 冻结 · 六轴口播 · demo 旧图归档  
3. **瓶颈简报 → Grok/GPT 共识** → Autopsy D（ρ(e1,g)≈0.80）→ `CONDITIONAL_PF_ALLOWED`  
4. **人口头 Go · PF**：`train_public_pushforward_probe.py` · tag `pf_clean_r1` · best **0.035216** · **NO_SIGNAL**  
5. **文档**：`submission/README.md` 详细折叠版；根 `README.md` 导航（五节**默认折叠**）；git commit + push 至 `main`

关键产物路径：

| 产物 | 路径 |
|------|------|
| Autopsy 明细 | `results/run_logs/error_autopsy_20260804/` |
| Autopsy 裁决卡 | `ERROR_AUTOPSY_VERDICT_2026-08-04.md` |
| PF 后续 / 结果 | `PF_FOLLOWUP_2026-08-04.md` · `fno_public_boost_pf_clean_r1_summary.json` |
| PF ckpt（本地，gitignore） | `fno_ns/checkpoints/fno_ns_public_pf_clean_r1_best.pt` |
| 共识 / 简报 | `GPT_GROK_CONSENSUS_*` · `GPT_BRIEF_*` |

---

## 4. 建议下一步（默认答辩，不开精度）

### P0（答辩）

1. 按 [`JUDGE_3MIN_PACK_2026-08-04.md`](JUDGE_3MIN_PACK_2026-08-04.md) 演练 3 分钟  
2. 口播串：主报数字 → Spectral 三档 + C2R 墙 → 流场图 → Autopsy「0.0128 不可比」→ 失败诚实（PF 近失）  
3. 确认 `demo/media/README.md` 现行图列表干净  

### P1（工程卫生，按需）

1. `run_loop.py --dry-run --strict` · `maintain_assets.sh check submit_gate`  
2. 稳定文件 sync `/workspace/ai4s/submission/`（**不要**把 `archives/` 解压树、重复 figures 一把加进 git）  
3. 若打包：有 promote 再 pack；当前无新 v，可不重打 tar  

### P2（仅用户口头 Go）

- 精度线默认 **停**。若再开，须新机制 + gate + `--stop-on-gate`；Autopsy 已授权的 PF 探针**已用尽**（近失）。  
- 勿重开：STLW 短探针、modes20/width48、hard_reweight、解冻 formal ms。

---

## 5. 脏树与 git 注意

远端 `main` 已含：回滚/材料/Autopsy/PF 脚本与摘要、两份 README 润色等。

本地常见**未提交**残留（勿盲目 `git add .`）：

- `submission/results/archives/fandougarden_submit_*/` 解压目录与部分 `.sha256`
- `submission/results/figures/*.png`（与 `demo/media` 重复；gitignore 有例外规则时可能显示 untracked）
- ckpt 目录整体在 `.gitignore`

提交前：点名 `git add`；对照 `.gitignore`。

---

## 6. 自检命令

```bash
cd /workspace/ai4s-f/submission
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2

python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
./scripts/maintain_assets.sh check submit_gate

# 读姿态
head -n 40 results/run_logs/CURRENT.md
python3 - <<'PY'
import json
from pathlib import Path
s=json.loads(Path("results/summary.json").read_text())
print(s["fno_ns"]["public_ns64"])
print(s["spectral_conv"]["perf"]["rows"])
PY
```

---

## 7. 历史 Handoff

| 文件 | 状态 |
|------|------|
| [`HANDOFF_NEW_SESSION_2026-08-04.md`](HANDOFF_NEW_SESSION_2026-08-04.md) | **现行** |
| [`HANDOFF_NEW_SESSION_2026-08-02.md`](HANDOFF_NEW_SESSION_2026-08-02.md) | 历史（仍写 v7/0.035725） |
| [`HANDOFF_NEW_SESSION_2026-07-31.md`](HANDOFF_NEW_SESSION_2026-07-31.md) | 历史（boost/squeeze） |
