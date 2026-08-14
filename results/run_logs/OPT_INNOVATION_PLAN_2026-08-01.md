# OPT_INNOVATION_PLAN · 2026-08-01

> **已被 Round-2 接棒**：现行行动总方针见 [`OPT_ROUND2_PLAN_2026-08-02.md`](OPT_ROUND2_PLAN_2026-08-02.md)。  
> 本文档保留为 Wave-0/1/2 历史执行副本；主报数字以 `summary.json` + Round-2 为准。  
> 地位（历史）：在 [`OPT_MASTER_PLAN_2026-07-31.md`](OPT_MASTER_PLAN_2026-07-31.md) **P3 收尾完成**之后的创新总方针。  
> 工作区：`/workspace/ai4s-f` · 合入：`/workspace/ai4s/submission` · **勿改 `ai4s-n`**

## 0. 基线与评分抓手

### 0.1 主报（滚动；Wave-2 后）

| 角色 | 指标 | 数值 |
|------|------|------|
| FNO 精度 | public NS64 rel-L2 | **0.036576**（`multistep_probe`；历史起点 sq3b=0.037520） |
| Spectral 性能 | idle 64/128/256 | **3.811 / 8.054 / 29.560 ms**（冻结） |
| Spectral 正确性 | worst rel | ~2.17e-7 |
| 相位 | submit_gate | done；Wave-0/1/2 done；Wave-3 skipped |

### 0.2 赛题权重（为什么换方向）

| 板块 | 维度 | 权重 | 现状判断 |
|------|------|------|----------|
| 必选 Spectral | 正确性 | 35% | 已极强 |
| 必选 | 性能 | 25% | ms 平台；靠**叙事/显存/工程**抬观感 |
| 必选 | Agent | 15% | 材料厚，缺可抽查索引 |
| 必选 | 扩展 | 15% | bwd/3D/irregular 已有；差闭环展示 |
| 进阶 FNO-C | 模型搭建 | 30% | 复用必选算子是加分轴 → **勿轻易换 F-FNO 主报** |
| 进阶 | 精度 | 25% | 0.037520 优秀带；再压边际递减 |
| 进阶 | 可视化 | 20% | 2026-08-01 已刷 public |
| 进阶 | Agent | 15% | 同必选 |
| 进阶 | 性能 | 10% | batch16/吞吐已有；旁注即可 |

**结论**：同构再 squeeze / 再抠 formal ms = **No-Go**。下一阶段以 **材料可证明深度 + 轻量精度探针 + 性能故事** 为主。

---

## 1. 四路评估汇总

| 方向 | Agent | Top-1 | 裁决 |
|------|-------|-------|------|
| A SUPA 可微训 | [SUPA可微训练](d9ee07dd-78e3-47d1-b296-42bc0597c347) | 材料化 CPU训→SUPA推→P4 bwd 闭环 | **Go（文档/短 demo）**；真长训 No-Go |
| B FNO 精度架构 | [FNO精度架构](095baa89-0aef-4413-998c-c347cf620e6a) | 多步 TF 辅助 + 轻量谱/能量 soft | **Go（先探针）**；期望 ΔL2≈0.001–0.003 |
| C 性能叙事 | [算子性能叙事](25ca4c45-870b-455b-97c6-12c6db843f4a) | fused 分段 profile + vs 官网 CPU 加速比故事 | **Go（高优先）** |
| D Agent/交付 | [Agent加分交付](b8bda474-7c4a-481d-8cc3-7d231ddb615e) | 日志场景索引 + 创新点/失败矩阵 | **Go（最高优先，零 GPU）** |

详细摘要：`INNOV_EVAL_{A,B,C,D}_*_2026-08-01.md`

---

## 2. 执行优先级（推荐序）

```text
Wave-0  零 GPU 材料（D + A5 + C 文案骨架）     ← 立即
Wave-1  性能故事证据（C：fused 分段短测旁注）   ← 半天，idle 单卡
Wave-2  精度探针（B1+B2，CPU，≤数小时）         ← 有信号才加长
Wave-3  条件触发（B3 蒸馏 / A4 吞吐旁注）       ← 仅当 Wave-2 无信号或评审要演示
Wave-X  永久 No-Go（见 §4）
```

### Wave-0 · 立即（0.5–1 人日，零 GPU）

| ID | 动作 | 验收 |
|----|------|------|
| D1 | `development_log.md` 顶部「场景对照索引」（≥3 类、8–12 精品段） | 评委可按 kernel/瓶颈/超参 跳转 |
| D2 | 创新点卡片 + `results/experiment_matrix.md`（KEEP/KILL/ABORT） | SCP/PPT 主报数字 = 0.037520 / 3.811 三档 |
| A5 | 一页「可微鸿沟闭环」：CPU 训路径 → SUPA 推理 → `test_backward` PASS | 不改主训；不写 formal ms |
| C0 | PPT/scp 预写加速比栏（vs official CPU ≈19.5×/11.1×/10.0×） | 标注「相对官网 CPU 参考脚本」，非竞品 GPU |
| D3 | 补 `skills/operator_opt_loop/SKILL.md` dry-run 剧本骨架 | `--dry-run` 默认；不占卡 |

### Wave-1 · 性能故事（0.5–1 人日）

| ID | 动作 | 验收 |
|----|------|------|
| C1 | fused 路径分段 profile → `run_logs/spectral_fused_segments_*.md` | **不写** `spectral_conv.perf`；C2R 墙与 OPT_MASTER 一致 |
| C2 | 刷新 `tune` / Skill 实验页口径到 3.811 板 | SOL 仅 proxy disclaimer |
| C3 | 可选：对照路径数字进 `optimization.*` 旁注 | 强标签「工程对照」 |

### Wave-2 · 精度探针（CPU，有信号才加码）

| ID | 动作 | 接受门槛 |
|----|------|----------|
| B1 | 多步 TF 辅助（T_out_train=2–3；eval 仍 step-1）从 public demo 热启，5–10 ep | test L2 **< 0.037520 − 1e-4** |
| B2 | 捆绑轻量能量/谱 soft（λ≪ hf） | 同 ep 优于纯 B1 |
| B+ | 门槛通过 → 短续训 → 仅公开 1000/128 严格更优才 promote | 同步 summary/results/disclosure/PPT/scp/visualize |

### Wave-3 · 条件触发

| ID | 触发条件 | 动作 |
|----|----------|------|
| B3 | B1/B2 无信号，且仍要搏精度 | width48 短训探针；ep15–30 轨迹不指向 <0.037 → **abort** |
| B4 | 仅对照 | F-FNO smoke；默认不 promote（损搭建衔接分） |
| A4 | 评审要「训练吞吐 SUPA」故事 | throughput 旁注字段；可差于 CPU，口径诚实 |
| A1 | 明确要求 e2e SUPA 训 demo | ≤数 epoch 微 batch；**禁止** promote / 写 formal |

---

## 3. KPI 与停条件

| KPI | 目标 | 停止 |
|-----|------|------|
| 材料可抽查 | Wave-0 全勾 | — |
| 性能叙事 | C1 落盘 + PPT/scp 一致 | 不追求 ms↓ |
| 公开 L2 | 有 promote 则 <0.037520；目标带 ~0.035–0.0365（非保证） | B1 探针无信号且 B3 abort → **停精度线** |
| Spectral formal | 保持冻结三档 | 任何回退 >3% 或正确性 FAIL → 回滚 |
| 口径 | 主报唯一；禁 0.005144/SOL 冒充 | grep 主材料 0 hit |

---

## 4. 全局 No-Go（继承并强化）

1. 同构再 squeeze / 解冻 Spectral formal ms  
2. SUPA fused 替换主训；`spectral_mul` 混合可微接长训热路径  
3. TTA 计入主报 L2；F-FNO/AFNO 替换必选算子定义进主报  
4. SOL-ExecBench / proxy 作得分句；伪官方 v2 L2  
5. Plan2d / `torch.fft@SUPA` / strided pack / NVIDIA 真融合热路径  
6. 与 `ai4s-n` 或长训并发写 GPU formal  

---

## 5. 执行状态（滚动）

| Wave | 状态 | 备注 |
|------|------|------|
| Wave-0 | **done** | 索引/matrix/skill loop/可微闭环 |
| Wave-1 | **done** | fused segments + vs CPU 加速比；formal ms 未改 |
| Wave-2 | **done / promoted** | multistep early-stop → L2 **0.036576** |
| Wave-3 | **skipped** | 见 `WAVE3_CONDITIONAL_SKIP_2026-08-01.md` |
| 评估 | **done** | 四路 agent 2026-08-01 |

---

## 6. 下一步（已接棒）

本 Plan 的 Wave-0/1/2 已完成；后续执行见 [`OPT_ROUND2_PLAN_2026-08-02.md`](OPT_ROUND2_PLAN_2026-08-02.md)（Wave-R0→R2）。  
合入纪律不变：稳定成果再 sync `/workspace/ai4s/submission/`；promote 才改主报数字。
