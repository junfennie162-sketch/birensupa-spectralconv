# Δ-match 短探针计划（2026-08-06）

> 用户口头 Go · 选项 2：新机制 + 预注册 gate + `--stop-on-gate` + 禁自动 promote

## 机制（相对已封存项正交）

| 禁重开 | 本探针 |
|--------|--------|
| PF / STLW / hard_reweight / modes20 / width48 / 同构 freeze deepen | **Δ-match**：`L2 + λ_δ·L2(Δpred, Δgt)`，直打 Autopsy `q_t` 相关 |

脚本：`fno_ns/train_public_delta_match_probe.py` · tag `delta_match_r1`

## 预注册

| 项 | 值 |
|----|-----|
| init | `fno_ns_public_demo.pt`（freeze_r11 · **0.035223**） |
| gate | **0.03512327**（baseline − 1e−4） |
| epochs | ≤4 · early-stop patience=2 · rebound abort +5e−4 |
| λ_δ / hf / lr | 0.5 / 0.15 / 5e−6 |
| freeze spectral | 是 |
| promote | **关**；破 gate 仅「eligible」，须人工确认 |

## 裁决分支

- **A SIGNAL**（best < gate）→ 停训汇报；待你确认后再 `promote_public_ckpt.py`
- **B 近失 / 弱升未破 gate** → INCUBATE 或 NO_SIGNAL；主报不动；精度再停
- **C 变差 / rebound** → ABORT；不覆盖 demo
