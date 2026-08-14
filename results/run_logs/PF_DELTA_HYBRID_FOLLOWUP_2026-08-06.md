# PF+Δ hybrid 再冲结果（2026-08-06）

## Soup（step1）

| soup | L2 |
|------|-----|
| uniform_all | 0.035208 |
| **uniform_last2**（pf+delta） | **0.035203** |
| uniform_first_last | 0.035215 |

vs demo 0.035223 · gate 0.035123 → **NO_SIGNAL**（差≈8.0e−5）

## Hybrid `pf_delta_r1`（step2）

| 项 | 值 |
|----|-----|
| init | soup_near3（0.035203） |
| best | **0.035192** |
| gate | **0.035123** |
| Δ vs soup / vs demo | +1.2e−5 / +3.2e−5 |
| beat_gate | **false** · early_stop · **未 promote** |
| 裁决 | **NO_SIGNAL** |

主报继续 **0.035223 · freeze_r11**。旁注 INCUBATE：本地最优 sidecar ≈**0.035192**（hybrid）仍不进版本链。

精度姿态：**再停**。再同构组合期望 <1e−4 gate。
