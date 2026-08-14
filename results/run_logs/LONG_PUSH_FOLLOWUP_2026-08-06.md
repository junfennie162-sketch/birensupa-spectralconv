# Long push 结果（2026-08-06）

## Wave-1（已结束 · NO_SIGNAL）

| 阶段 | best L2 | 备注 |
|------|---------|------|
| qt_over_r1 | ~0.03518（见 summary） | q_t 过采样 |
| last_thaw_r1 | **0.035165** | 末层解冻 |
| dualview_r1 | **0.035160** | 链内最优 sidecar |
| soup_long | 0.035162 | 未超 dualview |

相对 demo **0.035223**：Δ≈**+6.3e−5**；gate **0.035123** 仍差 ≈**3.7e−5**。  
主报 demo **未 promote**。

## Wave-2（已结束 · **SIGNAL**）

| 阶段 | best | 备注 |
|------|------|------|
| last_thaw_r2 (k=2) | **0.035116** | 首破 gate · stop_on_gate |
| qt_over_r2 | 未再升 | 保持 thaw |
| dualview_r2 | **0.035115** | 链最优 · 复评一致 |
| soup_wave2 | 0.035154 | 稀释，忽略 |

正式裁决卡：[`LONG_PUSH_SIGNAL_2026-08-06.md`](LONG_PUSH_SIGNAL_2026-08-06.md)  
**未自动 promote** —— 等人口头 Go。
