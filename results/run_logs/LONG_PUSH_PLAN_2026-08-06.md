# Long push 长链（2026-08-06）

> 用户：「再优化，时间还长」· 三机制串行 + 终场 soup · gate=demo−1e−4 · 禁自动 promote

| 阶段 | 机制 | 相对已试项 |
|------|------|------------|
| 1 `qt_over_r1` | 按 q_t 加权过采样 + PF/Δ | ≠ hard_reweight（改采样非改损） |
| 2 `last_thaw_r1` | 只解冻最后 1 层 spectral | ≠ 扩 modes / 全解冻 |
| 3 `dualview_r1` | 双 roll 一致性 | 新正则 |
| 4 soup | 全 sidecar 再平均 | 收口 |

init 起点：`pf_delta_r1`（0.035192）· gate **0.035123**  
日志：`fno_public_long_push_chain_*.log` · 摘要：`fno_public_long_push_chain_summary.json`
