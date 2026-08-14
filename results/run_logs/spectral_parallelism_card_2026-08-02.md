# Spectral 计算并行度叙事卡（旁注 · 2026-08-02）

> 对齐赛题加分措辞「计算并行度提升」。**不写** formal ms。

| 已落地手段 | 评分词对齐 | 效果 |
|------------|------------|------|
| dual_scatter / dual_out | 合并 launch / 写回 | 双角零填+scatter 合一 |
| gather-scatter packed trunc | 减少无效并行元 | 只搬低频有效区 |
| float2 + `#pragma unroll` | SIMT/ILP | mul kernel 微优化 |
| `SPECTRAL_MUL_BLOCK` | 占用率调参 | block 级并行粒度 |

**金句**：mul 已是噪声级；并行收益主要在 launch 合并与稀疏写回，墙仍在 C2R。
