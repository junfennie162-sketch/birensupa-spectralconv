# 裁剪 DFT 样本探针（2026-08-15）

> 旁注。**未晋级。** 不写 formal idle ms。热路径仍是 fused suFFT。

计划：[`PRUNED_DFT_PROBE_PLAN_2026-08-15.md`](PRUNED_DFT_PROBE_PLAN_2026-08-15.md)

## S0–S2 小样本正确性（CPU）

门槛 1e-4。分解 DFT 两角 vs `rfft2` 切片；端到端 vs 官网双角 reference。

| 样本 | 角点 vs rfft2 | e2e scatter 逆 | e2e GEMM 逆 | 结论 |
|------|-------------:|---------------:|------------:|------|
| 8×8 modes=2 | 9.89e-7 | 8.06e-7 | 6.55e-7 | PASS |
| 32×32 modes=8 | 4.66e-6 | 3.43e-6 | 3.20e-6 | PASS |
| 64×64 modes=12（正确性档） | 8.28e-6 | 6.09e-6 | 6.20e-6 | PASS |

数学方向成立。误差大约 1e-6，比 fused 的 2e-7 松，仍过赛题线。

## S3 官方 64 正确性

`B=4, Cin=32, Cout=64, H=W=64, modes=16`

| 路径 | rel | ≤1e-4 |
|------|----:|:-----:|
| pruned（分解 DFT + scatter irfft2） | 6.045e-06 | PASS |
| fused 现行 | 2.551e-07 | PASS |
| 混合：trunc rfft + mul + scatter 逆 | 2.519e-07 | PASS |
| 混合：trunc rfft + mul + GEMM 逆 | 6.039e-06 | PASS |

## S4 计时（warmup=5, iters=20, 非正式）

| 路径 | ms | 相对 fused |
|------|---:|-----------|
| **fused 现行** | **3.797** | 基线（与冻结正式 64 档一致） |
| pruned CPU einsum | 49.541 | 慢约 **13×** |
| 混合 trunc + CPU scatter 逆 | 24.966 | 慢约 **6.6×** |
| 混合 trunc + GEMM 逆 | 29.806 | 慢约 **7.9×** |

## 裁决

**No-Go，不改默认热路径。**

对方那条路的「只算保留频点」在数学上可以接到咱们的双角合同上。第一枪用 einsum/GEMM 实现，FLOP 是 O(N·modes) 而不是 O(N log N)，64 档必然更慢——这和他们自己写的「首版标量裁剪 DFT 比 R2C 慢」一致。他们后来赢在 **shared-memory radix-2 + 融合逆**，不是这版 Python 分解 DFT。

下一步若继续：要写 SUPA 上的小 FFT / 行级 shared 逆，而不是再堆 einsum。128/256 本轮不做。
