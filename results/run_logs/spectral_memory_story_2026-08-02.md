# Spectral 显存演进故事（旁注 · 2026-08-02）

> Formal 峰值以 idle `test_perf` 为准：**225.3 / 253.3 / 353.3 MB**（64/128/256）。  
> 历史数字口径可能略有脚本差异，标注「同脚本峰值定义」时再横向比。

## 现行 formal 峰值

| 分辨率 | peak MB | formal ms |
|--------|--------:|----------:|
| 64×64 | 225.3 | 3.811 |
| 128×128 | 253.3 | 8.054 |
| 256×256 | 353.3 | 29.560 |

## 手段对照（工程优化，非「假减半」口号）

| 手段 | 作用 | 证据 |
|------|------|------|
| packed trunc（P4/P8b） | 频域宽度缩到 `modes2`，减少 pad/拷贝 | `opt_p4_*` / `opt_p7_p8_*` |
| `_OUT_FREQ_CACHE` | 复用 out_freq buffer，避免每 call 分配 | `spectral_conv_ops.py` |
| host staging / `_HOST_OUT_CACHE` | CPU 出缓冲复用；返回 buffer 本身避免 clone | skill.md §峰值显存 |
| dual_scatter | 零填充 + 双角 scatter 合一 launch | `opt_r8_dual_scatter_*` |
| buffer_max 上限 | 防极端 shape 切换撑爆 cache | tune / ops |

## 与历史对照（旁注）

- 早期/对照路径在 256 上曾出现更高峰值（日志中可见数百 MB 级）；现行 formal 已落到 **353.3 MB**。  
- microbench 单路径峰值可能更低，**不得**用其替换 formal 主表。  
- 显存故事服务评分「性能维·显存」观感，不作为解冻 ms 的借口。
