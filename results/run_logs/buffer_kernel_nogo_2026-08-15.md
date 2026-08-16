# 缓冲 / kernel 形状探针 No-Go（2026-08-15 晚）

> 旁注。**未写** `summary.json` / 正式 idle。热路径未改。

正式：公开 NS64 L2 **0.035012** · Spectral idle **3.797 / 8.037 / 29.295 ms**。

撤回后非正式 CPU 入约 **0.97 / 2.19 / 8.05 ms**（与 KEEP 抖动带一致）。

## 试过什么

| 尝试 | 正确性 | 性能 | 结论 |
|------|--------|------|------|
| FNO ping-pong 返回 irfft dest | chain rel **1.34 FAIL** | — | No-Go。与 C++ stage 别名同类：不能把复用缓冲交给 add/IN |
| CPU 入 pageable→pinned→H2D | 三案 PASS | **6.93 / 14.92 / 49.08 ms** | No-Go。多一次 CPU 拷贝，Biren pin 不赚 DMA |
| 256 irfft 双 n1 + float2 store（独立 TU） | modes=16 PASS | 隔离 irfft **3.46 vs 0.95 ms** | No-Go。寄存器压力；不要再试 dual-n1 |
| 256 ifft_h float2 写回 | PASS | 隔离 **0.31 vs ~0.29 ms** | No-Go。标量写已够 |
| 去掉 D2H 前 `synchronize` | PASS | 64 略快、256 略慢，抖动带 | 不换。保持 `to_cpu` 先 sync 再 copy |

## 仍不要试

- `SPECTRAL_PRUNED_SKIP_ROUNDTRIP` 给 FNO
- 把 C++ `get_stage_buffer` / Python 缓存 spatial 直接返回 FNO
- Biren `shfl` 做 bin 广播
- Goertzel 融合逆 `SPECTRAL_FUSED_INV256`
- 往 `pruned_irfft_w256_pair.su` 同一 TU 再塞 kernel
- 往 `pruned_rfft_w_fact.su` 再塞 32 KB sibling（会污染 KEEP 8-row）
- 256 rFFT 16 行 / 256 线程（独立 TU 也慢 42%）
- 64 档双流拆 B（切开更慢约 9%）
- 256 融合逆 **128 线程/块**（隔离 1.362 vs 1.334 ms）

## 2026-08-16 续

| 尝试 | 正确性 | 性能 | 结论 |
|------|--------|------|------|
| 256 融合逆 16 行 / 256 线程（新 TU） | modes=16 vs 两 kernel rel **0** | e2e 256 **7.10 vs 7.32 ms** | **KEEP** 默认 `SPECTRAL_FUSED_MIXED256`；主表仍 7.870 |
| 256 融合逆 8 行 / 128 线程 | rel 0 | 隔离 **1.362 vs 1.334** | No-Go（占用差） |
| CPU 入 device-allocated 缓冲 | rel 0 | 256 持平 | 默认关 |
| 64 融合逆 16 行 / 256 线程 | rel 0 | 隔离 **0.155 vs 0.33 ms**；e2e **0.79–0.84 vs 1.08** | **KEEP** 默认 `SPECTRAL_FUSED_MIXED64`；主表仍 0.961 |
| 128 融合逆 16 行 / 256 线程 | rel 0 | 隔离 0.382 vs 0.396；e2e **2.200 vs 2.204** | 代码留下，默认关 |
| 64 融合前向 16 行 / 256 线程 | rel 0 | 隔离 **0.094 vs 0.128 ms**；e2e 复测 **0.750 vs 0.789** | **KEEP** 默认 `SPECTRAL_FUSED_FWD64`；主表仍 0.961 |
| 128 融合前向 16 行 / 256 线程 | rel 0 | 隔离 **0.162 vs 0.221 ms**；e2e 复测 **2.10–2.14 vs 2.18** | **KEEP** 默认 `SPECTRAL_FUSED_FWD128`；主表仍 2.207 |
| C++ 一站式 rFFT+mul+iRFFT | rel 0 | 64 快 0.3–0.8%；128/256 抖动 | 代码留下，`SPECTRAL_FUSED_E2E` 默认关 |
| 128 融合逆（前向已融合后复测） | rel 0 | e2e **0.5–3.1%** 不稳 | 仍默认关 |
| 256 整平面融合前向 | 未跑 | brcc：**49152 > 32768** smem | **硬 No-Go**。Biren 单核 smem 上限 32 KB；`row[256][16]` 已满格 |
| 256 融合逆 4-n1 + 两轮 irfft | rel 0 | 隔离 **0.877 vs 1.291**；e2e **6.82–6.88 vs 7.10–7.16** | **KEEP** 默认 `SPECTRAL_FUSED_MIXED256_N4`；主表仍 7.870 |
| 256 n4 `rest` 除数 `/16`→`/8` | vs 2-n1 rel **0** | 隔离 **0.817 vs 1.039**；e2e 仍约 3–4% | 与 grid `B*C*8` 对齐；KEEP 仍开 |
| 128 融合逆 4-n1 + 两轮 irfft | rel 0 | 隔离 **0.248 vs 0.403**；e2e **1.89–1.93 vs 2.05–2.15**（约 8–11%） | **KEEP** 默认 `SPECTRAL_FUSED_MIXED128_N4`；主表仍 2.207 |
| 64 融合逆 8-n1 + 两轮 irfft | rel 0 | 隔离 0.155 vs 0.156；e2e 抖动带 | 代码留下，`SPECTRAL_FUSED_MIXED64_N8` 默认关 |
| 256 流式融合前向（8 行×32 波 / 一平面一块） | 行 rFFT vs torch **2.1e-7**；合成 packed 未对齐 | 网格 `B*C` vs KEEP `B*C*32`，并行被串掉 | **No-Go**。不要再试整平面或单 block 扫 256 行前向 |
| 128 融合逆 8-n1 + 四轮 irfft | rel 0 | 隔离 **0.258 vs 0.300**；e2e **0.6–3.0%** 不稳 | 代码留下，`SPECTRAL_FUSED_MIXED128_N8` 默认关 |
| 256 rFFT 16 行 / 256 线程（**同 TU** 32 KB） | KEEP packed rel **~0.99** | — | **正确性 No-Go**。brcc 同文件 32 KB sibling 污染 8-row |
| 256 rFFT 16 行 / 256 线程（**独立 TU**） | vs KEEP rel **0** | 隔离 **0.746 vs 0.525**（−42%）；e2e **7.61 vs 7.42** | **性能 No-Go**。`SPECTRAL_RFFT256_N16` 默认关 |
| C++ 一站式复测（融合 KEEP 后） | rel 0 | 三轮 64/128/256 约 **−0.8%** | 仍默认关 |
| 双流拆 B（64） | rel 0 | e2e **0.844 vs 0.774**（−9%） | **No-Go**。64 拷贝不够大，切开更亏 |
| 双流拆 B（128/256） | rel 0 | 128 **1.831 vs 2.021**（+9.4%）；256 **6.572 vs 7.530**（+12.7%） | **KEEP** 默认 `SPECTRAL_PIPE_B`；主表未改 |

256 拷贝仍是大头，但两段重叠后非正式 256 约 **6.57 ms**。
