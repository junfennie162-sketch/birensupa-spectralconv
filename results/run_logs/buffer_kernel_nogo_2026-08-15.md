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

## 还没做（若继续）

256 墙仍是 **irfft ~0.95 ms + H2D/D2H ~5.5 ms**。混合基 ifft_h 与 vec4 irfft 的 **smem 平面融合**（不是 Goertzel 融合逆）尚未单独做过。
