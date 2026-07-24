# Auto-Tune run log — 2026-07-23

Tool: `spectral_conv_combo/tune.py --quick --shape 64 128`
Settings: warmup=3, iters=3 (smoke sweep).

## Resolutions scanned

| min(H,W) | chosen path | buffer_max | fused_block | forward_ms | peak_mb |
|----------|-------------|-----------:|------------:|-----------:|--------:|
| 64       | fused       | 2          | 256         | 85.62      | 41.7    |
| 128      | fused       | 4          | 256         | 77.608     | 137.9   |

## Hand-tuned baseline (today's earlier `test_perf.py`)

| min(H,W) | path (hand-tuned) | forward_ms | peak_mb |
|----------|-------------------|-----------:|--------:|
| 64       | v1 (auto)         | 10.15      | 9.5     |
| 128      | fused (auto)      | 13.80      | 137.9   |
| 256      | fused (auto)      | 52.54      | 522.2   |

## Notes

- `tune.py --quick` 跑得**比 test_perf 慢**（85 vs 10 ms at 64），是因为扫
  描每次都要 reset SUPA 缓存、冷启 `_OUT_FREQ_CACHE`，且 3 iters 统计噪声
  大。tune 的目的是**选配置**，不是测极限时延；正式 forward 走 `test_perf.py`。
- 64 选 fused 是因为扫描里 fused buf=2 block=256 赢了 Pareto（41.7 MB 比
  v1 的 ~10 MB 高，但 forward 接近）。若想强制 64 走 v1，可加：
  ```python
  ops._AUTO_TUNE_TABLE[64] = {"use_sufft": False, "buffer_max": 4}
  ```
- 128/256 默认 fused，与手调一致。