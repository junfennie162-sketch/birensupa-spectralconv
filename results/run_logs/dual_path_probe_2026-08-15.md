# 双路径探针（2026-08-15 晚）

> 旁注。**未写** `summary.json` / 正式 idle。算子协议：正确性三案 + modes=16 + warmup=10/iters=100。

## 基线（改代码前）

| 项 | 64 | 128 | 256 |
|----|---:|----:|----:|
| CPU 入 ms | 0.972 | 2.326 | 7.839 |
| 设备常驻 ms | 0.569 | 0.862 | 2.333 |
| FNO B16 | 12.24 ms/batch · **5.35M** gps · chain rel 7.5e-5 PASS |

## 试过什么

| 尝试 | 结果 |
|------|------|
| 跳过 suFFT 的 host-origin 回拷（`SPECTRAL_PRUNED_SKIP_ROUNDTRIP`） | FNO rel≈0.5 **No-Go**。GELU/add/einsum/Conv/IN 写出的 device tensor，裁剪 kernel 读不对；`synchronize` / `clone` 也不行。必须 `copy_` 进 host-seeded 缓冲。 |
| C++ `get_stage_buffer` 复用 irfft 输出并直接返回 | FNO rel≈0.3 **No-Go**（和下层 add/IN 别名） |
| CPU 入：输入走缓存 H2D + irfft 写进 Python 缓存、只在 `to_cpu=True` 用 | 三案 / modes16 / FNO 链 **PASS** |

## 当前 KEEP（非正式）

| 路径 | 64 ms | 128 ms | 256 ms |
|------|------:|-------:|-------:|
| CPU 入 | **0.961** | **2.207** | **7.870** |
| 设备常驻 | 0.569 | 0.753 | 2.348 |
| FNO B16 | 12.45 ms · **5.26M** gps · rel 7.5e-5 | | |

相对正式 idle 3.797 / 8.037 / 29.295：约 **75% / 73% / 73%**（未 promote）。

FNO 吞吐相对 summary 旁注 1.60M：裁剪 DFT 已在链上，约 **3.3×**（未改 L2，未写 summary）。
