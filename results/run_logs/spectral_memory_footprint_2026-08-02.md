# Spectral 显存 footprint 对照卡（旁注 · 2026-08-02）

> Formal 峰值：**225.3 / 253.3 / 353.3 MB** @64/128/256。不写 formal 主表新数。

| 概念 | 说明 |
|------|------|
| 全频谱缓冲 | ≈ B×C×H×(W/2+1)×8 B（complex64）量级 |
| trunc/pack 后 | 宽度缩到 `modes2`，有效频谱体积显著下降 |
| `_OUT_FREQ_CACHE` | 复用 out_freq，避免每 call 分配抬峰 |
| host staging | D2H 复用；返回 buffer 本身避免 clone 双峰 |

手段细节见 `spectral_memory_story_2026-08-02.md`。
