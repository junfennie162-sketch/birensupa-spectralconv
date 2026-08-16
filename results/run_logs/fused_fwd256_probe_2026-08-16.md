# 256 流式融合前向 No-Go（2026-08-16）

> 旁注。**未写** `summary.json`。`SPECTRAL_FUSED_FWD256` **未进热路径**（dispatch 已撤）。

整平面 `row[256][16]` 仍是硬 No-Go（32 KB 上限）。本轮改成 8 行一波 rFFT（与 KEEP 同构），高度样本放寄存器，不把 256 行频域平面塞进 smem。

| 项 | 结果 |
|----|------|
| 8 行 rFFT vs `torch.fft.rfft` | rel **2.1e-7**（行数据对） |
| 一平面一 block、32 波串行 | 网格只有 `B*C`，KEEP rFFT 是 `B*C*32` 块并行 |
| 合成 packed vs 两 kernel | 首版 rel ~1（fft_h 段未对齐） |

结论：**No-Go**。就算 fft_h 对齐，串行 32 波也会把占用打掉，很难赢现有两 kernel。不要再试「一个 block 扫完整 256 平面」的融合前向。
