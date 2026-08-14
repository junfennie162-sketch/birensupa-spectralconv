# R13 · host-seeded ping-pong skip D2D · 2026-07-29 · ROLLBACK

## 猜想
层间把激活写进 host-seeded ping-pong，下层 `_roundtrip_supa_input` 见同 `data_ptr` 则跳过 D2D（~0.3 ms/层）。

## 结果
- 墙钟看似快 ~4%，但 chain CPU↔SUPA 一致性 **FAIL**（rel ~1e-2～1e-1）
- **必须回滚**

## 根因（重要反模式）
1. suFFT 只认「从未被 SUDNN Conv2d 当输入」的 host-origin storage。
2. 把 safe 缓冲当成 FNO 层激活返回后，`conv(x)` 会 **永久污染** 该 storage；之后即使 `copy_` 回填，suFFT 谱仍错（rel≈1）。
3. 因此 safe 缓冲必须 **只给 FFT 用**；1×1 conv 必须读原始 device 激活。跳过层间 D2D 与「conv 读同一 buffer」不可兼得。
4. 附带：双流 D2D∥conv / spectral∥conv 均更慢（layer 12.0 vs 11.6；chain 49 vs 48）。

## 对照
| setup | 正确性 |
|-------|--------|
| R7：roundtrip→safe 仅 FFT；conv(原 x) | OK |
| ping-pong 返回 safe 给下层作 x | FAIL |
| conv 先于 FFT 且共享 safe | FAIL（conv 先污染） |
| conv 污染后 copy_ 进同一 safe 再 FFT | FAIL（永久污染） |
| conv 污染后 copy_ 进 **新** host-seeded | OK |

## 代码
已回滚 `model.py` / `spectral_conv_ops.py`；仅在 `_roundtrip_supa_input` 文档中留下警告。
