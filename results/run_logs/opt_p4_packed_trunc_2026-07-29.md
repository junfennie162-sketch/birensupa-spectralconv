# P4 · packed trunc spectrum → mul · 2026-07-29

## Change

- `rfft2_sufft_trunc` returns packed `[B,C,H,modes2,2]` (no full-Wf zero/copy).
- Fused path: `out_freq` width = `modes2` when trunc on; R11 gather-scatter uses `Wf=modes2`.
- `irfft2_sufft_trunc` accepts packed or full; C2R still pads to `W/2+1`.

## Correctness fix (critical)

Reusing stage-cached buffers as **C2C/C2R outputs** made trunc irfft wrong from the **2nd call** onward (~1.03 rel vs first/full). P3’s stage-cached spatial out had the same latent risk (single-call accuracy tests would not catch it).

Mitigation: fresh `empty`/`empty_like` for C2C out + C2R out; pad buffer may stay stage-cached with `zero_` each call.

Multi-call fused vs CPU reference: PASS (5×).

## Formal (`test_accuracy` / `test_perf`, auto)

| res | ms | MB |
|-----|---:|---:|
| 64 | 4.625 | 213.3 |
| 128 | 9.587 | 299.3 |
| 256 | 35.760 | 619.3 |

Device-only A/B vs trunc off: about **+16% / +34% / +40%**.

Vs prior P3 table (4.563/9.607/32.024): 64/128 flat; 256 slower but **multi-call correct**; peak memory much lower.

## Verdict

**KEEP**. No ai4s merge this round.
