# P5 · skip full pad zero_ on irfft trunc · 2026-07-29

## Change
Reuse stage pad allocated with `zeros` once; later calls only `copy_` modes2 columns (no full `zero_`).

## Result
- multi-call vs CPU: PASS
- formal: 4.641 / 9.598 / 35.427 ms (vs P4 4.614 / 9.599 / 35.768)
- Verdict: **KEEP** (tiny 256 win, no correctness risk observed)
