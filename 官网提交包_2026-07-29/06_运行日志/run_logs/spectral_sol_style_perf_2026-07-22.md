# SpectralConv SOL-ExecBench-style perf (Biren adapted)

- time_utc: 2026-07-22T13:13:15Z
- reference: https://github.com/nvidia/sol-execbench
- warmup/iters/trials: 10/50/3
- scoring_baseline: team v1 (cpu fft + bridged mul)
- ref_target: handbook PyTorch-on-SUPA narrative ms

| res | v1_med | fused_med | auto_med | speedup_auto | ref_ms | gap_auto | proxy_sol | mem_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64x64 | 13.985 | 37.88 | 14.034 | 0.996 | 5.3 | 8.734 | 0.4972 | 17.0 |
| 128x128 | 18.208 | 40.176 | 18.01 | 1.011 | 8.7 | 9.31 | 0.5104 | 29.0 |
| 256x256 | 287.91 | 88.79 | 88.434 | 3.256 | 27.7 | 60.734 | 0.8833 | 553.8 |

## Notes

- Official contest table in `results.md` still uses §3.2 iters=100 wall clock;
  this log is the rigorous cross-check for optimization decisions (O2).
- Full NVIDIA SOL-Score needs SOLAR bounds on B200; Biren uses proxy only.
