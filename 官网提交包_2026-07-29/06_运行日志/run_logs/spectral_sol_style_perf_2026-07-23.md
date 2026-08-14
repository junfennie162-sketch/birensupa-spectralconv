# SpectralConv SOL-ExecBench-style perf (Biren adapted)

- time_utc: 2026-07-23T03:58:29Z
- reference: https://github.com/nvidia/sol-execbench
- warmup/iters/trials: 10/50/3
- scoring_baseline: team v1 (cpu fft + bridged mul)
- ref_target: handbook PyTorch-on-SUPA narrative ms

| res | v1_med | fused_med | auto_med | speedup_auto | ref_ms | gap_auto | proxy_sol | mem_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64x64 | 46.001 | 39.757 | 46.107 | 0.998 | 5.3 | 40.807 | 0.4987 | 16.8 |
| 128x128 | 45.949 | 40.949 | 50.002 | 0.919 | 8.7 | 41.302 | 0.4456 | 28.8 |
| 256x256 | 292.03 | 81.75 | 80.862 | 3.611 | 27.7 | 53.162 | 0.8994 | 553.7 |

## Notes

- Official contest table in `results.md` still uses §3.2 iters=100 wall clock;
  this log is the rigorous cross-check for optimization decisions (O2).
- Full NVIDIA SOL-Score needs SOLAR bounds on B200; Biren uses proxy only.
