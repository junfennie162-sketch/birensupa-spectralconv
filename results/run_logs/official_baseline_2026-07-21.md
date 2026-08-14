# Official SpectralConv baseline (PyTorch reference)

- time_utc: 2026-07-21T13:40:14Z
- baseline_device: cpu
- correctness_ok: True
- supa_fft_probe_ok: False

## Correctness (official shapes / forward / backward)
```json
{
  "ok": true,
  "device": "cpu",
  "input_shape": [
    4,
    32,
    128,
    128
  ],
  "output_shape": [
    4,
    64,
    128,
    128
  ],
  "output_min": -0.00282877404242754,
  "output_max": 0.002808363176882267,
  "grad_shape": [
    4,
    32,
    128,
    128
  ],
  "grad_min": 0.014115295372903347,
  "grad_max": 0.017278257757425308,
  "modes": [
    16,
    16
  ]
}
```

## Performance

| 分辨率 | 前向耗时 (ms) | 显存 (MB) |
|---|---:|---:|
| 64x64 | 74.142 | 0.0 |
| 128x128 | 89.000 | 0.0 |
| 256x256 | 295.983 | 0.0 |

## SUPA torch.fft probe
```json
{
  "ok": false,
  "max_abs": 99.8006362915039,
  "rel_vs_cpu_peak": 1.0015693799628278,
  "threshold": 0.0001,
  "note": "If false, do not put full SpectralConv (incl. FFT) on device=supa via torch.fft"
}
```

Artifacts: `/workspace/ai4s-f/submission/results/run_logs/official_baseline_2026-07-21.json`
