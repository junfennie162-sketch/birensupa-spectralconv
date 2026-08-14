# Opt vs new standards · SpectralConv dual-corner auto

- time_utc: 2026-07-23T13:16:03Z
- warmup/iters: 10/100 (table C 口径)
- API: spectral_conv2d_supa(..., use_sufft="auto", weights2=...)

| res | auto_ms | device_ms |
|---|---:|---:|
| 64x64 | 12.077 | — |
| 128x128 | 15.969 | — |
| 256x256 | 79.041 | 41.487 |
