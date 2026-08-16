# fused 16-row/256-thread + device-in 探针（2026-08-16）

> 旁注。**未写** formal idle。

- modes16@256 fused vs CPU 1.380e-06 vs keep 0.000e+00
- device-in vs CPU 1.380e-06 vs keep 0.000e+00
- isolated inv two 1.312 / fused 1.283 rel 0.000e+00
- promote {'fused_inv_isolated': True, 'fused_e2e256': True, 'devicein_e2e256': False, 'both_e2e256': True, 'official_ok': True}

| 路径 | 64 | 128 | 256 | 256 repeat |
|------|---:|----:|----:|-----------:|
| keep | 1.065 | 2.234 | 7.312 | 7.320 |
| fused | 1.068 | 2.204 | 7.114 | 7.092 |
| devicein | 1.065 | 2.226 | 7.326 | 7.316 |
| fused+devicein | 1.067 | 2.216 | 7.180 | 7.087 |
